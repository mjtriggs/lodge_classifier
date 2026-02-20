from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from lodge_classifier.config import PipelineConfig
from lodge_classifier.language.detect import detect_language_strict
from lodge_classifier.normalise import normalise_lodge_name
from lodge_classifier.ontology.classify import classify_ontology_v1
from lodge_classifier.theme.classify import resolve_theme_v1


@dataclass(frozen=True)
class PipelineOutputs:
    """Paths for pipeline outputs."""

    classified_path: Path
    review_queue_path: Path


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_pipeline(
    cfg: PipelineConfig,
    lodge_names_path: Path,
    manual_curation_path: Path | None = None,
) -> PipelineOutputs:
    """Run the lodge naming classification pipeline.

    This version implements:
        - Manual curation merge (if provided)
        - Normalisation
        - Strict Option A language detection
        - Ontology v1 classification (dictionary-driven)
        - Theme v1 resolution (priority rules)
        - CSV outputs (classified + review queue)

    Args:
        cfg: Pipeline configuration.
        lodge_names_path: CSV with at least 'lodge_name_raw' column.
        manual_curation_path: Optional CSV for manual curation layer.

    Returns:
        Paths to the classified output and review queue output.
    """
    run_id = uuid.uuid4().hex[:10]
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting run_id={run_id}", run_id=run_id)
    logger.info("Reading input: {path}", path=str(lodge_names_path))

    df = pd.read_csv(lodge_names_path)
    if "lodge_name_raw" not in df.columns:
        raise ValueError("Input must include 'lodge_name_raw' column")

    df["lodge_name_raw"] = df["lodge_name_raw"].astype(str)

    if manual_curation_path and manual_curation_path.exists():
        logger.info("Merging manual curation: {path}", path=str(manual_curation_path))
        m = pd.read_csv(manual_curation_path)
        if "lodge_name_raw" not in m.columns:
            raise ValueError("Manual curation must include 'lodge_name_raw' column")
        df = df.merge(m, on="lodge_name_raw", how="left")
    else:
        logger.info("No manual curation provided")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = row["lodge_name_raw"]

        curated_lang = row.get("curated_language_override", None)
        curated_ontology_hint = row.get("curated_ontology_hint", None)
        curated_theme_hint = row.get("curated_theme_hint", None)

        norm = normalise_lodge_name(raw)

        # Language (Strict Option A)
        lang_res = detect_language_strict(
            tokens=norm.tokens,
            dicts_dir=cfg.paths.dicts_dir,
            curated_language_override=curated_lang if isinstance(curated_lang, str) else None,
        )

        # Ontology (v1)
        ont_res = classify_ontology_v1(
            tokens=norm.tokens,
            dicts_dir=cfg.paths.dicts_dir,
            language_primary=lang_res.language_primary,
            curated_ontology_hint=(
                curated_ontology_hint if isinstance(curated_ontology_hint, str) else None
            ),
        )

        # Theme (v1)
        theme_res = resolve_theme_v1(
            ontology_primary=ont_res.ontology_primary,
            ontology_secondary=ont_res.ontology_secondary,
            tokens=norm.tokens,
            language_primary=lang_res.language_primary,
            curated_theme_hint=curated_theme_hint if isinstance(curated_theme_hint, str) else None,
        )

        flags = []
        flags.extend(lang_res.flags)
        flags.extend(ont_res.flags)
        flags.extend(theme_res.flags)

        evidence = {
            "run_id": run_id,
            "pipeline_version": cfg.pipeline_version,
            "normalise": {
                "clean": norm.clean,
                "normalised": norm.normalised,
                "tokens": norm.tokens,
            },
            "language": {
                "language_primary": lang_res.language_primary,
                "confidence_language": lang_res.confidence_language,
                "evidence": lang_res.evidence,
            },
            "ontology": {
                "ontology_primary": ont_res.ontology_primary,
                "ontology_secondary": ont_res.ontology_secondary,
                "confidence_ontology": ont_res.confidence_ontology,
                "evidence": ont_res.evidence,
            },
            "theme": {
                "theme_primary": theme_res.theme_primary,
                "theme_secondary": theme_res.theme_secondary,
                "confidence_theme": theme_res.confidence_theme,
                "evidence": theme_res.evidence,
            },
            "manual": {
                "curated_language_override": row.get("curated_language_override", None),
                "curated_ontology_hint": row.get("curated_ontology_hint", None),
                "curated_theme_hint": row.get("curated_theme_hint", None),
                "curated_alias": row.get("curated_alias", None),
                "curated_notes": row.get("curated_notes", None),
            },
        }

        # Simple review logic v1: flag if any layer is weak
        review_required = (
            lang_res.confidence_language < cfg.review_confidence_threshold
            or ont_res.confidence_ontology < 0.60
            or theme_res.confidence_theme < 0.60
        )

        records.append(
            {
                "lodge_name_raw": raw,
                "lodge_name_clean": norm.clean,
                "ontology_primary": ont_res.ontology_primary,
                "ontology_secondary": ont_res.ontology_secondary,
                "theme_primary": theme_res.theme_primary,
                "theme_secondary": theme_res.theme_secondary,
                "language_primary": lang_res.language_primary,
                "confidence_theme": theme_res.confidence_theme,
                "confidence_language": lang_res.confidence_language,
                "flags": "|".join(sorted(set(flags))) if flags else "",
                "evidence_json": json.dumps(evidence, ensure_ascii=False),
                "review_required": review_required,
                "created_at": _utc_now_iso(),
            }
        )

    out = pd.DataFrame.from_records(records)

    classified_path = cfg.paths.outputs_dir / f"classified_{run_id}.csv"
    review_queue_path = cfg.paths.outputs_dir / f"review_queue_{run_id}.csv"

    out.drop(columns=["review_required"]).to_csv(classified_path, index=False)
    out.loc[out["review_required"]].drop(columns=["review_required"]).to_csv(
        review_queue_path, index=False
    )

    logger.info("Wrote classified: {path}", path=str(classified_path))
    logger.info("Wrote review queue: {path}", path=str(review_queue_path))

    return PipelineOutputs(classified_path=classified_path, review_queue_path=review_queue_path)
