from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from loguru import logger

from lodge_classifier.config import PipelineConfig
from lodge_classifier.dicts.cache import DictCache
from lodge_classifier.language.detect import detect_language_strict
from lodge_classifier.normalise import normalise_lodge_name
from lodge_classifier.ontology.classify import classify_ontology_v1
from lodge_classifier.theme.classify import resolve_theme_v1


@dataclass(frozen=True)
class PipelineOutputs:
    """Paths for pipeline outputs."""

    classified_path: Path
    review_queue_path: Path
    manifest_path: Path


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 hash for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dicts_dir(dicts_dir: Path) -> dict[str, str]:
    """Hash all CSV files in a dicts directory.

    Returns:
        Mapping of filename to SHA-256 digest.
    """
    hashes: dict[str, str] = {}
    if not dicts_dir.exists():
        return hashes

    for p in sorted(dicts_dir.glob("*.csv")):
        hashes[p.name] = _file_sha256(p)
    return hashes


def _summarise_counts(out: pd.DataFrame) -> dict[str, Any]:
    """Create simple run summary stats for the manifest."""

    def _top_counts(col: str, n: int = 25) -> list[dict[str, Any]]:
        vc = out[col].fillna("").value_counts(dropna=False).head(n)
        return [{"value": k, "count": int(v)} for k, v in vc.items() if k != ""]

    return {
        "rows": int(len(out)),
        "review_required_rows": (
            int(out["review_required"].sum()) if "review_required" in out else 0
        ),
        "top_ontology_primary": _top_counts("ontology_primary"),
        "top_theme_primary": _top_counts("theme_primary"),
        "top_language_primary": _top_counts("language_primary"),
        "top_review_reason": _top_counts("review_reason"),
    }


def _review_reason(
    confidence_language: float,
    confidence_ontology: float,
    confidence_theme: float,
    ontology_primary: str,
    language_primary: str,
    flags: Iterable[str],
    cfg: PipelineConfig,
) -> tuple[bool, str, str]:
    """Determine whether an item requires review and provide a reason."""
    flags_set = set(flags)

    if confidence_language < cfg.review_confidence_threshold:
        return True, "LOW_LANGUAGE_CONFIDENCE", "Language confidence below threshold"

    if confidence_ontology < 0.60:
        if ontology_primary == "UNK_":
            return True, "ONTOLOGY_UNKNOWN", "Ontology fell back to UNK_"
        return True, "LOW_ONTOLOGY_CONFIDENCE", "Ontology confidence below threshold"

    if confidence_theme < 0.60:
        return True, "LOW_THEME_CONFIDENCE", "Theme confidence below threshold"

    ambiguous = [f for f in flags_set if "AMBIG" in f or "CONFLICT" in f]
    if ambiguous:
        return (
            True,
            "AMBIGUOUS_SIGNALS",
            "Ambiguous/conflicting signals: " + ",".join(sorted(ambiguous)),
        )

    if language_primary == "Unknown":
        return True, "LANGUAGE_UNKNOWN", "Language classified as Unknown"

    return False, "", ""


def _as_nonempty_str(value: Any) -> str | None:
    """Return a stripped string if value is a non-empty string; otherwise None.

    This avoids pandas NaN values (floats) being treated as truthy, which can
    incorrectly trigger manual-hint logic.
    """
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Defensive: some pipelines end up with literal "nan"
    if s.lower() == "nan":
        return None

    return s


def _make_join_key(raw_name: Any) -> str:
    """Create a robust join key for manual curation merges.

    This key is intended to be resilient to common formatting differences between
    the input file and the manual curation file, such as:
    - casing differences
    - multiple spaces or non-breaking spaces
    - curly apostrophes

    Parameters
    ----------
    raw_name : Any
        Raw lodge name from input or manual curation file.

    Returns
    -------
    str
        Canonical join key.
    """
    if raw_name is None:
        return ""

    s = str(raw_name)
    if not s.strip():
        return ""

    clean = normalise_lodge_name(s).clean
    clean = clean.replace("\u00a0", " ")  # non-breaking space
    clean = clean.replace("’", "'").replace("`", "'")
    clean = " ".join(clean.split())  # collapse internal whitespace
    return clean.lower().strip()


def _is_blank_like(value: Any) -> bool:
    """Return True if the value should be treated as a blank input.

    This includes:
    - None
    - NaN/NA values
    - empty/whitespace-only strings
    - common string placeholders: "nan", "null", "none"
    """
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        # pd.isna can raise on some exotic objects; treat those as not blank here.
        pass

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return True
        if s.lower() in {"nan", "null", "none"}:
            return True

    return False


_NON_DESCRIPTIVE_STOPWORDS = {
    "lodge",
    "loge",
    "no",
    "n",
    "nr",
    "number",
    "numero",
    "num",
    "nº",
    "n°",
    "provincial",
    "prov",
    "district",
    "distr",
}


_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$", flags=re.IGNORECASE)


def _is_non_descriptive_tokens(tokens: list[str]) -> bool:
    """Return True if tokens contain no meaningful description.

    Intended to catch cases like:
    - "lodge no. 2"
    - "no. 2 provincial"
    - "loge n° 17"

    Approach:
    - remove boilerplate tokens (lodge/no/number/etc.)
    - if the remainder is empty, or only numbers/ordinals -> non-descriptive
    """
    if not tokens:
        return True

    stripped: list[str] = []
    for t in tokens:
        tt = t.strip().lower()
        if not tt:
            continue
        if tt in _NON_DESCRIPTIVE_STOPWORDS:
            continue
        stripped.append(tt)

    if not stripped:
        return True

    def _is_numeric_or_ordinal(tok: str) -> bool:
        return tok.isdigit() or bool(_ORDINAL_RE.match(tok))

    return all(_is_numeric_or_ordinal(t) for t in stripped)


def run_pipeline(
    cfg: PipelineConfig,
    lodge_names_path: Path,
    manual_curation_path: Path | None = None,
) -> PipelineOutputs:
    """Run the lodge naming classification pipeline."""
    run_id = uuid.uuid4().hex[:10]
    cfg.paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting run_id={run_id}", run_id=run_id)
    logger.info("Reading input: {path}", path=str(lodge_names_path))

    df = pd.read_csv(lodge_names_path)

    required_cols = {"new_id", "lodge_name_raw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input must include columns: {sorted(missing)}")

    # Keep as-is for now; we handle NaN/"nan"/whitespace robustly later.
    df["lodge_name_raw"] = df["lodge_name_raw"].astype(str)

    if manual_curation_path and manual_curation_path.exists():
        logger.info(
            "Merging manual curation (normalised join): {path}",
            path=str(manual_curation_path),
        )
        m = pd.read_csv(manual_curation_path)

        if "lodge_name_raw" not in m.columns:
            raise ValueError("Manual curation must include 'lodge_name_raw' column")

        df["_join_key"] = df["lodge_name_raw"].apply(_make_join_key)
        m["_join_key"] = m["lodge_name_raw"].apply(_make_join_key)

        # Guardrail: duplicates after normalisation cause ambiguous merges
        dup = m[m["_join_key"].ne("") & m["_join_key"].duplicated(keep=False)]
        if not dup.empty:
            examples = dup["_join_key"].value_counts().head(10).to_dict()
            raise ValueError(
                "Manual curation contains duplicate entries after normalisation. "
                f"Examples: {examples}"
            )

        df = df.merge(m.drop(columns=["lodge_name_raw"]), on="_join_key", how="left")

        # Useful warning: manual entries that match nothing
        matched_keys = set(df.loc[df["curated_notes"].notna(), "_join_key"].unique())
        manual_keys = set(m.loc[m["_join_key"].ne(""), "_join_key"].unique())
        unmatched = manual_keys - matched_keys
        if unmatched:
            logger.warning(
                "Some manual curation rows did not match any input lodge names after "
                "normalisation. Unmatched count={n}.",
                n=len(unmatched),
            )

        df = df.drop(columns=["_join_key"])
    else:
        logger.info("No manual curation provided")

    dict_hashes = _hash_dicts_dir(cfg.paths.dicts_dir)
    cache = DictCache(dicts_dir=cfg.paths.dicts_dir)

    records: list[dict[str, Any]] = []

    review_true_count = 0
    manual_priority_count = 0
    manual_hint_count = 0

    for row in df.itertuples(index=False):
        id_ = getattr(row, "new_id")
        raw = getattr(row, "lodge_name_raw")

        curated_lang = _as_nonempty_str(getattr(row, "curated_language_override", None))
        curated_ontology_hint = _as_nonempty_str(getattr(row, "curated_ontology_hint", None))
        curated_theme_hint = _as_nonempty_str(getattr(row, "curated_theme_hint", None))
        curated_priority_flag = getattr(row, "curated_priority_flag", None)

        is_manual_priority = False
        if curated_priority_flag is not None and str(curated_priority_flag).strip() != "":
            v = str(curated_priority_flag).strip().lower()
            is_manual_priority = v in {"1", "true", "t", "yes", "y"}

        norm = normalise_lodge_name(raw)

        has_any_manual_hint = bool(curated_lang or curated_ontology_hint or curated_theme_hint)

        # Pipeline-level defaulting for blank / non-descriptive names.
        # Manual hints override this by design.
        default_reason: str | None = None
        if not has_any_manual_hint:
            if _is_blank_like(raw) or _is_blank_like(norm.clean) or not norm.tokens:
                default_reason = "INPUT_BLANK"
            elif _is_non_descriptive_tokens(norm.tokens):
                default_reason = "INPUT_NON_DESCRIPTIVE"

        if default_reason is not None:
            language_primary = "English"
            confidence_language = 1.0

            ontology_primary = "UNK_"
            ontology_secondary = None
            confidence_ontology = 1.0

            theme_primary = None
            theme_secondary = None
            confidence_theme = 1.0

            flags_sorted = sorted(
                {
                    default_reason,
                    "CLASSIFICATION_DEFAULTED",
                }
            )

            review_required, review_reason, review_notes = _review_reason(
                confidence_language=confidence_language,
                confidence_ontology=confidence_ontology,
                confidence_theme=confidence_theme,
                ontology_primary=str(ontology_primary),
                language_primary=str(language_primary),
                flags=flags_sorted,
                cfg=cfg,
            )

            if is_manual_priority:
                flags_sorted = sorted(set(flags_sorted) | {"MANUAL_PRIORITY"})
                review_required = False
                review_reason = ""
                review_notes = "Manual priority override"
                manual_priority_count += 1

            evidence = {
                "run_id": run_id,
                "pipeline_version": cfg.pipeline_version,
                "dict_hashes": dict_hashes,
                "normalise": {
                    "clean": norm.clean,
                    "normalised": norm.normalised,
                    "tokens": norm.tokens,
                },
                "language": {
                    "language_primary": language_primary,
                    "confidence_language": confidence_language,
                    "evidence": {"defaulted": True, "reason": default_reason},
                },
                "ontology": {
                    "ontology_primary": ontology_primary,
                    "ontology_secondary": ontology_secondary,
                    "confidence_ontology": confidence_ontology,
                    "evidence": {"defaulted": True, "reason": default_reason},
                },
                "theme": {
                    "theme_primary": theme_primary,
                    "theme_secondary": theme_secondary,
                    "confidence_theme": confidence_theme,
                    "evidence": {"defaulted": True, "reason": default_reason},
                },
                "manual": {
                    "curated_language_override": getattr(row, "curated_language_override", None),
                    "curated_ontology_hint": getattr(row, "curated_ontology_hint", None),
                    "curated_theme_hint": getattr(row, "curated_theme_hint", None),
                    "curated_alias": getattr(row, "curated_alias", None),
                    "curated_notes": getattr(row, "curated_notes", None),
                    "curated_priority_flag": curated_priority_flag,
                    "is_manual_priority": is_manual_priority,
                },
                "review": {
                    "review_required": review_required,
                    "review_reason": review_reason,
                    "review_notes": review_notes,
                },
            }

            if review_required:
                review_true_count += 1

            records.append(
                {
                    "id": id_,
                    "lodge_name_raw": raw,
                    "lodge_name_clean": norm.clean,
                    "language_primary": language_primary,
                    "confidence_language": confidence_language,
                    "ontology_primary": ontology_primary,
                    "ontology_secondary": ontology_secondary,
                    "confidence_ontology": confidence_ontology,
                    "theme_primary": theme_primary,
                    "theme_secondary": theme_secondary,
                    "confidence_theme": confidence_theme,
                    "flags": "|".join(flags_sorted) if flags_sorted else "",
                    "review_required": bool(review_required),
                    "review_reason": review_reason,
                    "review_notes": review_notes,
                    "evidence_json": json.dumps(evidence, ensure_ascii=False),
                    "created_at": _utc_now_iso(),
                }
            )
            continue

        # Normal path: call classifiers.
        lang_res = detect_language_strict(
            tokens=norm.tokens,
            dicts_dir=cfg.paths.dicts_dir,
            curated_language_override=curated_lang,
            cache=cache,
        )

        ont_res = classify_ontology_v1(
            tokens=norm.tokens,
            dicts_dir=cfg.paths.dicts_dir,
            language_primary=lang_res.language_primary,
            curated_ontology_hint=curated_ontology_hint,
            cache=cache,
        )

        theme_res = resolve_theme_v1(
            ontology_primary=ont_res.ontology_primary,
            ontology_secondary=ont_res.ontology_secondary,
            tokens=norm.tokens,
            language_primary=lang_res.language_primary,
            dicts_dir=cfg.paths.dicts_dir,
            curated_theme_hint=curated_theme_hint,
            cache=cache,
        )

        flags: list[str] = []
        flags.extend(lang_res.flags)
        flags.extend(ont_res.flags)
        flags.extend(theme_res.flags)

        if curated_lang or curated_ontology_hint or curated_theme_hint:
            flags.append("MANUAL_HINT_APPLIED")

        if is_manual_priority:
            flags.append("MANUAL_PRIORITY")

        flags_sorted = sorted(set(flags))

        review_required, review_reason, review_notes = _review_reason(
            confidence_language=float(lang_res.confidence_language),
            confidence_ontology=float(ont_res.confidence_ontology),
            confidence_theme=float(theme_res.confidence_theme),
            ontology_primary=str(ont_res.ontology_primary),
            language_primary=str(lang_res.language_primary),
            flags=flags_sorted,
            cfg=cfg,
        )

        if curated_lang or curated_ontology_hint or curated_theme_hint:
            manual_hint_count += 1
        if is_manual_priority:
            manual_priority_count += 1
        if review_required:
            review_true_count += 1

        if is_manual_priority:
            review_required = False
            review_reason = ""
            review_notes = "Manual priority override"

        evidence = {
            "run_id": run_id,
            "pipeline_version": cfg.pipeline_version,
            "dict_hashes": dict_hashes,
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
                "curated_language_override": getattr(row, "curated_language_override", None),
                "curated_ontology_hint": getattr(row, "curated_ontology_hint", None),
                "curated_theme_hint": getattr(row, "curated_theme_hint", None),
                "curated_alias": getattr(row, "curated_alias", None),
                "curated_notes": getattr(row, "curated_notes", None),
                "curated_priority_flag": curated_priority_flag,
                "is_manual_priority": is_manual_priority,
            },
            "review": {
                "review_required": review_required,
                "review_reason": review_reason,
                "review_notes": review_notes,
            },
        }

        records.append(
            {
                "id": id_,
                "lodge_name_raw": raw,
                "lodge_name_clean": norm.clean,
                "language_primary": lang_res.language_primary,
                "confidence_language": float(lang_res.confidence_language),
                "ontology_primary": ont_res.ontology_primary,
                "ontology_secondary": ont_res.ontology_secondary,
                "confidence_ontology": float(ont_res.confidence_ontology),
                "theme_primary": theme_res.theme_primary,
                "theme_secondary": theme_res.theme_secondary,
                "confidence_theme": float(theme_res.confidence_theme),
                "flags": "|".join(flags_sorted) if flags_sorted else "",
                "review_required": bool(review_required),
                "review_reason": review_reason,
                "review_notes": review_notes,
                "evidence_json": json.dumps(evidence, ensure_ascii=False),
                "created_at": _utc_now_iso(),
            }
        )

    out = pd.DataFrame.from_records(records)

    classified_path = cfg.paths.outputs_dir / f"classified_{run_id}.csv"
    review_queue_path = cfg.paths.outputs_dir / f"review_queue_{run_id}.csv"
    manifest_path = cfg.paths.outputs_dir / f"run_manifest_{run_id}.json"

    out.drop(columns=["review_required"]).to_csv(classified_path, index=False)
    out.loc[out["review_required"]].drop(columns=["review_required"]).to_csv(
        review_queue_path, index=False
    )

    manifest = {
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "pipeline_version": cfg.pipeline_version,
        "input_path": str(lodge_names_path),
        "manual_curation_path": str(manual_curation_path) if manual_curation_path else None,
        "row_count_input": int(len(df)),
        "dicts_dir": str(cfg.paths.dicts_dir),
        "dict_hashes": dict_hashes,
        "review_confidence_threshold_language": float(cfg.review_confidence_threshold),
        "summary": _summarise_counts(out),
        "cache_meta": cache.meta(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Run summary: rows={rows} review_required={rev} manual_hints={mh} manual_priority={mp}",
        rows=len(records),
        rev=review_true_count,
        mh=manual_hint_count,
        mp=manual_priority_count,
    )

    logger.info("Wrote classified: {path}", path=str(classified_path))
    logger.info("Wrote review queue: {path}", path=str(review_queue_path))
    logger.info("Wrote manifest: {path}", path=str(manifest_path))

    return PipelineOutputs(
        classified_path=classified_path,
        review_queue_path=review_queue_path,
        manifest_path=manifest_path,
    )
