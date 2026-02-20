from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OntologyResult:
    """Ontology classification output for a single lodge name."""

    ontology_primary: str
    ontology_secondary: str | None
    confidence_ontology: float
    flags: list[str]
    evidence: dict[str, Any]


def _load_set(path: Path, column: str = "token") -> set[str]:
    """Load a lowercased token set from a CSV dictionary file."""
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' in {path}")
    return set(df[column].astype(str).str.strip().str.lower())


def classify_ontology_v1(
    tokens: list[str],
    dicts_dir: Path,
    language_primary: str | None = None,
    curated_ontology_hint: str | None = None,
) -> OntologyResult:
    """Classify ontology using deterministic dictionary rules (v1).

    Primary ontology is mutually exclusive. Secondary ontology captures a strong
    co-signal (e.g. GRP_EDU for "University", or LOC_BDG for "Fort").

    Precedence:
        1) curated_ontology_hint (if provided)
        2) High-signal dictionaries for primary ontology
        3) Secondary ontology capture for strong co-signals
        4) Strict fallback: Latin single-token -> ABS_LAT (unless already matched)
        5) Fallback UNK_

    Args:
        tokens: Normalised tokens (excluding "lodge").
        dicts_dir: Directory containing dictionary CSV files.
        language_primary: Detected primary language (Strict Option A).
        curated_ontology_hint: Optional forced ontology from manual curation layer.

    Returns:
        OntologyResult with primary/secondary, confidence, flags, and evidence.
    """
    flags: list[str] = []
    evidence: dict[str, Any] = {"tokens": tokens, "language_primary": language_primary}

    if curated_ontology_hint and curated_ontology_hint.strip():
        flags.append("OVERRIDE_APPLIED")
        return OntologyResult(
            ontology_primary=curated_ontology_hint.strip(),
            ontology_secondary=None,
            confidence_ontology=0.99,
            flags=flags,
            evidence={**evidence, "reason": "curated_ontology_hint"},
        )

    token_set = {t.lower() for t in tokens}

    # Dictionaries
    religious_person_terms = _load_set(dicts_dir / "religious_person_terms.csv")
    religious_place_terms = _load_set(dicts_dir / "religious_place_terms.csv")
    royal_titles = _load_set(dicts_dir / "royal_titles.csv")
    military_terms = _load_set(dicts_dir / "military_terms.csv")
    masonic_terms = _load_set(dicts_dir / "masonic_terms.csv")
    virtues = _load_set(dicts_dir / "virtues.csv")
    fraternal_terms = _load_set(dicts_dir / "fraternal_terms.csv")
    philosophical_terms = _load_set(dicts_dir / "philosophical_terms.csv")
    botanical = _load_set(dicts_dir / "botanical.csv")
    animals = _load_set(dicts_dir / "animals.csv")
    astronomical = _load_set(dicts_dir / "astronomical.csv")

    # Composite secondary signals
    edu_terms = _load_set(dicts_dir / "edu_terms.csv")
    myth_terms = _load_set(dicts_dir / "myth_terms.csv")
    building_terms = _load_set(dicts_dir / "building_terms.csv")

    # Prevent double-counting
    building_terms = building_terms.difference(religious_place_terms)

    edu_hits = sorted(token_set.intersection(edu_terms))
    bdg_hits = sorted(token_set.intersection(building_terms))

    # Secondary ontology selection (single slot)
    # Priority: education first (for "X University"), then civic building.
    ontology_secondary: str | None = None
    if edu_hits:
        ontology_secondary = "GRP_EDU"
    elif bdg_hits:
        ontology_secondary = "LOC_BDG"

    # PRS_REL
    rel_hits = sorted(token_set.intersection(religious_person_terms))
    if rel_hits:
        return OntologyResult(
            ontology_primary="PRS_REL",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.90,
            flags=flags,
            evidence={
                **evidence,
                "rule": "prs_rel_terms",
                "hits": rel_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
            },
        )

    # PRS_ROY
    roy_hits = sorted(token_set.intersection(royal_titles))
    if roy_hits:
        return OntologyResult(
            ontology_primary="PRS_ROY",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.88,
            flags=flags,
            evidence={
                **evidence,
                "rule": "prs_roy_titles",
                "hits": roy_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
            },
        )

    # PRS_MYTH
    myth_hits = sorted(token_set.intersection(myth_terms))
    if myth_hits:
        return OntologyResult(
            ontology_primary="PRS_MYTH",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.88,
            flags=flags,
            evidence={
                **evidence,
                "rule": "prs_myth_terms",
                "hits": myth_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
            },
        )

    # GRP_MIL
    mil_hits = sorted(token_set.intersection(military_terms))
    if mil_hits:
        return OntologyResult(
            ontology_primary="GRP_MIL",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.90,
            flags=flags,
            evidence={
                **evidence,
                "rule": "grp_mil_terms",
                "hits": mil_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
            },
        )

    # LOC_REL
    rel_hits = sorted(token_set.intersection(religious_place_terms))
    if rel_hits:
        return OntologyResult(
            ontology_primary="LOC_REL",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.88,
            flags=flags,
            evidence={
                **evidence,
                "rule": "loc_rel_terms",
                "hits": rel_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
            },
        )

    # GRP_MAS
    mas_hits = sorted(token_set.intersection(masonic_terms))
    if mas_hits:
        return OntologyResult(
            ontology_primary="GRP_MAS",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.88,
            flags=flags,
            evidence={
                **evidence,
                "rule": "grp_mas_terms",
                "hits": mas_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
            },
        )

    # GRP_EDU (primary only if nothing else matched)
    if edu_hits:
        return OntologyResult(
            ontology_primary="GRP_EDU",
            ontology_secondary=None,
            confidence_ontology=0.82,
            flags=flags,
            evidence={**evidence, "rule": "grp_edu_terms", "hits": edu_hits},
        )

    # ABS_VRT
    vrt_hits = sorted(token_set.intersection(virtues))
    if vrt_hits:
        return OntologyResult(
            ontology_primary="ABS_VRT",
            ontology_secondary=None,
            confidence_ontology=0.85,
            flags=flags,
            evidence={**evidence, "rule": "abs_vrt_terms", "hits": vrt_hits},
        )

    # ABS_FRAT
    frat_hits = sorted(token_set.intersection(fraternal_terms))
    if frat_hits:
        return OntologyResult(
            ontology_primary="ABS_FRAT",
            ontology_secondary=None,
            confidence_ontology=0.80,
            flags=flags,
            evidence={**evidence, "rule": "abs_frat_terms", "hits": frat_hits},
        )

    # ABS_PHI
    phi_hits = sorted(token_set.intersection(philosophical_terms))
    if phi_hits:
        return OntologyResult(
            ontology_primary="ABS_PHI",
            ontology_secondary=None,
            confidence_ontology=0.80,
            flags=flags,
            evidence={**evidence, "rule": "abs_phi_terms", "hits": phi_hits},
        )

    # NAT_BOT
    bot_hits = sorted(token_set.intersection(botanical))
    if bot_hits:
        return OntologyResult(
            ontology_primary="NAT_BOT",
            ontology_secondary=None,
            confidence_ontology=0.75,
            flags=flags,
            evidence={**evidence, "rule": "nat_bot_terms", "hits": bot_hits},
        )

    # NAT_ANI
    ani_hits = sorted(token_set.intersection(animals))
    if ani_hits:
        return OntologyResult(
            ontology_primary="NAT_ANI",
            ontology_secondary=None,
            confidence_ontology=0.75,
            flags=flags,
            evidence={**evidence, "rule": "nat_ani_terms", "hits": ani_hits},
        )

    # NAT_AST
    ast_hits = sorted(token_set.intersection(astronomical))
    if ast_hits:
        return OntologyResult(
            ontology_primary="NAT_AST",
            ontology_secondary=None,
            confidence_ontology=0.75,
            flags=flags,
            evidence={**evidence, "rule": "nat_ast_terms", "hits": ast_hits},
        )

    # Strict fallback: Latin single-token -> ABS_LAT
    stopwords = {"of", "the", "and"}
    meaningful = [t for t in tokens if t.lower() not in stopwords]

    if (language_primary == "Latin") and len(meaningful) == 1:
        flags.append("STRICT_LATIN_FALLBACK")
        return OntologyResult(
            ontology_primary="ABS_LAT",
            ontology_secondary=None,
            confidence_ontology=0.70,
            flags=flags,
            evidence={**evidence, "rule": "abs_lat_single_token", "meaningful": meaningful},
        )

    return OntologyResult(
        ontology_primary="UNK_",
        ontology_secondary=None,
        confidence_ontology=0.20,
        flags=["REVIEW_REQUIRED"],
        evidence={**evidence, "rule": "fallback"},
    )
