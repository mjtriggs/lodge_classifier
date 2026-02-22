from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodge_classifier.dicts.cache import DictCache


@dataclass(frozen=True)
class ThemeResult:
    """Theme classification output for a single lodge name."""

    theme_primary: str
    theme_secondary: str | None
    confidence_theme: float
    flags: list[str]
    evidence: dict[str, Any]


_PRIORITY_ORDER: list[str] = [
    "Virtue / Moral Ideal",
    "Religious",
    "Royal / Aristocratic",
    "Military / Service",
    "Educational / Institutional",
    "Masonic / Administrative",
    "Professional / Trade",
    "Clubs / Association",
    "Mythological / Classical",
    "Geographic / Civic",
    "Nature",
    "Symbolic / Esoteric",
]


def resolve_theme_v1(
    ontology_primary: str,
    ontology_secondary: str | None,
    tokens: list[str],
    language_primary: str,
    dicts_dir: Path,
    curated_theme_hint: str | None = None,
    cache: DictCache | None = None,
) -> ThemeResult:
    """Resolve theme using ontology and token signals."""
    flags: list[str] = []
    token_set = {t.strip().lower() for t in tokens if t and t.strip()}

    evidence: dict[str, Any] = {
        "ontology_primary": ontology_primary,
        "ontology_secondary": ontology_secondary,
        "tokens": tokens,
        "language_primary": language_primary,
    }

    if curated_theme_hint and curated_theme_hint.strip():
        flags.append("OVERRIDE_APPLIED")
        return ThemeResult(
            theme_primary=curated_theme_hint.strip(),
            theme_secondary=None,
            confidence_theme=0.99,
            flags=flags,
            evidence={**evidence, "reason": "curated_theme_hint", "rule_ids": ["THEME_OVERRIDE"]},
        )

    candidates: set[str] = set()

    def add_from_ontology(code: str) -> None:
        if code.startswith("ABS_"):
            candidates.add("Virtue / Moral Ideal")
        if code in {"PRS_REL", "REL_TEM", "LOC_REL"}:
            candidates.add("Religious")
        if code == "PRS_ROY":
            candidates.add("Royal / Aristocratic")
        if code == "GRP_MIL":
            candidates.add("Military / Service")
        if code == "GRP_EDU":
            candidates.add("Educational / Institutional")
        if code == "GRP_MAS":
            candidates.add("Masonic / Administrative")
        if code == "GRP_JOB":
            candidates.add("Professional / Trade")
        if code == "GRP_INT":
            candidates.add("Clubs / Association")
        if code == "PRS_MYTH":
            candidates.add("Mythological / Classical")
        if code.startswith("LOC_"):
            candidates.add("Geographic / Civic")
        if code.startswith("NAT_"):
            candidates.add("Nature")
        if code.startswith("OBJ_"):
            candidates.add("Symbolic / Esoteric")

    add_from_ontology(ontology_primary)
    if ontology_secondary:
        add_from_ontology(ontology_secondary)

    cache = cache or DictCache(dicts_dir=dicts_dir)
    religious_place_terms = cache.load_set("religious_place_terms.csv", column="token")
    religious_hits = sorted(token_set.intersection(religious_place_terms))
    if religious_hits:
        candidates.add("Religious")
        flags.append("TOKEN_RELIGIOUS_PLACE")
        evidence["religious_place_hits"] = religious_hits
        evidence["rule_ids"] = ["THEME_TOKEN_RELIGIOUS_PLACE"]
        evidence["sources"] = ["religious_place_terms.csv"]

    if not candidates:
        candidates.add("Unknown")

    ordered = [t for t in _PRIORITY_ORDER if t in candidates]
    if ordered:
        primary = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else None
    else:
        primary = sorted(candidates)[0]
        secondary = sorted(candidates)[1] if len(candidates) > 1 else None

    confidence = 0.85 if primary != "Unknown" else 0.30

    return ThemeResult(
        theme_primary=primary,
        theme_secondary=secondary,
        confidence_theme=confidence,
        flags=flags,
        evidence={**evidence, "candidates": sorted(candidates), "priority_order": _PRIORITY_ORDER},
    )
