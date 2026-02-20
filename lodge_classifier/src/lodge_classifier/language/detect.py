from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LanguageResult:
    """Language classification output for a single lodge name."""

    language_primary: str
    confidence_language: float
    flags: list[str]
    evidence: dict[str, Any]


def _load_csv_set(path: Path, column: str) -> set[str]:
    """Load a CSV file and return a lowercased set of values from a column."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required dictionary file: {path}")

    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' in {path}")

    return set(df[column].astype(str).str.strip().str.lower())


def _try_load_csv_set(path: Path, column: str) -> set[str]:
    """Load a CSV set if the file exists, otherwise return an empty set.

    This allows optional language dictionaries (e.g. French) without breaking the pipeline.
    """
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' in {path}")

    return set(df[column].astype(str).str.strip().str.lower())


def detect_language_strict(
    tokens: list[str],
    dicts_dir: Path,
    curated_language_override: str | None = None,
) -> LanguageResult:
    """Detect primary language using strict lexical-origin rules (Option A).

    Strict rule:
        - Classical names are classified by origin even if anglicised.
          e.g. "polaris" -> Latin, "zeus" -> Greek.

    Precedence:
        1) curated_language_override (if present and non-empty)
        2) classical lexicon (Latin, Greek) [required dictionaries]
        3) Welsh markers [required dictionary]
        4) French token lexicon [optional dictionary]
        5) fallback English if tokens exist, else Unknown

    Required dictionary files in dicts_dir:
        - classical_latin.csv with column: token
        - classical_greek.csv with column: token
        - welsh_markers.csv with column: marker

    Optional dictionary files in dicts_dir:
        - french_tokens.csv with column: token
    """
    flags: list[str] = []
    evidence: dict[str, Any] = {"tokens": tokens}

    if curated_language_override and curated_language_override.strip():
        flags.append("OVERRIDE_APPLIED")
        return LanguageResult(
            language_primary=curated_language_override.strip(),
            confidence_language=0.99,
            flags=flags,
            evidence={**evidence, "reason": "curated_language_override"},
        )

    latin = _load_csv_set(dicts_dir / "classical_latin.csv", column="token")
    greek = _load_csv_set(dicts_dir / "classical_greek.csv", column="token")
    welsh_markers = _load_csv_set(dicts_dir / "welsh_markers.csv", column="marker")

    # Optional dictionaries
    french_tokens = _try_load_csv_set(dicts_dir / "french_tokens.csv", column="token")
    german_tokens = _try_load_csv_set(dicts_dir / "german_tokens.csv", column="token")
    italian_tokens = _try_load_csv_set(dicts_dir / "italians_tokens.csv", column="token")

    token_set = {t.strip().lower() for t in tokens if t and str(t).strip()}

    # Latin (strict classical)
    latin_hits = sorted(token_set.intersection(latin))
    if latin_hits:
        flags.append("CLASSICAL_OVERRIDE")
        return LanguageResult(
            language_primary="Latin",
            confidence_language=0.95,
            flags=flags,
            evidence={**evidence, "latin_hits": latin_hits},
        )

    # Greek (strict classical)
    greek_hits = sorted(token_set.intersection(greek))
    if greek_hits:
        flags.append("CLASSICAL_OVERRIDE")
        return LanguageResult(
            language_primary="Greek",
            confidence_language=0.95,
            flags=flags,
            evidence={**evidence, "greek_hits": greek_hits},
        )

    # Welsh markers
    welsh_hits = sorted(token_set.intersection(welsh_markers))
    if welsh_hits:
        return LanguageResult(
            language_primary="Welsh",
            confidence_language=0.90,
            flags=flags,
            evidence={**evidence, "welsh_hits": welsh_hits},
        )

    # French lexicon (optional)
    french_hits = sorted(token_set.intersection(french_tokens))
    if french_hits:
        flags.append("DICT_MATCH")
        return LanguageResult(
            language_primary="French",
            confidence_language=0.90,
            flags=flags,
            evidence={**evidence, "french_hits": french_hits},
        )

    # German lexicon (optional)
    german_hits = sorted(token_set.intersection(german_tokens))
    if german_hits:
        flags.append("DICT_MATCH")
        return LanguageResult(
            language_primary="German",
            confidence_language=0.90,
            flags=flags,
            evidence={**evidence, "german_hits": german_hits},
        )

    # Italian lexicon (optional)
    italian_hits = sorted(token_set.intersection(italian_tokens))
    if italian_hits:
        flags.append("DICT_MATCH")
        return LanguageResult(
            language_primary="Italian",
            confidence_language=0.90,
            flags=flags,
            evidence={**evidence, "italian_hits": italian_hits},
        )

    # Fallbacks
    if token_set:
        return LanguageResult(
            language_primary="English",
            confidence_language=0.60,
            flags=flags,
            evidence={**evidence, "reason": "fallback_non_empty"},
        )

    return LanguageResult(
        language_primary="Unknown",
        confidence_language=0.40,
        flags=flags,
        evidence={**evidence, "reason": "no_tokens"},
    )
