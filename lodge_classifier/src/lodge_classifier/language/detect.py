from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodge_classifier.dicts.cache import DictCache
from lodge_classifier.matching.ngrams import build_ngrams


@dataclass(frozen=True)
class LanguageResult:
    """Language classification output for a single lodge name."""

    language_primary: str
    confidence_language: float
    flags: list[str]
    evidence: dict[str, Any]


def _as_nonempty_str(value: Any) -> str | None:
    """Return a stripped string if value is a non-empty string; otherwise None.

    This avoids pandas NaN values (floats) being treated as truthy.
    """
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s or s.lower() == "nan":
        return None

    return s


def detect_language_strict(
    tokens: list[str],
    dicts_dir: Path,
    curated_language_override: str | None = None,
    cache: DictCache | None = None,
) -> LanguageResult:
    """Detect language using strict dictionary rules (Option A), with UK-place boosts.

    Behaviour (precedence):
        1) Manual override (if provided) -> high confidence.
        2) Non-English dictionary evidence (French/German/Italian/etc.) -> high confidence.
        3) UK place evidence via gazetteer dictionaries (cities/towns, regions, landmarks)
           -> English with high confidence (cities/towns highest).
        4) English dictionary evidence -> English with high confidence.
        5) Otherwise -> English fallback with low confidence.

    Notes:
        - UK place matches are checked using n-grams to support multi-word places.
        - UK place evidence is only used when no stronger non-English evidence exists.

    Args:
        tokens: Normalised token list for the lodge name.
        dicts_dir: Directory containing dictionary CSVs.
        curated_language_override: Optional manual override for language.
        cache: Optional DictCache instance for reuse.

    Returns:
        LanguageResult with language_primary, confidence, flags, and evidence.
    """
    flags: list[str] = []
    evidence: dict[str, Any] = {"tokens": tokens}

    curated_language_override = _as_nonempty_str(curated_language_override)
    if curated_language_override:
        flags.append("OVERRIDE_APPLIED")
        return LanguageResult(
            language_primary=curated_language_override,
            confidence_language=0.99,
            flags=flags,
            evidence={
                **evidence,
                "reason": "curated_language_override",
                "rule_ids": ["LANG_OVERRIDE"],
            },
        )

    cache = cache or DictCache(dicts_dir=dicts_dir)
    token_set = {t.lower() for t in tokens if t and t.strip()}
    phrases = build_ngrams(tokens, max_n=5)

    # Language token dictionaries (existing)
    french_tokens = cache.load_set("french_tokens.csv", column="token")
    german_tokens = cache.load_set("german_tokens.csv", column="token")
    italian_tokens = cache.load_set("italian_tokens.csv", column="token")
    latin_tokens = cache.load_set("latin_tokens.csv", column="token")
    greek_tokens = cache.load_set("greek_tokens.csv", column="token")

    # Optional English token dictionary (if you have it)
    # If you don't yet, keep the file small/high-signal as discussed.
    try:
        english_tokens = cache.load_set("english_tokens.csv", column="token")
    except FileNotFoundError:
        english_tokens = set()

    fr_hits = sorted(token_set.intersection(french_tokens))
    de_hits = sorted(token_set.intersection(german_tokens))
    it_hits = sorted(token_set.intersection(italian_tokens))
    la_hits = sorted(token_set.intersection(latin_tokens))
    gr_hits = sorted(token_set.intersection(greek_tokens))
    en_hits = sorted(token_set.intersection(english_tokens)) if english_tokens else []

    evidence.update(
        {
            "french_hits": fr_hits,
            "german_hits": de_hits,
            "italian_hits": it_hits,
            "latin_hits": la_hits,
            "greek_hits": gr_hits,
            "english_hits": en_hits,
        }
    )

    # 1) Non-English wins on any positive evidence (tune thresholds later if needed)
    if fr_hits:
        return LanguageResult(
            language_primary="French",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_FRENCH_DICT"],
                "sources": ["french_tokens.csv"],
            },
        )

    if de_hits:
        return LanguageResult(
            language_primary="German",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_GERMAN_DICT"],
                "sources": ["german_tokens.csv"],
            },
        )

    if it_hits:
        return LanguageResult(
            language_primary="Italian",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_ITALIAN_DICT"],
                "sources": ["italian_tokens.csv"],
            },
        )

    if la_hits:
        return LanguageResult(
            language_primary="Latin",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_LATIN_DICT"],
                "sources": ["latin_tokens.csv"],
            },
        )

    if gr_hits:
        return LanguageResult(
            language_primary="Greek",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_GREEK_DICT"],
                "sources": ["greek_tokens.csv"],
            },
        )

    # 2) UK place evidence boosts English confidence (cities/towns strongest)
    # These files already exist in your dicts folder.
    uk_cities_towns = cache.load_set("cities_and_towns.csv", column="token")
    uk_regions = cache.load_set("regions.csv", column="token")
    uk_landmarks = cache.load_set("landmarks.csv", column="token")

    uk_cty_hits = sorted(phrases.intersection(uk_cities_towns))
    uk_reg_hits = sorted(phrases.intersection(uk_regions))
    uk_lan_hits = sorted(phrases.intersection(uk_landmarks))

    evidence.update(
        {
            "uk_city_town_hits": uk_cty_hits,
            "uk_region_hits": uk_reg_hits,
            "uk_landmark_hits": uk_lan_hits,
        }
    )

    if uk_cty_hits:
        return LanguageResult(
            language_primary="English",
            confidence_language=0.95,
            flags=["DICT_MATCH", "UK_PLACE_BOOST"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_ENGLISH_UK_CITY_TOWN"],
                "sources": ["cities_and_towns.csv"],
                "uk_place_hits": uk_cty_hits,
            },
        )

    if uk_reg_hits:
        return LanguageResult(
            language_primary="English",
            confidence_language=0.90,
            flags=["DICT_MATCH", "UK_PLACE_BOOST"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_ENGLISH_UK_REGION"],
                "sources": ["regions.csv"],
                "uk_place_hits": uk_reg_hits,
            },
        )

    if uk_lan_hits:
        return LanguageResult(
            language_primary="English",
            confidence_language=0.88,
            flags=["DICT_MATCH", "UK_PLACE_BOOST"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_ENGLISH_UK_LANDMARK"],
                "sources": ["landmarks.csv"],
                "uk_place_hits": uk_lan_hits,
            },
        )

    # 3) English dictionary evidence (if present)
    if en_hits:
        return LanguageResult(
            language_primary="English",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_ENGLISH_DICT"],
                "sources": ["english_tokens.csv"],
            },
        )

    # 4) Fallback English with low confidence
    flags.append("ENGLISH_FALLBACK_LOW_CONF")
    return LanguageResult(
        language_primary="English",
        confidence_language=0.55,
        flags=flags,
        evidence={**evidence, "rule_ids": ["LANG_ENGLISH_FALLBACK"]},
    )
