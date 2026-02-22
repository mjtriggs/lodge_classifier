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
    """Detect language using strict dictionary rules (Option A), with UK-place and person boosts.

    Behaviour (precedence):
        1) Manual override (if provided) -> high confidence.
        2) Non-English dictionary evidence (French/German/Italian/etc.) -> high confidence.
        3) UK place evidence via gazetteer dictionaries (cities/towns, regions, landmarks)
           -> English with high confidence (cities/towns highest).
        4) English dictionary evidence -> English with high confidence.
        5) Otherwise -> English fallback with low confidence.

    Enhancements:
        - If a recognised person is present (people.csv), boost English confidence for (3)-(5).
        - Person evidence never overrides non-English evidence.

    Notes:
        - UK place matches are checked using n-grams to support multi-word places.
        - Person matches are checked using n-grams first, then token fallback.

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

    # --- Person dictionary (used as an English confidence boost only) ---
    # Phrase-first for multi-word names; token fallback.
    # NOTE: if your file is actually named people.csv, change "people.csv" below.
    try:
        person_terms = cache.load_set("prs/people.csv", column="token")
    except FileNotFoundError:
        person_terms = set()

    person_hits = sorted(phrases.intersection(person_terms)) if person_terms else []
    if not person_hits and person_terms:
        person_hits = sorted(token_set.intersection(person_terms))

    # Language token dictionaries (existing)
    french_tokens = cache.load_set("language/french_tokens.csv", column="token")
    german_tokens = cache.load_set("language/german_tokens.csv", column="token")
    italian_tokens = cache.load_set("language/italian_tokens.csv", column="token")
    latin_tokens = cache.load_set("language/latin_tokens.csv", column="token")
    greek_tokens = cache.load_set("language/greek_tokens.csv", column="token")
    maori_tokens = cache.load_set("language/maori_tokens.csv", column="token")

    # Optional English token dictionary (if you have it)
    try:
        english_tokens = cache.load_set("language/english_tokens.csv", column="token")
    except FileNotFoundError:
        english_tokens = set()

    fr_hits = sorted(token_set.intersection(french_tokens))
    de_hits = sorted(token_set.intersection(german_tokens))
    it_hits = sorted(token_set.intersection(italian_tokens))
    la_hits = sorted(token_set.intersection(latin_tokens))
    gr_hits = sorted(token_set.intersection(greek_tokens))
    ma_hits = sorted(token_set.intersection(maori_tokens))
    en_hits = sorted(token_set.intersection(english_tokens)) if english_tokens else []

    evidence.update(
        {
            "french_hits": fr_hits,
            "german_hits": de_hits,
            "italian_hits": it_hits,
            "latin_hits": la_hits,
            "greek_hits": gr_hits,
            "maori_hits": ma_hits,
            "english_hits": en_hits,
            "person_hits": person_hits,
        }
    )

    # 1) Non-English wins on any positive evidence
    if fr_hits:
        return LanguageResult(
            language_primary="French",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_FRENCH_DICT"],
                "sources": ["language/french_tokens.csv"],
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
                "sources": ["language/german_tokens.csv"],
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
                "sources": ["language/italian_tokens.csv"],
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
                "sources": ["language/latin_tokens.csv"],
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
                "sources": ["language/greek_tokens.csv"],
            },
        )

    if ma_hits:
        return LanguageResult(
            language_primary="Maori",
            confidence_language=0.90,
            flags=["DICT_MATCH"],
            evidence={
                **evidence,
                "rule_ids": ["LANG_MAORI_DICT"],
                "sources": ["language/maori_tokens.csv"],
            },
        )

    # 2) UK place evidence boosts English confidence (cities/towns strongest)
    uk_cities_towns = cache.load_set("loc/cities_and_towns.csv", column="token")
    uk_regions = cache.load_set("loc/regions.csv", column="token")
    uk_landmarks = cache.load_set("loc/landmarks.csv", column="token")

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
        conf = 0.95
        out_flags = ["DICT_MATCH", "UK_PLACE_BOOST"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_UK_CITY_TOWN"],
            "sources": ["loc/cities_and_towns.csv"],
            "uk_place_hits": uk_cty_hits,
        }

        if person_hits:
            out_flags.append("PERSON_BOOST")
            conf_before = conf
            conf = min(0.99, conf + 0.02)
            out_evidence = {
                **out_evidence,
                "rule_ids": [*out_evidence["rule_ids"], "LANG_ENGLISH_PERSON_BOOST"],
                "sources": [*out_evidence["sources"], "people.csv"],
                "person_boost": {"before": conf_before, "after": conf, "boost": 0.02, "cap": 0.99},
            }

        return LanguageResult(
            language_primary="English",
            confidence_language=conf,
            flags=out_flags,
            evidence=out_evidence,
        )

    if uk_reg_hits:
        conf = 0.90
        out_flags = ["DICT_MATCH", "UK_PLACE_BOOST"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_UK_REGION"],
            "sources": ["loc/regions.csv"],
            "uk_place_hits": uk_reg_hits,
        }

        if person_hits:
            out_flags.append("PERSON_BOOST")
            conf_before = conf
            conf = min(0.97, conf + 0.03)
            out_evidence = {
                **out_evidence,
                "rule_ids": [*out_evidence["rule_ids"], "LANG_ENGLISH_PERSON_BOOST"],
                "sources": [*out_evidence["sources"], "people.csv"],
                "person_boost": {"before": conf_before, "after": conf, "boost": 0.03, "cap": 0.97},
            }

        return LanguageResult(
            language_primary="English",
            confidence_language=conf,
            flags=out_flags,
            evidence=out_evidence,
        )

    if uk_lan_hits:
        conf = 0.88
        out_flags = ["DICT_MATCH", "UK_PLACE_BOOST"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_UK_LANDMARK"],
            "sources": ["loc/landmarks.csv"],
            "uk_place_hits": uk_lan_hits,
        }

        if person_hits:
            out_flags.append("PERSON_BOOST")
            conf_before = conf
            conf = min(0.95, conf + 0.04)
            out_evidence = {
                **out_evidence,
                "rule_ids": [*out_evidence["rule_ids"], "LANG_ENGLISH_PERSON_BOOST"],
                "sources": [*out_evidence["sources"], "people.csv"],
                "person_boost": {"before": conf_before, "after": conf, "boost": 0.04, "cap": 0.95},
            }

        return LanguageResult(
            language_primary="English",
            confidence_language=conf,
            flags=out_flags,
            evidence=out_evidence,
        )

    # 3) English dictionary evidence (if present)
    if en_hits:
        conf = 0.90
        out_flags = ["DICT_MATCH"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_DICT"],
            "sources": ["language/english_tokens.csv"],
        }

        if person_hits:
            out_flags.append("PERSON_BOOST")
            conf_before = conf
            conf = min(0.97, conf + 0.05)
            out_evidence = {
                **out_evidence,
                "rule_ids": [*out_evidence["rule_ids"], "LANG_ENGLISH_PERSON_BOOST"],
                "sources": [*out_evidence["sources"], "people.csv"],
                "person_boost": {"before": conf_before, "after": conf, "boost": 0.05, "cap": 0.97},
            }

        return LanguageResult(
            language_primary="English",
            confidence_language=conf,
            flags=out_flags,
            evidence=out_evidence,
        )

    # 4) Fallback English with low confidence (+ person boost if present)
    flags.append("ENGLISH_FALLBACK_LOW_CONF")
    conf = 0.55
    out_evidence = {**evidence, "rule_ids": ["LANG_ENGLISH_FALLBACK"]}

    if person_hits:
        flags.append("PERSON_BOOST")
        conf_before = conf
        conf = min(0.70, conf + 0.10)
        out_evidence = {
            **out_evidence,
            "rule_ids": [*out_evidence["rule_ids"], "LANG_ENGLISH_PERSON_BOOST"],
            "sources": ["people.csv"],
            "person_boost": {"before": conf_before, "after": conf, "boost": 0.10, "cap": 0.70},
        }

    return LanguageResult(
        language_primary="English",
        confidence_language=conf,
        flags=flags,
        evidence=out_evidence,
    )
