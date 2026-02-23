from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


from lodge_classifier.dicts.cache import DictCache
from lodge_classifier.matching.ngrams import build_ngrams


@dataclass(frozen=True)
class LanguageResult:
    """Language classification output for a single lodge name."""

    language_primary: str
    confidence_language: float
    flags: list[str]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class _LangSpec:
    """Specification for a non-English language dictionary rule."""

    language: str
    rel_path: str
    rule_id: str


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


def _safe_load_set(cache: DictCache, rel_path: str, column: str = "token") -> set[str]:
    """Load a token set from the cache; return empty set if the file is missing."""
    try:
        return cache.load_set(rel_path, column=column)
    except FileNotFoundError:
        return set()


def _intersect_sorted(a: Iterable[str], b: set[str]) -> list[str]:
    """Return a sorted intersection between an iterable and a set."""
    return sorted(set(a).intersection(b))


def detect_language_strict(
    tokens: list[str],
    dicts_dir: Path,
    curated_language_override: str | None = None,
    cache: DictCache | None = None,
) -> LanguageResult:
    """Detect language using strict dictionary rules (Option A), with UK-place and person boosts.

    Behaviour (precedence):
        1) Manual override (if provided) -> high confidence.
        2) Non-English dictionary evidence (French/German/Italian/etc./Japanese) -> high confidence.
        3) UK place evidence via gazetteer dictionaries (cities/towns, regions, landmarks)
           -> English with high confidence (cities/towns highest).
        4) English dictionary evidence -> English with high confidence.
        5) Otherwise -> English fallback with low confidence.

    Enhancements:
        - If a recognised person is present (prs/people.csv), boost English confidence for (3)-(5).
        - Person evidence never overrides non-English evidence.

    Notes:
        - UK place matches are checked using n-grams to support multi-word places.
        - Person matches are checked using n-grams first, then token fallback.

    Japanese support (anglicised / romaji):
        - Add a dictionary file at: language/japanese_tokens.csv (column: token)
        - Populate with romaji tokens (e.g., "sakura", "fuji", "nippon", etc.)
        - Detection is the same as other token dictionaries (token-level).

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
    person_terms = _safe_load_set(cache, "prs/people.csv", column="token")
    person_hits = (
        _intersect_sorted((p.lower() for p in phrases), {x.lower() for x in person_terms})
        if person_terms
        else []
    )
    if not person_hits and person_terms:
        person_hits = _intersect_sorted(token_set, {x.lower() for x in person_terms})

    # --- Non-English token dictionaries (including Japanese romaji) ---
    non_english_specs: list[_LangSpec] = [
        _LangSpec("French", "language/french_tokens.csv", "LANG_FRENCH_DICT"),
        _LangSpec("German", "language/german_tokens.csv", "LANG_GERMAN_DICT"),
        _LangSpec("Italian", "language/italian_tokens.csv", "LANG_ITALIAN_DICT"),
        _LangSpec("Latin", "language/latin_tokens.csv", "LANG_LATIN_DICT"),
        _LangSpec("Greek", "language/greek_tokens.csv", "LANG_GREEK_DICT"),
        _LangSpec("Maori", "language/maori_tokens.csv", "LANG_MAORI_DICT"),
        _LangSpec("Portuguese", "language/portuguese_tokens.csv", "LANG_PORTUGUESE_DICT"),
        _LangSpec("Japanese", "language/japanese_tokens.csv", "LANG_JAPANESE_DICT"),
    ]

    non_english_hits: dict[str, list[str]] = {}
    non_english_sources: dict[str, str] = {}

    for spec in non_english_specs:
        token_dict = _safe_load_set(cache, spec.rel_path, column="token")
        hits = _intersect_sorted(token_set, {x.lower() for x in token_dict}) if token_dict else []
        non_english_hits[spec.language] = hits
        non_english_sources[spec.language] = spec.rel_path

    # Optional English token dictionary (if you have it)
    english_tokens = _safe_load_set(cache, "language/english_tokens.csv", column="token")
    en_hits = (
        _intersect_sorted(token_set, {x.lower() for x in english_tokens}) if english_tokens else []
    )

    evidence.update(
        {
            "person_hits": person_hits,
            "english_hits": en_hits,
            "non_english_hits": non_english_hits,
        }
    )

    # 1) Non-English wins on any positive evidence (in listed precedence order)
    for spec in non_english_specs:
        hits = non_english_hits.get(spec.language, [])
        if hits:
            return LanguageResult(
                language_primary=spec.language,
                confidence_language=0.90,
                flags=["DICT_MATCH"],
                evidence={
                    **evidence,
                    "rule_ids": [spec.rule_id],
                    "sources": [non_english_sources[spec.language]],
                    "matched_tokens": hits,
                },
            )

    # 2) UK place evidence boosts English confidence (cities/towns strongest)
    uk_cities_towns = _safe_load_set(cache, "loc/cities_and_towns.csv", column="token")
    uk_regions = _safe_load_set(cache, "loc/regions.csv", column="token")
    uk_landmarks = _safe_load_set(cache, "loc/landmarks.csv", column="token")

    uk_cities_towns_l = {x.lower() for x in uk_cities_towns}
    uk_regions_l = {x.lower() for x in uk_regions}
    uk_landmarks_l = {x.lower() for x in uk_landmarks}

    phrases_l = {p.lower() for p in phrases}

    uk_cty_hits = sorted(phrases_l.intersection(uk_cities_towns_l))
    uk_reg_hits = sorted(phrases_l.intersection(uk_regions_l))
    uk_lan_hits = sorted(phrases_l.intersection(uk_landmarks_l))

    evidence.update(
        {
            "uk_city_town_hits": uk_cty_hits,
            "uk_region_hits": uk_reg_hits,
            "uk_landmark_hits": uk_lan_hits,
        }
    )

    def _apply_person_boost(
        *,
        base_conf: float,
        boost: float,
        cap: float,
        out_flags: list[str],
        out_evidence: dict[str, Any],
        sources: list[str] | None = None,
    ) -> tuple[float, list[str], dict[str, Any]]:
        """Apply person boost to confidence/evidence if person hits exist."""
        if not person_hits:
            return base_conf, out_flags, out_evidence

        out_flags = [*out_flags, "PERSON_BOOST"]
        conf_before = base_conf
        conf_after = min(cap, base_conf + boost)

        out_evidence = {
            **out_evidence,
            "rule_ids": [*out_evidence.get("rule_ids", []), "LANG_ENGLISH_PERSON_BOOST"],
            "sources": [*(out_evidence.get("sources", sources or [])), "prs/people.csv"],
            "person_boost": {
                "before": conf_before,
                "after": conf_after,
                "boost": boost,
                "cap": cap,
            },
        }

        return conf_after, out_flags, out_evidence

    if uk_cty_hits:
        conf = 0.95
        out_flags = ["DICT_MATCH", "UK_PLACE_BOOST"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_UK_CITY_TOWN"],
            "sources": ["loc/cities_and_towns.csv"],
            "uk_place_hits": uk_cty_hits,
        }
        conf, out_flags, out_evidence = _apply_person_boost(
            base_conf=conf,
            boost=0.02,
            cap=0.99,
            out_flags=out_flags,
            out_evidence=out_evidence,
            sources=["loc/cities_and_towns.csv"],
        )
        return LanguageResult("English", conf, out_flags, out_evidence)

    if uk_reg_hits:
        conf = 0.90
        out_flags = ["DICT_MATCH", "UK_PLACE_BOOST"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_UK_REGION"],
            "sources": ["loc/regions.csv"],
            "uk_place_hits": uk_reg_hits,
        }
        conf, out_flags, out_evidence = _apply_person_boost(
            base_conf=conf,
            boost=0.03,
            cap=0.97,
            out_flags=out_flags,
            out_evidence=out_evidence,
            sources=["loc/regions.csv"],
        )
        return LanguageResult("English", conf, out_flags, out_evidence)

    if uk_lan_hits:
        conf = 0.88
        out_flags = ["DICT_MATCH", "UK_PLACE_BOOST"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_UK_LANDMARK"],
            "sources": ["loc/landmarks.csv"],
            "uk_place_hits": uk_lan_hits,
        }
        conf, out_flags, out_evidence = _apply_person_boost(
            base_conf=conf,
            boost=0.04,
            cap=0.95,
            out_flags=out_flags,
            out_evidence=out_evidence,
            sources=["loc/landmarks.csv"],
        )
        return LanguageResult("English", conf, out_flags, out_evidence)

    # 3) English dictionary evidence (if present)
    if en_hits:
        conf = 0.90
        out_flags = ["DICT_MATCH"]
        out_evidence = {
            **evidence,
            "rule_ids": ["LANG_ENGLISH_DICT"],
            "sources": ["language/english_tokens.csv"],
        }
        conf, out_flags, out_evidence = _apply_person_boost(
            base_conf=conf,
            boost=0.05,
            cap=0.97,
            out_flags=out_flags,
            out_evidence=out_evidence,
            sources=["language/english_tokens.csv"],
        )
        return LanguageResult("English", conf, out_flags, out_evidence)

    # 4) Fallback English with low confidence (+ person boost if present)
    out_flags = [*flags, "ENGLISH_FALLBACK_LOW_CONF"]
    conf = 0.55
    out_evidence = {**evidence, "rule_ids": ["LANG_ENGLISH_FALLBACK"]}

    conf, out_flags, out_evidence = _apply_person_boost(
        base_conf=conf,
        boost=0.10,
        cap=0.70,
        out_flags=out_flags,
        out_evidence=out_evidence,
        sources=[],
    )

    return LanguageResult(
        language_primary="English",
        confidence_language=conf,
        flags=out_flags,
        evidence=out_evidence,
    )
