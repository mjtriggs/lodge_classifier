from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodge_classifier.dicts.cache import DictCache
from lodge_classifier.matching.ngrams import build_ngrams


@dataclass(frozen=True)
class OntologyResult:
    """Ontology classification output for a single lodge name."""

    ontology_primary: str
    ontology_secondary: str | None
    confidence_ontology: float
    flags: list[str]
    evidence: dict[str, Any]


def classify_ontology_v1(
    tokens: list[str],
    dicts_dir: Path,
    language_primary: str | None = None,
    curated_ontology_hint: str | None = None,
    cache: DictCache | None = None,
) -> OntologyResult:
    """Classify ontology using deterministic dictionary rules (v1)."""
    flags: list[str] = []
    evidence: dict[str, Any] = {"tokens": tokens, "language_primary": language_primary}

    if curated_ontology_hint and curated_ontology_hint.strip():
        flags.append("OVERRIDE_APPLIED")
        return OntologyResult(
            ontology_primary=curated_ontology_hint.strip(),
            ontology_secondary=None,
            confidence_ontology=0.99,
            flags=flags,
            evidence={**evidence, "reason": "curated_ontology_hint", "rule_ids": ["ONT_OVERRIDE"]},
        )

    cache = cache or DictCache(dicts_dir=dicts_dir)

    # Tokens are already normalised upstream; build phrase candidates (1..5-grams)
    phrases = build_ngrams(tokens, max_n=5)
    token_set = {t.lower() for t in tokens if t and t.strip()}

    # Load Dictionaries
    ## Person (PRS)
    people = cache.load_set("prs/people.csv", column="token")
    myth_terms = cache.load_set("prs/myth_terms.csv", column="token")
    religious_person_terms = cache.load_set("prs/religious_person_terms.csv", column="token")
    royal_titles = cache.load_set("prs/royal_titles.csv", column="token")

    ## Location (LOC)
    global_places = cache.load_set("loc/global_places.csv", column="token")
    religious_place_terms = cache.load_set("loc/religious_place_terms.csv", column="token")
    uk_loc_reg = cache.load_set("loc/regions.csv", column="token")
    uk_loc_cty = cache.load_set("loc/cities_and_towns.csv", column="token")
    uk_loc_lan = cache.load_set("loc/landmarks.csv", column="token")

    ## Collective / Body (GRP)
    edu_terms = cache.load_set("grp/edu_terms.csv", column="token")
    job_terms = cache.load_set("grp/job_terms.csv", column="token")
    masonic_terms = cache.load_set("grp/masonic_terms.csv", column="token")
    military_terms = cache.load_set("grp/military_terms.csv", column="token")
    special_interests = cache.load_set("grp/special_interests.csv", column="token")
    nationality_terms = cache.load_set("grp/nationality_terms.csv", column="token")

    ## Nature (NAT)
    animals = cache.load_set("nat/animals.csv", column="token")
    astronomical = cache.load_set("nat/astronomical.csv", column="token")
    botanical = cache.load_set("nat/botanical.csv", column="token")

    ## Object (OBJ)
    building_terms = cache.load_set("obj/building_terms.csv", column="token")

    ## Abstract Concept (ABS)
    fraternal_terms = cache.load_set("abs/fraternal_terms.csv", column="token")
    philosophical_terms = cache.load_set("abs/philosophical_terms.csv", column="token")
    virtues = cache.load_set("abs/virtues.csv", column="token")

    # UK place dictionaries
    loc_hits_cty = sorted(phrases.intersection(uk_loc_cty))
    loc_hits_reg = sorted(phrases.intersection(uk_loc_reg))
    loc_hits_lan = sorted(phrases.intersection(uk_loc_lan))

    # Choose a single UK location secondary for abstract concepts (precedence)
    loc_secondary_uk: str | None = None
    loc_hits_uk_any: list[str] = []
    if loc_hits_cty:
        loc_secondary_uk = "LOC_CTY"
        loc_hits_uk_any = loc_hits_cty
    elif loc_hits_reg:
        loc_secondary_uk = "LOC_REG"
        loc_hits_uk_any = loc_hits_reg
    elif loc_hits_lan:
        loc_secondary_uk = "LOC_LAN"
        loc_hits_uk_any = loc_hits_lan

    # Prevent double-counting religious buildings as generic buildings
    building_terms = building_terms.difference(religious_place_terms)

    edu_hits = sorted(token_set.intersection(edu_terms))
    bdg_hits = sorted(token_set.intersection(building_terms))
    nationality_hits = sorted(token_set.intersection(nationality_terms))

    # Global places: phrase matching (1..5 grams)
    loc_hits_global = sorted(phrases.intersection(global_places))
    loc_secondary_global = "LOC_REG" if loc_hits_global else None

    # Default secondary ontology (single slot)
    ontology_secondary: str | None = None
    if edu_hits:
        ontology_secondary = "GRP_EDU"
    elif bdg_hits:
        ontology_secondary = "LOC_BDG"
    elif nationality_hits:
        ontology_secondary = "GRP_NAT"

    # --- High-signal primaries ---
    rel_person_hits = sorted(token_set.intersection(religious_person_terms))
    if rel_person_hits:
        return OntologyResult(
            ontology_primary="PRS_REL",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.90,
            flags=flags,
            evidence={
                **evidence,
                "rule": "prs_rel_terms",
                "hits": rel_person_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
                "rule_ids": ["ONT_PRS_REL_TERMS"],
                "sources": ["religious_person_terms.csv"],
            },
        )

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
                "rule_ids": ["ONT_PRS_ROY_TITLES"],
                "sources": ["royal_titles.csv"],
            },
        )

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
                "rule_ids": ["ONT_PRS_MYTH_TERMS"],
                "sources": ["myth_terms.csv"],
            },
        )

    # People list: phrase-first to support multi-word names, with token fallback
    people_hits = sorted(phrases.intersection(people))
    if not people_hits:
        people_hits = sorted(token_set.intersection(people))

    if people_hits:
        return OntologyResult(
            ontology_primary="PRS_HIS",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.86,
            flags=flags,
            evidence={
                **evidence,
                "rule": "prs_people_list",
                "hits": people_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
                "rule_ids": ["ONT_PRS_PEOPLE_LIST"],
                "sources": ["people.csv"],
            },
        )

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
                "rule_ids": ["ONT_GRP_MIL_TERMS"],
                "sources": ["military_terms.csv"],
            },
        )

    rel_place_hits = sorted(token_set.intersection(religious_place_terms))
    if rel_place_hits:
        return OntologyResult(
            ontology_primary="LOC_REL",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.88,
            flags=flags,
            evidence={
                **evidence,
                "rule": "loc_rel_terms",
                "hits": rel_place_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
                "rule_ids": ["ONT_LOC_REL_TERMS"],
                "sources": ["religious_place_terms.csv"],
            },
        )

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
                "rule_ids": ["ONT_GRP_MAS_TERMS"],
                "sources": ["masonic_terms.csv"],
            },
        )

    # Match special interests against phrases (supports multi-word entries)
    int_hits = sorted(phrases.intersection(special_interests))
    if not int_hits:
        int_hits = sorted(token_set.intersection(special_interests))

    if int_hits:
        return OntologyResult(
            ontology_primary="GRP_INT",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.88,
            flags=flags,
            evidence={
                **evidence,
                "rule": "grp_int_terms",
                "hits": int_hits,
                "edu_hits": edu_hits,
                "bdg_hits": bdg_hits,
                "rule_ids": ["ONT_GRP_INT_TERMS"],
                "sources": ["special_interests.csv"],
            },
        )

    job_hits = sorted(token_set.intersection(job_terms))
    if job_hits:
        return OntologyResult(
            ontology_primary="GRP_JOB",
            ontology_secondary=None,
            confidence_ontology=0.82,
            flags=flags,
            evidence={
                **evidence,
                "rule": "grp_job_terms",
                "hits": job_hits,
                "rule_ids": ["ONT_GRP_JOB_TERMS"],
                "sources": ["job_terms.csv"],
            },
        )

    if edu_hits:
        return OntologyResult(
            ontology_primary="GRP_EDU",
            ontology_secondary=None,
            confidence_ontology=0.82,
            flags=flags,
            evidence={
                **evidence,
                "rule": "grp_edu_terms",
                "hits": edu_hits,
                "rule_ids": ["ONT_GRP_EDU_TERMS"],
                "sources": ["edu_terms.csv"],
            },
        )

    if nationality_hits:
        return OntologyResult(
            ontology_primary="GRP_NAT",
            ontology_secondary=None,
            confidence_ontology=0.82,
            flags=flags,
            evidence={
                **evidence,
                "rule": "grp_nat_terms",
                "hits": edu_hits,
                "rule_ids": ["ONT_GRP_NAT_TERMS"],
                "sources": ["nationality_terms.csv"],
            },
        )

    # --- Abstract concepts ---
    vrt_hits = sorted(token_set.intersection(virtues))
    if vrt_hits:
        secondary = ontology_secondary or loc_secondary_uk or loc_secondary_global
        return OntologyResult(
            ontology_primary="ABS_VRT",
            ontology_secondary=secondary,
            confidence_ontology=0.85,
            flags=flags,
            evidence={
                **evidence,
                "rule": "abs_vrt_terms",
                "hits": vrt_hits,
                "loc_hits_uk": loc_hits_uk_any,
                "loc_hits_global": loc_hits_global,
                "rule_ids": ["ONT_ABS_VRT_TERMS"],
                "sources": ["virtues.csv"],
            },
        )

    frat_hits = sorted(token_set.intersection(fraternal_terms))
    if frat_hits:
        secondary = ontology_secondary or loc_secondary_uk or loc_secondary_global
        return OntologyResult(
            ontology_primary="ABS_FRAT",
            ontology_secondary=secondary,
            confidence_ontology=0.80,
            flags=flags,
            evidence={
                **evidence,
                "rule": "abs_frat_terms",
                "hits": frat_hits,
                "loc_hits_uk": loc_hits_uk_any,
                "loc_hits_global": loc_hits_global,
                "rule_ids": ["ONT_ABS_FRAT_TERMS"],
                "sources": ["fraternal_terms.csv"],
            },
        )

    phi_hits = sorted(token_set.intersection(philosophical_terms))
    if phi_hits:
        secondary = ontology_secondary or loc_secondary_uk or loc_secondary_global
        return OntologyResult(
            ontology_primary="ABS_PHI",
            ontology_secondary=secondary,
            confidence_ontology=0.80,
            flags=flags,
            evidence={
                **evidence,
                "rule": "abs_phi_terms",
                "hits": phi_hits,
                "loc_hits_uk": loc_hits_uk_any,
                "loc_hits_global": loc_hits_global,
                "rule_ids": ["ONT_ABS_PHI_TERMS"],
                "sources": ["philosophical_terms.csv"],
            },
        )

    # --- Nature ---
    bot_hits = sorted(token_set.intersection(botanical))
    if bot_hits:
        return OntologyResult(
            ontology_primary="NAT_BOT",
            ontology_secondary=None,
            confidence_ontology=0.75,
            flags=flags,
            evidence={
                **evidence,
                "rule": "nat_bot_terms",
                "hits": bot_hits,
                "rule_ids": ["ONT_NAT_BOT_TERMS"],
                "sources": ["botanical.csv"],
            },
        )

    ani_hits = sorted(token_set.intersection(animals))
    if ani_hits:
        return OntologyResult(
            ontology_primary="NAT_ANI",
            ontology_secondary=None,
            confidence_ontology=0.75,
            flags=flags,
            evidence={
                **evidence,
                "rule": "nat_ani_terms",
                "hits": ani_hits,
                "rule_ids": ["ONT_NAT_ANI_TERMS"],
                "sources": ["animals.csv"],
            },
        )

    ast_hits = sorted(token_set.intersection(astronomical))
    if ast_hits:
        return OntologyResult(
            ontology_primary="NAT_AST",
            ontology_secondary=None,
            confidence_ontology=0.75,
            flags=flags,
            evidence={
                **evidence,
                "rule": "nat_ast_terms",
                "hits": ast_hits,
                "rule_ids": ["ONT_NAT_AST_TERMS"],
                "sources": ["astronomical.csv"],
            },
        )

    # --- UK place fallback (primary) ---
    if loc_hits_cty:
        return OntologyResult(
            ontology_primary="LOC_CTY",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.86,
            flags=flags,
            evidence={
                **evidence,
                "rule": "loc_uk_cty",
                "hits": loc_hits_cty,
                "rule_ids": ["ONT_LOC_UK_CTY"],
                "sources": ["cities_and_towns.csv"],
            },
        )

    if loc_hits_reg:
        return OntologyResult(
            ontology_primary="LOC_REG",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.84,
            flags=flags,
            evidence={
                **evidence,
                "rule": "loc_uk_reg",
                "hits": loc_hits_reg,
                "rule_ids": ["ONT_LOC_UK_REG"],
                "sources": ["regions.csv"],
            },
        )

    if loc_hits_lan:
        return OntologyResult(
            ontology_primary="LOC_LAN",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.82,
            flags=flags,
            evidence={
                **evidence,
                "rule": "loc_uk_lan",
                "hits": loc_hits_lan,
                "rule_ids": ["ONT_LOC_UK_LAN"],
                "sources": ["landmarks.csv"],
            },
        )

    # --- Global place fallback (primary) ---
    if loc_hits_global:
        return OntologyResult(
            ontology_primary="LOC_REG",
            ontology_secondary=ontology_secondary,
            confidence_ontology=0.85,
            flags=flags,
            evidence={
                **evidence,
                "rule": "loc_global_places",
                "hits": loc_hits_global,
                "rule_ids": ["ONT_LOC_GLOBAL_PLACES"],
                "sources": ["global_places.csv"],
            },
        )

    # --- Strict Latin fallback ---
    stopwords = {"of", "the", "and"}
    meaningful = [t for t in tokens if t.lower() not in stopwords]

    if (language_primary == "Latin") and len(meaningful) == 1:
        flags.append("STRICT_LATIN_FALLBACK")
        return OntologyResult(
            ontology_primary="ABS_LAT",
            ontology_secondary=None,
            confidence_ontology=0.70,
            flags=flags,
            evidence={
                **evidence,
                "rule": "abs_lat_single_token",
                "meaningful": meaningful,
                "rule_ids": ["ONT_ABS_LAT_SINGLE_TOKEN"],
            },
        )

    flags.append("REVIEW_REQUIRED")
    return OntologyResult(
        ontology_primary="UNK_",
        ontology_secondary=None,
        confidence_ontology=0.20,
        flags=flags,
        evidence={**evidence, "rule": "fallback", "rule_ids": ["ONT_FALLBACK_UNKNOWN"]},
    )
