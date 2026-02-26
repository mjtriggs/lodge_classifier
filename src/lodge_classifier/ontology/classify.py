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
    """Classify ontology using deterministic dictionary rules (v1).

    Precedence (high level):
    1) Curated override
    2) High-signal primaries (PRS/GRP/LOC_REL/GRP_MAS/GRP_INT/etc.)
    3) Abstract primaries (including ABS_TMP) — ABS_TMP outranks location fallbacks
    4) Nature
    5) Object fallback (OBJ_STR)
    6) Location fallbacks (UK then global, incl. LOC_BDG)
    7) Strict Latin fallback
    8) Unknown

    Secondary behaviour:
    - If multiple ontology primaries are triggered, choose:
      * primary = highest precedence match
      * secondary = next-highest precedence match
      using the same PRIMARY_PRECEDENCE ordering.
    """
    flags: list[str] = []
    evidence_base: dict[str, Any] = {"tokens": tokens, "language_primary": language_primary}

    # ------------------------------------------------------------------
    # Primary precedence (single source of truth)
    # ------------------------------------------------------------------
    PRIMARY_PRECEDENCE: list[str] = [
        # PRS
        "PRS_REL",
        "PRS_ROY",
        "PRS_MYTH",
        "PRS_FIC",
        "PRS_HIS",
        # GRP / high-signal non-person
        "GRP_MIL",
        "LOC_REL",
        "GRP_MAS",
        "GRP_INT",
        "GRP_JOB",
        "GRP_EDU",
        "GRP_NAT",
        # ABS (ABS_TMP must outrank location fallbacks)
        "ABS_TMP",
        "ABS_VRT",
        "ABS_FRAT",
        "ABS_PHI",
        # NAT
        "NAT_BOT",
        "NAT_ANI",
        "NAT_AST",
        "NAT_GEO",
        # OBJ
        "OBJ_STR",
        "OBJ_EMB",
        "OBJ_SCI",
        "OBJ_MYTH",
        # LOC fallbacks
        "LOC_CTY",
        "LOC_REG",
        "LOC_LAN",
        "LOC_BDG",
        # Misc fallbacks
        "ABS_LAT",
        "UNK_",
    ]
    precedence_map = {code: i for i, code in enumerate(PRIMARY_PRECEDENCE)}
    if len(precedence_map) != len(PRIMARY_PRECEDENCE):
        raise ValueError("PRIMARY_PRECEDENCE contains duplicate ontology codes.")

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------
    if curated_ontology_hint and curated_ontology_hint.strip():
        flags.append("OVERRIDE_APPLIED")
        return OntologyResult(
            ontology_primary=curated_ontology_hint.strip(),
            ontology_secondary=None,
            confidence_ontology=0.99,
            flags=flags,
            evidence={
                **evidence_base,
                "reason": "curated_ontology_hint",
                "rule_ids": ["ONT_OVERRIDE"],
            },
        )

    cache = cache or DictCache(dicts_dir=dicts_dir)

    # Tokens are already normalised upstream; build phrase candidates (1..5-grams)
    phrases = build_ngrams(tokens, max_n=5)
    token_set = {t.lower() for t in tokens if t and t.strip()}

    def _hits(
        source: set[str], *, phrase_first: bool = False, token_only: bool = False
    ) -> list[str]:
        """Return sorted hits against a dictionary set.

        Args:
            source: Set of dictionary entries (assumed already normalised to lower-case).
            phrase_first: If True, prefer phrase hits and only fall back to token hits if empty.
            token_only: If True, match only against token_set (no phrase matching).
        """
        if token_only:
            return sorted(token_set.intersection(source))

        phrase_hits = phrases.intersection(source)
        token_hits = token_set.intersection(source)

        if phrase_first and phrase_hits:
            return sorted(phrase_hits)

        return sorted(set(phrase_hits) | set(token_hits))

    def _result(
        *,
        primary: str,
        secondary: str | None,
        confidence: float,
        rule: str,
        rule_ids: list[str],
        sources: list[str] | None = None,
        hits: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> OntologyResult:
        """Build a consistent OntologyResult payload."""
        ev: dict[str, Any] = {**evidence_base, "rule": rule, "rule_ids": rule_ids}
        if sources:
            ev["sources"] = sources
        if hits is not None:
            ev["hits"] = hits
        if extra:
            ev.update(extra)

        ev["primary_precedence_rank"] = precedence_map.get(primary, 9999)
        if secondary:
            ev["secondary_precedence_rank"] = precedence_map.get(secondary, 9999)

        return OntologyResult(
            ontology_primary=primary,
            ontology_secondary=secondary,
            confidence_ontology=confidence,
            flags=flags,
            evidence=ev,
        )

    def _add_candidate(
        candidates: list[dict[str, Any]],
        *,
        primary: str,
        confidence: float,
        rule: str,
        rule_ids: list[str],
        sources: list[str] | None = None,
        hits: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Add a candidate primary classification.

        Note:
            Secondary is not set here; it is derived later from the precedence ordering
            across all candidates (primary is #1, secondary is #2).
        """
        if primary not in precedence_map:
            raise ValueError(
                f"Candidate primary '{primary}' is not in PRIMARY_PRECEDENCE. "
                "Add it to PRIMARY_PRECEDENCE to make the selection order explicit."
            )

        candidates.append(
            {
                "primary": primary,
                "confidence": float(confidence),
                "rule": rule,
                "rule_ids": rule_ids,
                "sources": sources or [],
                "hits": hits or [],
                "extra": extra or {},
            }
        )

    def _dedupe_keep_best_per_primary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """If multiple candidates exist for the same primary, keep the strongest one.

        Strength is measured by:
        1) higher confidence
        2) more hits
        3) stable tie-breaker: keep first seen
        """
        best: dict[str, dict[str, Any]] = {}
        for c in candidates:
            key = c["primary"]
            if key not in best:
                best[key] = c
                continue

            incumbent = best[key]
            if (c["confidence"], len(c["hits"])) > (
                incumbent["confidence"],
                len(incumbent["hits"]),
            ):
                best[key] = c

        return list(best.values())

    def _sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort candidates by precedence, then confidence, then hit count."""
        return sorted(
            candidates,
            key=lambda c: (
                precedence_map[c["primary"]],
                -float(c["confidence"]),
                -len(c.get("hits") or []),
            ),
        )

    def _pick_primary_secondary(
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str | None]:
        """Choose primary and secondary from precedence ordering.

        Secondary is only set if a second NON-UNK_ primary exists.
        """
        if not candidates:
            raise ValueError("No ontology candidates were produced (unexpected).")

        deduped = _dedupe_keep_best_per_primary(candidates)
        ranked = _sort_candidates(deduped)

        primary = ranked[0]

        # Exclude UNK_ from secondary consideration
        non_unknown = [c for c in ranked if c["primary"] != "UNK_"]

        if len(non_unknown) >= 2:
            secondary = non_unknown[1]["primary"]
        else:
            secondary = None

        return primary, secondary

    # --------------------
    # Dictionaries
    # --------------------
    # PRS
    people = cache.load_set("prs/prs_his.csv", column="token")
    myth_terms = cache.load_set("prs/prs_myth.csv", column="token")
    religious_person_terms = cache.load_set("prs/prs_rel.csv", column="token")
    royals = cache.load_set("prs/prs_roy.csv", column="token")

    # LOC (auto + manual combined)
    cty_auto = cache.load_set("loc/loc_cty_auto.csv", column="token")
    cty_manual = cache.load_set("loc/loc_cty_manual.csv", column="token")
    reg_auto = cache.load_set("loc/loc_reg_auto.csv", column="token")
    reg_manual = cache.load_set("loc/loc_reg_manual.csv", column="token")

    uk_loc_cty = set(cty_auto) | set(cty_manual)
    uk_loc_reg = set(reg_auto) | set(reg_manual)

    religious_place_terms = cache.load_set("loc/loc_rel.csv", column="token")
    uk_loc_lan = cache.load_set("loc/loc_lan.csv", column="token")
    loc_bdg_terms = cache.load_set("loc/loc_bdg.csv", column="token")

    # GRP
    edu_terms = cache.load_set("grp/grp_edu.csv", column="token")
    job_terms = cache.load_set("grp/grp_job.csv", column="token")
    masonic_terms = cache.load_set("grp/grp_mas.csv", column="token")
    military_terms = cache.load_set("grp/grp_mil.csv", column="token")
    special_interests = cache.load_set("grp/grp_int.csv", column="token")
    nationality_terms = cache.load_set("grp/grp_nat.csv", column="token")

    # NAT
    animals = cache.load_set("nat/nat_ani.csv", column="token")
    astronomical = cache.load_set("nat/nat_ast.csv", column="token")
    botanical = cache.load_set("nat/nat_bot.csv", column="token")

    # OBJ
    obj_structure_terms = cache.load_set("obj/obj_str.csv", column="token")
    obj_emblem_terms = cache.load_set("obj/obj_emb.csv", column="token")
    obj_mythic_terms = cache.load_set("obj/obj_myth.csv", column="token")
    obj_scientific_terms = cache.load_set("obj/obj_sci.csv", column="token")

    # ABS
    fraternal_terms = cache.load_set("abs/abs_frat.csv", column="token")
    philosophical_terms = cache.load_set("abs/abs_phi.csv", column="token")
    virtues = cache.load_set("abs/abs_vrt.csv", column="token")
    temporal_terms = cache.load_set("abs/abs_tmp.csv", column="token")

    # --------------------
    # Precompute hits used across candidates (for evidence convenience)
    # --------------------
    edu_hits = sorted(token_set.intersection(edu_terms))
    natl_hits = sorted(token_set.intersection(nationality_terms))

    obj_str_hits = _hits(obj_structure_terms, phrase_first=False)
    obj_emb_hits = _hits(obj_emblem_terms, phrase_first=False)
    obj_myth_hits = _hits(obj_mythic_terms, phrase_first=False)
    obj_sci_hits = _hits(obj_scientific_terms, phrase_first=False)

    tmp_hits_phrase = sorted(phrases.intersection(temporal_terms))
    tmp_hits_token = sorted(token_set.intersection(temporal_terms))
    tmp_hits = sorted(set(tmp_hits_phrase + tmp_hits_token))

    # Updated location hit logic (auto+manual union, phrase + token)
    loc_hits_cty = _hits(uk_loc_cty, phrase_first=True)
    loc_hits_reg = _hits(uk_loc_reg, phrase_first=True)
    loc_hits_lan = _hits(uk_loc_lan, phrase_first=True)
    loc_hits_bdg = _hits(loc_bdg_terms, phrase_first=True) if loc_bdg_terms else []

    # --------------------
    # Build all candidates
    # --------------------
    candidates: list[dict[str, Any]] = []

    # High-signal PRS/GRP/LOC
    rel_person_hits = _hits(religious_person_terms, token_only=True)
    if rel_person_hits:
        _add_candidate(
            candidates,
            primary="PRS_REL",
            confidence=0.90,
            rule="prs_rel_terms",
            rule_ids=["ONT_PRS_REL_TERMS"],
            sources=["prs/prs_rel.csv"],
            hits=rel_person_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    roy_hits = _hits(royals, token_only=True)
    if roy_hits:
        _add_candidate(
            candidates,
            primary="PRS_ROY",
            confidence=0.88,
            rule="prs_roy_titles",
            rule_ids=["ONT_PRS_ROY_TITLES"],
            sources=["prs/prs_roy.csv"],
            hits=roy_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    myth_hits = _hits(myth_terms, token_only=True)
    if myth_hits:
        _add_candidate(
            candidates,
            primary="PRS_MYTH",
            confidence=0.88,
            rule="prs_myth_terms",
            rule_ids=["ONT_PRS_MYTH_TERMS"],
            sources=["prs/prs_myth.csv"],
            hits=myth_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    people_hits = _hits(people, phrase_first=True)
    if people_hits:
        _add_candidate(
            candidates,
            primary="PRS_HIS",
            confidence=0.86,
            rule="prs_people_list",
            rule_ids=["ONT_PRS_PEOPLE_LIST"],
            sources=["prs/prs_his.csv"],
            hits=people_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    mil_hits = _hits(military_terms, token_only=True)
    if mil_hits:
        _add_candidate(
            candidates,
            primary="GRP_MIL",
            confidence=0.90,
            rule="grp_mil_terms",
            rule_ids=["ONT_GRP_MIL_TERMS"],
            sources=["grp/grp_mil.csv"],
            hits=mil_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    rel_place_hits = _hits(religious_place_terms, token_only=True)
    if rel_place_hits:
        _add_candidate(
            candidates,
            primary="LOC_REL",
            confidence=0.88,
            rule="loc_rel_terms",
            rule_ids=["ONT_LOC_REL_TERMS"],
            sources=["loc/loc_rel.csv"],
            hits=rel_place_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    mas_hits = _hits(masonic_terms, token_only=True)
    if mas_hits:
        _add_candidate(
            candidates,
            primary="GRP_MAS",
            confidence=0.88,
            rule="grp_mas_terms",
            rule_ids=["ONT_GRP_MAS_TERMS"],
            sources=["grp/grp_mas.csv"],
            hits=mas_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    int_hits = _hits(special_interests, phrase_first=True)
    if int_hits:
        _add_candidate(
            candidates,
            primary="GRP_INT",
            confidence=0.88,
            rule="grp_int_terms",
            rule_ids=["ONT_GRP_INT_TERMS"],
            sources=["grp/grp_int.csv"],
            hits=int_hits,
            extra={"edu_hits": edu_hits, "obj_str_hits": obj_str_hits, "tmp_hits": tmp_hits},
        )

    # Lower-signal GRP
    job_hits = sorted(token_set.intersection(job_terms))
    if job_hits:
        _add_candidate(
            candidates,
            primary="GRP_JOB",
            confidence=0.82,
            rule="grp_job_terms",
            rule_ids=["ONT_GRP_JOB_TERMS"],
            sources=["grp/grp_job.csv"],
            hits=job_hits,
        )

    if edu_hits:
        _add_candidate(
            candidates,
            primary="GRP_EDU",
            confidence=0.82,
            rule="grp_edu_terms",
            rule_ids=["ONT_GRP_EDU_TERMS"],
            sources=["grp/grp_edu.csv"],
            hits=edu_hits,
        )

    if natl_hits:
        _add_candidate(
            candidates,
            primary="GRP_NAT",
            confidence=0.82,
            rule="grp_nat_terms",
            rule_ids=["ONT_GRP_NAT_TERMS"],
            sources=["grp/grp_nat.csv"],
            hits=natl_hits,
        )

    # ABS (ABS_TMP outranks location fallbacks via precedence)
    if tmp_hits:
        _add_candidate(
            candidates,
            primary="ABS_TMP",
            confidence=0.83,
            rule="abs_tmp_terms",
            rule_ids=["ONT_ABS_TMP_TERMS"],
            sources=["abs/abs_tmp.csv"],
            hits=tmp_hits,
            extra={
                "tmp_hits_phrase": tmp_hits_phrase,
                "tmp_hits_token": tmp_hits_token,
                "loc_hits_cty": loc_hits_cty,
                "loc_hits_reg": loc_hits_reg,
                "loc_hits_lan": loc_hits_lan,
                "loc_hits_bdg": loc_hits_bdg,
            },
        )

    vrt_hits = sorted(token_set.intersection(virtues))
    if vrt_hits:
        _add_candidate(
            candidates,
            primary="ABS_VRT",
            confidence=0.85,
            rule="abs_vrt_terms",
            rule_ids=["ONT_ABS_VRT_TERMS"],
            sources=["abs/abs_vrt.csv"],
            hits=vrt_hits,
        )

    frat_hits = sorted(token_set.intersection(fraternal_terms))
    if frat_hits:
        _add_candidate(
            candidates,
            primary="ABS_FRAT",
            confidence=0.80,
            rule="abs_frat_terms",
            rule_ids=["ONT_ABS_FRAT_TERMS"],
            sources=["abs/abs_frat.csv"],
            hits=frat_hits,
        )

    phi_hits = sorted(token_set.intersection(philosophical_terms))
    if phi_hits:
        _add_candidate(
            candidates,
            primary="ABS_PHI",
            confidence=0.80,
            rule="abs_phi_terms",
            rule_ids=["ONT_ABS_PHI_TERMS"],
            sources=["abs/abs_phi.csv"],
            hits=phi_hits,
        )

    # NAT
    bot_hits = sorted(token_set.intersection(botanical))
    if bot_hits:
        _add_candidate(
            candidates,
            primary="NAT_BOT",
            confidence=0.75,
            rule="nat_bot_terms",
            rule_ids=["ONT_NAT_BOT_TERMS"],
            sources=["nat/nat_bot.csv"],
            hits=bot_hits,
        )

    ani_hits = sorted(token_set.intersection(animals))
    if ani_hits:
        _add_candidate(
            candidates,
            primary="NAT_ANI",
            confidence=0.75,
            rule="nat_ani_terms",
            rule_ids=["ONT_NAT_ANI_TERMS"],
            sources=["nat/nat_ani.csv"],
            hits=ani_hits,
        )

    ast_hits = sorted(token_set.intersection(astronomical))
    if ast_hits:
        _add_candidate(
            candidates,
            primary="NAT_AST",
            confidence=0.75,
            rule="nat_ast_terms",
            rule_ids=["ONT_NAT_AST_TERMS"],
            sources=["nat/nat_ast.csv"],
            hits=ast_hits,
        )

    # OBJ
    if obj_str_hits:
        _add_candidate(
            candidates,
            primary="OBJ_STR",
            confidence=0.70,
            rule="obj_str_terms",
            rule_ids=["ONT_OBJ_STR_TERMS"],
            sources=["obj/obj_str.csv"],
            hits=obj_str_hits,
        )

    if obj_emb_hits:
        _add_candidate(
            candidates,
            primary="OBJ_EMB",
            confidence=0.70,
            rule="obj_emb_terms",
            rule_ids=["ONT_OBJ_EMB_TERMS"],
            sources=["obj/obj_emb.csv"],
            hits=obj_str_hits,
        )

    if obj_myth_hits:
        _add_candidate(
            candidates,
            primary="OBJ_MYTH",
            confidence=0.70,
            rule="obj_myth_terms",
            rule_ids=["ONT_OBJ_MYTH_TERMS"],
            sources=["obj/obj_myth.csv"],
            hits=obj_str_hits,
        )

    if obj_sci_hits:
        _add_candidate(
            candidates,
            primary="OBJ_SCI",
            confidence=0.70,
            rule="obj_sci_terms",
            rule_ids=["ONT_OBJ_SCI_TERMS"],
            sources=["obj/obj_sci.csv"],
            hits=obj_str_hits,
        )

    # LOC fallbacks (primary)
    if loc_hits_cty:
        _add_candidate(
            candidates,
            primary="LOC_CTY",
            confidence=0.86,
            rule="loc_uk_cty",
            rule_ids=["ONT_LOC_UK_CTY"],
            sources=["loc/loc_cty_auto.csv", "loc/loc_cty_manual.csv"],
            hits=loc_hits_cty,
        )

    if loc_hits_reg:
        _add_candidate(
            candidates,
            primary="LOC_REG",
            confidence=0.84,
            rule="loc_uk_reg",
            rule_ids=["ONT_LOC_UK_REG"],
            sources=["loc/loc_reg_auto.csv", "loc/loc_reg_manual.csv"],
            hits=loc_hits_reg,
        )

    if loc_hits_lan:
        _add_candidate(
            candidates,
            primary="LOC_LAN",
            confidence=0.82,
            rule="loc_uk_lan",
            rule_ids=["ONT_LOC_UK_LAN"],
            sources=["loc/loc_lan.csv"],
            hits=loc_hits_lan,
        )

    if loc_hits_bdg:
        _add_candidate(
            candidates,
            primary="LOC_BDG",
            confidence=0.82,
            rule="loc_uk_bdg",
            rule_ids=["ONT_LOC_UK_BDG"],
            sources=["loc/loc_bdg.csv"],
            hits=loc_hits_bdg,
        )

    # Strict Latin fallback
    stopwords = {"of", "the", "and"}
    meaningful = [t for t in tokens if t.lower() not in stopwords]
    if (language_primary == "Latin") and len(meaningful) == 1:
        flags.append("STRICT_LATIN_FALLBACK")
        _add_candidate(
            candidates,
            primary="ABS_LAT",
            confidence=0.70,
            rule="abs_lat_single_token",
            rule_ids=["ONT_ABS_LAT_SINGLE_TOKEN"],
            hits=meaningful,
            extra={"meaningful": meaningful},
        )

    # Unknown fallback always exists
    flags.append("REVIEW_REQUIRED")
    _add_candidate(
        candidates,
        primary="UNK_",
        confidence=0.20,
        rule="fallback",
        rule_ids=["ONT_FALLBACK_UNKNOWN"],
    )

    # ------------------------------------------------------------------
    # Pick primary + secondary using the SAME precedence ordering
    # ------------------------------------------------------------------
    primary_choice, secondary_choice = _pick_primary_secondary(candidates)

    # Lightweight audit trail (top 5 by ranking) to make review/debug easier
    ranked = _sort_candidates(_dedupe_keep_best_per_primary(candidates))[:5]
    audit_top5 = [
        {
            "primary": c["primary"],
            "confidence": c["confidence"],
            "hits_n": len(c.get("hits") or []),
            "precedence_rank": precedence_map[c["primary"]],
            "rule": c["rule"],
        }
        for c in ranked
    ]

    return _result(
        primary=primary_choice["primary"],
        secondary=secondary_choice,
        confidence=primary_choice["confidence"],
        rule=primary_choice["rule"],
        rule_ids=primary_choice["rule_ids"],
        sources=primary_choice["sources"],
        hits=primary_choice["hits"],
        extra={**primary_choice["extra"], "candidate_audit_top5": audit_top5},
    )
