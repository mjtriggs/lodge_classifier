from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lodge_classifier.dicts.loader import TermDicts


@dataclass(frozen=True)
class OntologyResult:
    """Ontology classification output.

    Attributes:
        ontology_primary: Primary ontology code.
        ontology_secondary: Optional secondary ontology code.
        confidence_ontology: Confidence score for ontology inference.
        flags: Pipeline flags for debugging/QA.
        evidence: Evidence payload for export into evidence_json.
    """

    ontology_primary: str
    ontology_secondary: str | None
    confidence_ontology: float
    flags: list[str]
    evidence: dict[str, Any]


_ONTOLOGY_PRECEDENCE: dict[str, int] = {
    # Lower number = higher precedence for primary selection.
    "PRS_REL": 10,
    "REL_TEM": 20,
    "GRP_MIL": 40,
    "GRP_EDU": 50,
    "LOC_BDG": 90,
    "UNK_": 999,
}


def _choose_primary_secondary(codes: list[str]) -> tuple[str, str | None]:
    """Select primary and secondary ontology using a precedence table."""
    uniq = sorted(set(codes), key=lambda c: (_ONTOLOGY_PRECEDENCE.get(c, 500), c))
    primary = uniq[0] if uniq else "UNK_"
    secondary = uniq[1] if len(uniq) > 1 else None
    return primary, secondary


def resolve_ontology_v1(
    *,
    tokens: list[str],
    language_primary: str,
    dicts: TermDicts,
) -> OntologyResult:
    """Resolve ontology (primary + secondary) from tokens using term dictionaries.

    This version adds a dedicated place-of-worship ontology code: REL_TEM.
    Tokens found in dicts.temple_terms will map to REL_TEM (not LOC_BDG).

    Args:
        tokens: Normalised tokens (lowercase).
        language_primary: Detected language label (strict option A).
        dicts: Loaded term dictionaries (from data/dicts/).

    Returns:
        OntologyResult with ontology codes, confidence, and evidence.
    """
    flags: list[str] = []
    token_set = set(tokens)

    prs_rel_hits = sorted(token_set.intersection(dicts.prs_rel_terms))
    temple_hits = sorted(token_set.intersection(dicts.temple_terms))
    mil_hits = sorted(token_set.intersection(dicts.mil_terms))
    edu_hits = sorted(token_set.intersection(dicts.edu_terms))
    bdg_hits = sorted(token_set.intersection(dicts.bdg_terms))

    codes: list[str] = []
    if prs_rel_hits:
        codes.append("PRS_REL")
    if temple_hits:
        codes.append("REL_TEM")
    if mil_hits:
        codes.append("GRP_MIL")
    if edu_hits:
        codes.append("GRP_EDU")
    if bdg_hits:
        codes.append("LOC_BDG")

    if not codes:
        codes = ["UNK_"]

    ontology_primary, ontology_secondary = _choose_primary_secondary(codes)

    confidence = 0.2 if ontology_primary == "UNK_" else 0.9
    if ontology_secondary is not None and confidence < 0.95:
        confidence = 0.9

    evidence: dict[str, Any] = {
        "tokens": tokens,
        "language_primary": language_primary,
        "rule": "resolve_ontology_v1",
        "prs_rel_hits": prs_rel_hits,
        "temple_hits": temple_hits,
        "mil_hits": mil_hits,
        "edu_hits": edu_hits,
        "bdg_hits": bdg_hits,
    }

    return OntologyResult(
        ontology_primary=ontology_primary,
        ontology_secondary=ontology_secondary,
        confidence_ontology=confidence,
        flags=flags,
        evidence=evidence,
    )
