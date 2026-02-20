from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TermDicts:
    """Container for all term dictionaries used by the pipeline.

    Attributes:
        mil_terms: Military/service related terms.
        edu_terms: Educational/institutional terms.
        bdg_terms: Secular building terms (places/buildings).
        temple_terms: Places-of-worship terms (treated as REL_TEM).
        prs_rel_terms: Religious person/title terms (e.g. st/saint).
    """

    mil_terms: set[str]
    edu_terms: set[str]
    bdg_terms: set[str]
    temple_terms: set[str]
    prs_rel_terms: set[str]


def _load_term_file(path: Path) -> set[str]:
    """Load a newline-separated term file into a normalised set.

    Lines beginning with '#' are treated as comments.
    Blank lines are ignored.

    Args:
        path: Path to a text file.

    Returns:
        A set of lowercase terms.
    """
    terms: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        terms.add(line.lower())
    return terms


def load_term_dicts(dicts_dir: Path) -> TermDicts:
    """Load all term dictionaries from a single directory.

    Expected files (you can adjust names to match your project):
        - mil_terms.txt
        - edu_terms.txt
        - bdg_terms.txt
        - temple_terms.txt
        - prs_rel_terms.txt

    Important:
        We treat bdg_terms as *secular buildings* by subtracting temple_terms,
        so 'church'/'cathedral' cannot be double-counted.

    Args:
        dicts_dir: Base directory containing dictionary text files.

    Returns:
        TermDicts containing all loaded term sets.
    """
    mil_terms = _load_term_file(dicts_dir / "mil_terms.txt")
    edu_terms = _load_term_file(dicts_dir / "edu_terms.txt")
    bdg_terms = _load_term_file(dicts_dir / "bdg_terms.txt")
    temple_terms = _load_term_file(dicts_dir / "temple_terms.txt")
    prs_rel_terms = _load_term_file(dicts_dir / "prs_rel_terms.txt")

    # Ensure worship terms are not treated as generic buildings.
    bdg_terms = bdg_terms.difference(temple_terms)

    return TermDicts(
        mil_terms=mil_terms,
        edu_terms=edu_terms,
        bdg_terms=bdg_terms,
        temple_terms=temple_terms,
        prs_rel_terms=prs_rel_terms,
    )
