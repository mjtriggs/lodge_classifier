from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalisedName:
    """A normalised representation of a lodge name."""

    raw: str
    clean: str
    normalised: str
    tokens: list[str]


_LODGE_TOKEN_RE = re.compile(r"\b(lodge)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalise_lodge_name(raw_name: str) -> NormalisedName:
    """Normalise a lodge name into a clean string, normalised string, and tokens.

    Behaviour:
        - Strips leading/trailing whitespace.
        - Standardises apostrophes to ASCII.
        - Removes the token "lodge" (case-insensitive).
        - Removes punctuation (including apostrophes).
        - Lowercases and collapses repeated whitespace.
        - Tokenises on spaces.

    Args:
        raw_name: Raw lodge name as provided by source data.

    Returns:
        NormalisedName containing raw, clean, normalised, and tokens.
    """
    clean = str(raw_name).strip()

    text = clean.lower()
    text = text.replace("’", "'").replace("`", "'")
    text = text.replace("'", " ")
    text = _LODGE_TOKEN_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = text.split(" ") if text else []

    return NormalisedName(raw=raw_name, clean=clean, normalised=text, tokens=tokens)
