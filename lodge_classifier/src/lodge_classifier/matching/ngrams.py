from __future__ import annotations

from typing import Set


def build_ngrams(tokens: list[str], max_n: int = 5) -> Set[str]:
    """
    Build 1..N-gram phrases from an already-normalised token list.

    This assumes tokens come from normalise_lodge_name and therefore:
        - are lowercased
        - contain no punctuation
        - contain no apostrophes
        - exclude the word "lodge"

    Parameters
    ----------
    tokens : list[str]
        Ordered tokens from NormalisedName.tokens.
    max_n : int
        Maximum n-gram size.

    Returns
    -------
    Set[str]
        Set of space-joined n-gram phrases.
    """
    toks = [t for t in tokens if t and t.strip()]
    out: Set[str] = set()

    for n in range(1, max_n + 1):
        for i in range(0, len(toks) - n + 1):
            out.add(" ".join(toks[i : i + n]))

    return out
