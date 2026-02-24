from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DictCache:
    """Cache CSV dictionaries loaded from disk.

    The pipeline uses many small CSV dictionaries. Loading them for every record is
    expensive and can make runs inconsistent. This cache loads each CSV once per
    run and returns normalised, lowercased sets for fast membership checks.
    """

    dicts_dir: Path
    _sets: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def load_set(self, filename: str, column: str) -> set[str]:
        """Load a required dictionary file as a lowercased set."""
        key = (filename, column)
        if key in self._sets:
            return self._sets[key]

        path = self.dicts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required dictionary file: {path}")

        df = pd.read_csv(path)
        if column not in df.columns:
            raise ValueError(f"Expected column '{column}' in {path}")

        values = set(df[column].astype(str).str.strip().str.lower())
        self._sets[key] = values
        return values

    def try_load_set(self, filename: str, column: str) -> set[str]:
        """Load an optional dictionary file as a lowercased set.

        Returns an empty set if the file does not exist.
        """
        path = self.dicts_dir / filename
        if not path.exists():
            return set()

        return self.load_set(filename=filename, column=column)

    def meta(self) -> dict[str, Any]:
        """Return lightweight metadata for debugging."""
        return {
            "dicts_dir": str(self.dicts_dir),
            "loaded_sets": [
                {"filename": f, "column": c, "size": len(s)} for (f, c), s in self._sets.items()
            ],
        }
