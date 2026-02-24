from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used by the pipeline."""

    project_root: Path
    data_dir: Path
    dicts_dir: Path
    manual_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    paths: PathsConfig
    review_confidence_threshold: float = 0.70
    strict_classical_language: bool = True
    pipeline_version: str = "0.1.0"


def build_default_config(project_root: Path) -> PipelineConfig:
    """Build a default config relative to the repository root."""
    data_dir = project_root / "data"

    paths = PathsConfig(
        project_root=project_root,
        data_dir=data_dir,
        dicts_dir=data_dir / "dicts",
        manual_dir=data_dir / "manual",
        outputs_dir=project_root / "outputs",
    )

    return PipelineConfig(paths=paths)
