from pathlib import Path

from lodge_classifier.language.detect import detect_language_strict


def test_strict_classical_polairs_is_latin(tmp_path: Path) -> None:
    # Arrange minimal dicts directory
    dicts = tmp_path / "dicts"
    dicts.mkdir(parents=True, exist_ok=True)

    (dicts / "classical_latin.csv").write_text("token\npolaris\n", encoding="utf-8")
    (dicts / "classical_greek.csv").write_text("token\nzeus\n", encoding="utf-8")
    (dicts / "welsh_markers.csv").write_text("marker\nyr\n", encoding="utf-8")

    # Act
    res = detect_language_strict(tokens=["polaris"], dicts_dir=dicts)

    # Assert
    assert res.language_primary == "Latin"
