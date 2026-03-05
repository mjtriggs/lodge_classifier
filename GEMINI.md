# Gemini CLI Project Context

This file contains foundational mandates and context for Gemini CLI's operations in this repository.

## Project Overview
- **Name:** lodge-classifier
- **Purpose:** Fraternal lodge name classification (Language + Ontology + Theme)
- **Tech Stack:** Python 3.11+, Pandas, Loguru, RapidFuzz

## Operational Mandates
- **Classification Logic:** Follow the rules defined in `src/lodge_classifier/ontology/classify.py` and `src/lodge_classifier/theme/classify.py`.
- **Taxonomy:** Strictly adhere to `data/taxonomy.md` for category codes and definitions.
- **Data Integrity:** Ensure `evidence_json` is always populated for auditability in any classification updates.
- **Testing:** Run `pytest` to verify pipeline integrity after changes.

## Key Directories
- `data/dicts/`: Source dictionaries for matching.
- `src/lodge_classifier/`: Core logic and pipeline implementation.
- `tests/`: Unit and smoke tests.

## Workspace Conventions
- Use `loguru` for all logging.
- Follow PEP 8 (handled by `ruff` and `black`).
- Type hints are required for new logic.
