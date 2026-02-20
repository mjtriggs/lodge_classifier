# Lodge Naming Classification System  
## Technical Design Document  
### Dual-Layer Ontology + Theme Model with Strict Option A Language Rules

---

# 1. System Overview

## Objective

Classify 10,000+ lodge names using:

1. A mutually exclusive **Ontology layer** (structural entity type)  
2. A single primary **Theme** (analytical intent)  
3. A strict **Language classification** based on lexical origin (Option A, strict mode)  
4. Deterministic **priority rules** for resolving theme conflicts  

The system must be:

- Fully explainable (every classification traceable to rules and evidence)
- Deterministic
- Scalable
- Extensible
- Strictly philological in language classification
- Capable of incorporating curated manual metadata prior to automated processing

---

# 2. Architectural Overview

The pipeline consists of six ordered passes:

1. **Manual Curation Layer (Pre-Processing Input)**
2. Normalisation & tokenisation  
3. Language detection (Strict Option A)  
4. Ontology signal extraction  
5. Theme extraction + priority resolution  
6. Confidence scoring + structured output  

Each layer is computed independently and reconciled at the end.

---

# 3. MANUAL CURATION LAYER (Pre-Processing)

## Purpose

Before automated classification begins, a small amount of structured metadata is manually added to improve accuracy, stabilise edge cases, and reduce downstream ambiguity.

This layer is especially important for:

- Ambiguous surnames vs places  
- Historic spelling variants  
- Known institutional abbreviations  
- Known language overrides  
- Colonial / imperial contextual interpretation  

---

## Manual Input Dataset (Recommended Fields)

| Column | Description |
|--------|------------|
| lodge_name_raw | Original lodge name |
| curated_notes | Free text notes |
| curated_language_override | Optional forced language |
| curated_ontology_hint | Optional structural hint |
| curated_theme_hint | Optional theme hint |
| curated_alias | Normalised or alternate known form |
| curated_priority_flag | Marks sensitive or historically important cases |

These fields are optional but override automation when present.

---

## Override Hierarchy

Manual layer overrides:

- Language detection
- Ontology assignment
- Theme resolution

All overrides must:

- Be logged
- Set flag: `OVERRIDE_APPLIED`
- Record curator ID and timestamp

---

# 4. ONTOLOGY LAYER  
### (Structural – What type of entity is referenced?)

## Design Principles

- Exactly one `ontology_primary`
- Optional `ontology_secondary`
- Structural classification only
- Independent of expressive intent

---

## PRS_ — Person

| Code | Definition |
|------|------------|
| PRS_HIS | Historic person (non-royal) |
| PRS_ROY | Royal or aristocratic figure |
| PRS_REL | Religious person (Saints, Bishops) |
| PRS_FIC | Fictional or literary character |
| PRS_MYTH | Mythological / classical figure |

---

## LOC_ — Place

| Code | Definition |
|------|------------|
| LOC_CTY | City, town, village |
| LOC_REG | Region, county, territory |
| LOC_LAN | Natural landmark |
| LOC_EST | Estate, manor, priory, abbey |
| LOC_BDG | Historic building / civic structure |

---

## GRP_ — Collective / Body

| Code | Definition |
|------|------------|
| GRP_EDU | Educational institution / alumni |
| GRP_MIL | Military unit / regiment / armed forces |
| GRP_JOB | Profession / trade |
| GRP_INT | Special interest / hobby / clubs |
| GRP_MAS | Masonic rank or administrative body |
| GRP_NAT | Nationality / ethnic identity |

---

## NAT_ — Natural Entity

| Code | Definition |
|------|------------|
| NAT_ANI | Animal |
| NAT_AST | Astronomical body |
| NAT_BOT | Botanical |
| NAT_GEO | Natural feature (if not clearly named place) |

---

## OBJ_ — Object / Symbol

| Code | Definition |
|------|------------|
| OBJ_STR | Physical structure |
| OBJ_EMB | Emblem / symbolic object |
| OBJ_SCI | Scientific instrument or technical term |
| OBJ_MYTH | Mythic object |

---

## ABS_ — Abstract Concept

| Code | Definition |
|------|------------|
| ABS_VRT | Virtue / moral quality |
| ABS_PHI | Philosophical / intellectual concept |
| ABS_FRAT | Fraternal / relational ideal |
| ABS_LAT | Untranslated Latin phrase (structural classification only) |

---

## UNK_

Unresolved / requires review.

---

# 5. THEME LAYER  
### (Analytical – What idea is being signalled?)

## Design Principles

- One `theme_primary`
- Optional `theme_secondary`
- Expressive intent dominates structural reference
- Resolved via priority rules

---

## Theme Categories

- Virtue / Moral Ideal  
- Religious  
- Royal / Aristocratic  
- Military / Service  
- Educational / Institutional  
- Masonic / Administrative  
- Professional / Trade  
- Clubs / Association  
- Mythological / Classical  
- Literary / Cultural  
- Scientific / Intellectual  
- Imperial / Colonial  
- Geographic / Civic  
- Nature  
- Symbolic / Esoteric  
- Modernity / Progress  
- Fraternal  
- Heritage  
- Unknown  

---

# 6. LANGUAGE LAYER (Strict Option A)

## Design Principle

Language classification is based strictly on **lexical origin**, not modern usage.

Example classifications:

- Polaris → Latin  
- Neptune → Latin  
- Zeus → Greek  
- Aurea Norma → Latin  
- Craig yr Hesg → Welsh  

Anglicised usage does not override original origin.

---

## Supported Categories

- English  
- Latin  
- Greek  
- Welsh  
- French  
- German  
- Spanish  
- Italian  
- Afrikaans  
- Dutch  
- Gaelic  
- Unknown  

---

## Strict Detection Rules

1. Classical lexicon match → Latin or Greek  
2. Welsh lexical markers → Welsh  
3. Romance language articles + lexicon → French / Spanish / Italian  
4. Germanic lexicon → German / Dutch / Afrikaans  
5. Mixed unresolved → Unknown  

Manual overrides take precedence.

---

# 7. PRIORITY RULES (Theme Resolution)

When multiple signals appear:

1. Virtue / Abstract  
2. Religious Person  
3. Royal / Aristocratic  
4. Military  
5. Educational / Institutional  
6. Masonic / Administrative  
7. Professional / Clubs  
8. Mythological / Classical  
9. Geographic (fallback)  

Expressive intent dominates administrative location.

---

# 8. Dataset Structure

## Required Columns

- lodge_name_raw  
- lodge_name_clean  
- ontology_primary  
- ontology_secondary  
- theme_primary  
- theme_secondary  
- language_primary  
- confidence_theme  
- confidence_language  
- flags  

---

## Recommended Audit Columns

- ontology_confidence  
- language_confidence  
- ontology_rule_id  
- theme_rule_id  
- language_rule_id  
- evidence_json  
- review_required  
- pipeline_version  
- created_at  

---

# 9. Confidence Scoring

Each layer scored independently.

### Ontology Confidence
- Exact match strength  
- Single candidate clarity  
- Structural disambiguation  

### Theme Confidence
- Priority resolution strength  
- Competing signal presence  

### Language Confidence
- Exact lexicon match  
- Override usage  
- Multi-token consistency  

---

# 10. End-to-End Processing Flow

```text
Manual Curation Layer
    ↓
Raw Name
    ↓
Normalise
    ↓
Strict Language Detection (Lexical Origin)
    ↓
Ontology Signal Extraction
    ↓
Ontology Primary Assignment
    ↓
Theme Extraction
    ↓
Theme Priority Resolution
    ↓
Confidence Scoring
    ↓
Apply Overrides (if any)
    ↓
Final Output Dataset
