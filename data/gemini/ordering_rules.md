# Gemini Custom Ordering Rules

These rules define the precedence for primary and secondary ontology classification within the `gemini` data experiments. These override or refine the default pipeline logic.

## Priority 1: High-Signal Identifiers
- **Interest > Location:** If a lodge name contains both a special interest (e.g., Golf, Fairway, Sports) and a location (e.g., Dorset, London), the **Interest** (`GRP_INT`) is Primary and the **Location** is Secondary.
- **Profession > Collective:** Direct references to trades or professions (`GRP_JOB`) are Primary.

## Priority 2: Abstract vs. Physical
- **Abstract > Emblem:** Virtues, philosophical concepts, or mottos (`ABS_VRT`, `ABS_PHI`) take precedence over physical emblems or symbols (`OBJ_EMB`).

## Priority 3: Structural Specificity
- **Building > Place:** Specific building types or historic structures (`LOC_BDG`) are prioritized over general city or town names (`LOC_CTY`).

## Priority 4: Historical/Royal
- **Royal/Historic Person:** Continues to be a high-signal primary classification.

## General Ordering (Descending Priority)
1. `GRP_JOB` / `GRP_INT` (Professional or Interest groups)
2. `PRS_HIS` / `PRS_ROY` / `PRS_REL` (Specific People)
3. `ABS_VRT` / `ABS_PHI` (Abstract Concepts)
4. `LOC_BDG` / `LOC_REL` (Specific Buildings)
5. `OBJ_EMB` / `OBJ_STR` (Physical Objects/Emblems)
6. `LOC_CTY` / `LOC_REG` (General Locations - now lower priority)
7. `NAT_...` (Natural Entities)
