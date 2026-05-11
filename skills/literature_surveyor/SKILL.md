---
name: literature_surveyor
description: Pipeline matching key for AutoResearch Stage 2. Aliases to systematic-review.
autoload: false
---

# literature_surveyor (alias skill)

This skill exists primarily as a **pipeline matching key** for AutoResearch's
`pipeline_engine.py`, which looks up `Stage 2` producer via the exact string
`"literature_surveyor"` (see `STAGES[1]` in OMC's `pipeline_engine.py`).

The actual methodology lives in **`systematic-review`** (which is autoloaded
into the system prompt). This file exists so the `skills/` folder structure
matches `profile.yaml.skills[0]` per Talent Market template convention
("skills list corresponding to folder names").

## When this skill activates

Whenever OMC's pipeline engine dispatches a task with skill key
`literature_surveyor`. No additional behavior beyond what `systematic-review`
already specifies.

## See also

- `skills/systematic-review/SKILL.md` — the 9-step workflow (autoloaded)
- `skills/claim-extraction/SKILL.md` — how to extract structured claims
- `skills/conflict-detection/SKILL.md` — cross-paper conflict + gap detection
- `skills/citation-verification/SKILL.md` — pre-submission cite verification

## Why a separate file

OMC's `pipeline_engine._find_employee_by_skill("literature_surveyor")` is a
hard-coded string match against `profile.yaml.skills` (see
`src/onemancompany/core/pipeline_engine.py` STAGES table). The talent
must declare exactly that skill string to be eligible for Stage 2 dispatch
— but the talent-template publishing guide also requires
`skills/<name>/SKILL.md` to exist for every listed skill. This file
reconciles both requirements.
