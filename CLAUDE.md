# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a **standalone Talent Market talent package** (https://one-man-company.com/add) that plugs into the **OneManCompany / AutoResearch** pipeline as the **Stage 2 (Literature Survey)** producer. The repo is intentionally self-contained — it is `git clone`-d into an OMC instance by the platform during hire, and its tools/skills/prompts are then copied into a new employee directory.

Read `README.md` first for the user-facing pitch. This file is about how to develop against the talent without surprises.

## Common commands

```bash
# Local dev setup (uv-first; falls back to plain venv)
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"           # includes pytest, pytest-mock, responses

# Run all 155 unit tests (mocked APIs, no network)
.venv/bin/python -m pytest tests/ -v -m "not network"

# Single test file or test
.venv/bin/python -m pytest tests/test_corpus_store.py -v
.venv/bin/python -m pytest tests/test_corpus_store.py::test_add_idempotent -v

# Live API tests (real arxiv / S2 / OpenAlex) — manual only
.venv/bin/python -m pytest tests/ -v -m network

# Manually invoke any @tool
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from tools.arxiv_search.arxiv_search import arxiv_search
print(arxiv_search.invoke({'query': 'RLHF small models', 'max_results': 3}))
"
```

## How a talent gets into OMC (the deployment path)

1. Talent Market scans `profile.yaml` + repo structure when you submit on https://one-man-company.com/add
2. A buyer (or `hire-from-cv` API call) triggers `git clone` of this repo into `<DATA_ROOT>/talents/literature-surveyor/`
3. `onboarding.execute_hire()` creates an employee dir like `employees/00015/` and:
   - Copies **everything in this repo's root** to the employee dir, **except** `tools/` and `skills/`
   - Copies `prompts/talent_persona.md` to the employee's prompts
   - Copies `vessel/vessel.yaml` to the employee's vessel config
   - For each `skills/<name>/SKILL.md` → copies the folder to `employees/<id>/skills/<name>/`
   - For each `tools/<name>/` → copies the entire dir to `company/assets/tools/<name>/` (a **shared pool**, not per-employee), and adds the new employee to `tool.yaml.allowed_users`
4. On next OMC backend start, `tool_registry.load_asset_tools()` walks `company/assets/tools/`, opens each `tool.yaml`, and (if `type: langchain_module`) imports `<folder>.py` and registers every `@tool` it finds — so `corpus_store.py` registers 5 tools, `claim_store.py` registers 5, `run_metadata.py` registers 3, etc. Total: **23 @tool functions** from 13 tool files.
5. The LangChain agent for the new employee gets these tools via `tool_registry.get_proxied_tools_for(employee_id)`.

If anything in this pipeline is broken, the employee can be hired but its tools won't be callable. See "Real-world gotchas" below.

## Architecture

### 23 tools = 5 functional groups (13 tool files)

| Group | Tools | What it does |
|-------|-------|--------------|
| **Search & retrieval** | `arxiv_search`, `semantic_scholar_search`, `openalex_search`, `parallel_multi_search`, `pdf_extract` | Hit external academic APIs + PDF → markdown |
| **Corpus persistence** | `corpus_add_paper`, `corpus_search`, `corpus_status`, `corpus_list_papers`, `corpus_get_paper` | Append-only JSONL + BM25-lite index. **Layered mode** (opt-in via `LITSURVEY_GLOBAL_CORPUS_DIR`) splits entities (global) from references (per-project). |
| **Claim retrieval** (NEW) | `claim_search`, `claim_list_by_paper`, `claim_get`, `claim_status`, `claim_find_evidence` | Read-side over `claims.jsonl` (written by `extract_claims`). `claim_find_evidence` is the anchor used by `fact_check_rendered_survey`. |
| **Quality gates** | `extract_claims` (v2: section-aware + evidence_quote roundtrip + `applies_to_dims`), `detect_conflicts` (LLM-judged pair contradictions), `verify_citations`, `fact_check_rendered_survey` (per-sentence attribution audit), `self_assess` | Multi-layered fact-grounding. |
| **Provenance** | `run_start`, `run_stage_done`, `run_finalize` | Hash inputs + prompts, log per-stage timings → `<WORKSPACE>/run.json` for cross-run diff |

### The 5 skills

```
skills/
├── systematic-review/         (autoload: true)  ← injected into every system prompt
├── claim-extraction/          (autoload: false) ← LLM loads on demand
├── conflict-detection/        (autoload: false)
├── citation-verification/     (autoload: false)
└── literature_surveyor/       (autoload: false) ← pipeline match key alias
```

### Pipeline match key (do NOT break this)

`profile.yaml.skills[0]` **MUST be the exact string `literature_surveyor`** (underscore, not hyphen). OMC's `src/onemancompany/core/pipeline_engine.py` STAGES table hard-codes:

```python
STAGES[1] = {"id": 2, "skill": "literature_surveyor", "name": "Literature Survey"}
```

`_find_employee_by_skill("literature_surveyor")` does a literal string match against each employee's `profile.yaml.skills`. If you rename this skill, the talent will no longer auto-takeover Stage 2 — you'd have to use `stage_assignments={"2": "<id>"}` per task or modify OMC itself.

The `skills/literature_surveyor/SKILL.md` file exists as a thin alias/pointer because Talent Market scanner requires every `profile.yaml.skills` entry to have a matching `skills/<name>/SKILL.md` folder. The actual methodology lives in `systematic-review` (the autoloaded one).

### Stage 2 data flow

```
[OMC pipeline_engine] dispatches Stage 2 task → LangChain ReAct agent
  ↓
LLM reads systematic-review SKILL (9-step workflow injected via system prompt)
  ↓
LLM calls run_start() → hashes prompts + research question → writes run.json
  ↓
LLM calls corpus_status() — IF layered mode, also corpus_search(scope="global")
  to see if previous projects already pulled relevant papers (reuse short-circuit)
  ↓
LLM calls parallel_multi_search(...) → corpus_add_paper(...) → pdf_extract(...)
  ↓
LLM calls extract_claims(paper_id) per paper — v2 splits by markdown headings,
  emits structured `applies_to_dims` (model_size / dataset / domain / language /
  regime / metric), validates each claim's `evidence_quote` is a substring of
  the source. In layered mode, re-extracting a paper already processed
  short-circuits with status="reused" (0 LLM calls).
  ↓
LLM calls detect_conflicts() → uses applies_to_dims as join key + LLM judge per
  candidate pair → list of Conflict (direct / methodological / scope / temporal)
  ↓
LLM organizes findings + writes stage2.json (Pydantic LiteratureSurveySchema
  with Findings.claim_ids[] anchored to extracted claims) → renders stage2_literature_surveyor.md
  ↓
LLM calls verify_citations(markdown) → existence check via arxiv/Crossref/S2
  ↓
LLM calls fact_check_rendered_survey(...) → per-sentence attribution audit
  using claim_find_evidence as anchor → AttributionAuditSummary (supported /
  partial / unsupported / unverifiable verdicts). Low-confidence verdicts
  trigger automatic context-expansion retry.
  ↓
LLM calls run_finalize(output_paths=[...]) → closes run.json
```

The behavioral contract is `prompts/talent_persona.md`. The procedure is `skills/systematic-review/SKILL.md` (autoloaded). Both are required reading before changing tool behavior.

## Real-world gotchas (every one cost a debug cycle)

### 1. `agent_family` cannot be empty

Talent Market scanner rejects `agent_family: ""` even though the template README implies it's optional. We use **`omctalent`** (OMC-native LangChain employee with `hosting: company`).

### 2. OMC does NOT pip install the talent's `requirements.txt`

When a buyer hires this talent, OMC copies files but does NOT install Python deps. The talent must be self-contained on **stdlib + what OMC already ships** (`requests`, `pydantic`, `langchain-core`).

- `backoff` was removed — we use an inline `_exp_backoff_retry` decorator
- `pymupdf4llm` + `pypdf` are marked **OPTIONAL** in `requirements.txt`; `pdf_extract` does graceful fallback if missing. Buyers should `pip install pymupdf4llm pypdf` in their OMC venv.

If you add a new tool with new deps: inline the impl, verify OMC already ships the dep, or add graceful fallback.

### 3. Sibling-tool imports break inside OMC's tool loader

OMC's `tool_registry.load_asset_tools()` imports each `tools/<name>/<name>.py` as a module named `asset_tool_<folder>` — **`tools` is not a Python package on sys.path**. So `from tools.arxiv_search.arxiv_search import arxiv_search` works in pytest but breaks in production.

`parallel_multi_search.py` resolves sibling tools via a 3-tier fallback in `_resolve_sibling()`:
1. `onemancompany.core.tool_registry.get_tool(name)` — production path
2. `sys.modules['asset_tool_<folder>']` — fallback
3. `from tools.<folder>.<folder> import <name>` — local dev path

**Any new tool that calls sibling @tools must use this same pattern.** Do not import directly.

### 4. Every tool dir needs `tool.yaml`, not just `TOOL.md`

`talent-template` README documents `TOOL.md`, but OMC's scanner needs `tool.yaml` (lowercase, no md) with `id`, `name`, `description`, `type: langchain_module`, `source_talent`. Without `tool.yaml`, tools land in `assets/tools/` but are never registered.

Both `TOOL.md` (LLM-readable) and `tool.yaml` (OMC machine-readable) must exist per tool dir.

### 5. Corpus path resolution — three layers

`corpus_store._corpus_dir()` (per-project / refs side):
1. `LITSURVEY_CORPUS_DIR` env var
2. `<CWD>/corpus/` if CWD looks like an OMC workspace
3. `~/.litsurvey_corpus/` fallback

OMC sets the LangChain agent's CWD to OMC main process dir (not project workspace), so unless `LITSURVEY_CORPUS_DIR` is set, the corpus falls into `~/.litsurvey_corpus/` and is shared across all projects (legacy mode).

`_global_corpus_dir()` (entity side, layered mode only):
- Set via `LITSURVEY_GLOBAL_CORPUS_DIR` env var. When set, paper entities + claim entities live in this global pool; per-project `LITSURVEY_CORPUS_DIR` only stores references. Re-extracting an already-processed paper short-circuits to `status="reused"` (0 LLM calls). Tests use the `layered_corpus` fixture in `tests/conftest.py`.

### 6. OMC's pipeline injects a non-existent `submit_result()` call

`pipeline_engine._dispatch_producer()` task description ends with `"Then call submit_result() with a summary."` — this tool **does not exist** in OMC. `prompts/talent_persona.md` explicitly tells the LLM to ignore this instruction. Full fix requires a PR to Memento-Research.

### 7. `tool_registry` only refreshes on backend start

After a hire, the new talent's tools land on disk in `assets/tools/` but are **not** registered into `tool_registry` until the backend restarts (`./scripts/reset.sh --start` in OMC).

## Profile.yaml — fields the scanner actually cares about

The talent-template README lists fields as optional that the live scanner sometimes rejects when empty. Currently confirmed:

- `agent_family` — must be a concrete framework name (we use `omctalent`)

`upstream_repo_url` and `image_model` are currently `""` and have passed the scanner so far (untested if rejected; remove fields if needed).

## Repo-specific quirks to preserve when editing

- **`requirements.txt` separates required vs optional** — PDF deps are explicitly marked OPTIONAL with install instructions. Don't merge them back.
- **All @tools live as folder/`<name>.py` + `tool.yaml` + `TOOL.md`** — never put a tool in the wrong directory level.
- **Test fixtures**:
  - `isolated_corpus` (tests/conftest.py) — legacy single-tenant mode, monkeypatches `LITSURVEY_CORPUS_DIR` only
  - `layered_corpus` — A2 layered mode, sets both `LITSURVEY_CORPUS_DIR` and `LITSURVEY_GLOBAL_CORPUS_DIR`
  - `two_layered_projects` — two project dirs sharing one global pool, for cross-project reuse tests

  Keep these fixtures working when changing `corpus_store` / `claim_store`.
- **TMAL Citation block in README MUST stay** — license condition (TMAL v1.0). Removing breaks the license.

## When a buyer hires this talent

(Documented in README "Required setup before first task" — repeated here for Claude Code:)

1. Set `OPENROUTER_API_KEY` in OMC's `.env`
2. `.venv/bin/pip install pymupdf4llm pypdf` in OMC venv (enables `pdf_extract`)
3. Optional: `S2_API_KEY` for Semantic Scholar rate limits, `OPENALEX_MAILTO` for OpenAlex polite pool
4. For layered corpus mode (cross-project reuse): set `LITSURVEY_GLOBAL_CORPUS_DIR=<path>` env var

## Cost note

Default `llm_model: anthropic/claude-opus-4.6`. A full Stage 2 run with Opus costs **~$3-5** in OpenRouter credits (verified empirically with Sonnet 4.5: 57 tool calls + 33 papers → ~$1-2 before pdf_extract; full run estimated $2-5). With layered mode + cross-project reuse, subsequent runs on overlapping topics cost roughly the new-paper fraction × baseline.

For experimentation, change `profile.yaml.llm_model` to `openai/gpt-4o-mini` (~20× cheaper) until the talent is tuned, then switch back to Opus/Sonnet for production runs.
