# Literature Surveyor

[![CI](https://github.com/WuizaKaseiyo/literature-surveyor/actions/workflows/ci.yml/badge.svg)](https://github.com/WuizaKaseiyo/literature-surveyor/actions/workflows/ci.yml)

Systematic literature review specialist for the [AutoResearch](https://github.com/Memento-Teams/Memento-Research) (OneManCompany) adversarial research pipeline. Designed as **Stage 2** — produces evidence-grounded literature surveys with verified citations and identified gaps for downstream Stage 3 idea generation.

> **Talent Market compliant** — packaged per [1mancompany/talent-template](https://github.com/1mancompany/talent-template) v1. **End-to-end verified** in a real OMC instance (28 unit tests + integration sandbox + live hire).

---

## What it does

- **Multi-source academic search** — arxiv (free) + Semantic Scholar (citation-aware, sorted by influence) + OpenAlex (240M works, all fields), concurrent via `parallel_multi_search` with cross-source dedup
- **PDF → markdown extraction** — `pymupdf4llm` preserves headings, tables, math; auto-fallback to `pypdf`
- **Per-project corpus** — append-only `papers.jsonl` + BM25-lite index; reusable across rounds
- **Structured claim extraction** — each paper → 5-15 typed claims with `evidence_span` (section + table refs) + `applies_to` (scope qualifier)
- **Cross-paper conflict detection** — finds same-setting / contradictory claims (4 levels: direct / methodological / scope / temporal)
- **Citation verification** — every `[Author Year, arxiv:X / doi:Y / S2:Z]` cite checked against real APIs; hallucinated IDs blocked pre-submission
- **Pydantic-validated output** — strict `LiteratureSurveySchema` JSON + rendered markdown

## Why use this instead of asking GPT directly

| Symptom of vanilla LLM | What this talent does |
|---|---|
| Invents arxiv IDs that don't exist | `verify_citations` blocks unverified cites before submitting |
| Cites papers truncated at training cutoff | Live arxiv / Semantic Scholar / OpenAlex |
| Output is unstructured prose | Pydantic schema → strict JSON + rendered markdown |
| Repeated work on similar topics | Per-project `corpus.jsonl` reusable across runs |
| No way to audit what was searched | Every API call logged, corpus inspectable |

---

## Quick install (recommended path)

In your OMC instance frontend, **Talent Market → Add Talent** and paste:

```
https://github.com/WuizaKaseiyo/literature-surveyor
```

OMC's onboarding flow will:

1. `git clone` this repo into `.onemancompany/talents/literature-surveyor/`
2. `execute_hire()` creates a new employee directory (e.g. `00015`)
3. `copy_talent_assets` copies skills/prompts/vessel/manifest into the employee
4. `register_tool_user` puts 9 tool dirs into `company/assets/tools/` (with `tool.yaml` for tool_registry registration)
5. `tool_registry.load_asset_tools()` registers 13 LangChain `@tool` functions on next backend start
6. Backend registers a LangChain agent for the new employee

### Or, programmatic install

```bash
curl -X POST http://localhost:8000/api/candidates/hire-from-cv \
  -H 'Content-Type: application/json' \
  -d '{
    "cv": {
      "name": "Literature Surveyor Pro",
      "role": "Researcher",
      "talent_id": "literature-surveyor",
      "source_repo": "https://github.com/WuizaKaseiyo/literature-surveyor",
      "skills": ["literature_surveyor", "systematic-review", "claim-extraction", "conflict-detection", "citation-verification"],
      "llm_model": "anthropic/claude-sonnet-4-5",
      "temperature": 0.3,
      "hosting": "company",
      "auth_method": "api_key",
      "api_provider": "openrouter"
    }
  }'
```

---

## ⚠️ Required setup before first task

These are **not optional** — the talent will fail or degrade without them.

### 1. API keys (in OMC's `.env`)

| Key | Required? | Where to get it |
|-----|-----------|-----------------|
| `OPENROUTER_API_KEY` | **REQUIRED** | https://openrouter.ai/keys (~$5 covers 4-7 full surveys) |
| `S2_API_KEY` | strongly recommended | https://www.semanticscholar.org/product/api (free, ~5min approval) — without it Semantic Scholar throttles aggressively |
| `OPENALEX_MAILTO` | strongly recommended | Your email — joins OpenAlex polite pool for stability |

Recommended `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-<your-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

S2_API_KEY=<your-key>
OPENALEX_MAILTO=you@example.com
```

### 2. Install PDF dependencies in OMC venv

OMC does **not** auto-install talent `requirements.txt`. Without these, `pdf_extract` returns a graceful error and Stage 2 falls back to using only abstracts (degraded quality):

```bash
cd /path/to/your/Memento-Research
.venv/bin/pip install pymupdf4llm pypdf
```

### 3. Stage 2 dispatch

Once hired, route Stage 2 work to the new employee. Two ways:

```bash
# Option A: explicit assignment per task
curl -X POST http://localhost:8000/api/ceo/task \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "Survey RLHF degradation in small reasoning models 1B-7B",
    "stage_assignments": {"2": "<new_employee_id>"}
  }'

# Option B: delete the old default Stage 2 employee (00007 in stock AutoResearch)
rm -rf .onemancompany/company/human_resource/employees/00007/
./scripts/reset.sh --start
# Now pipeline_engine._find_employee_by_skill('literature_surveyor') auto-picks the new employee
```

---

## Tools provided (13)

| Tool | Purpose |
|---|---|
| `arxiv_search` | Search arxiv.org export API — free, no key |
| `semantic_scholar_search` | Search Semantic Scholar (sorted by citation count) |
| `openalex_search` | Search OpenAlex (cross-domain, 240M works) |
| `parallel_multi_search` | Concurrent 3-source search with dedup by arxiv_id/DOI/title |
| `pdf_extract` | PDF (URL or path) → markdown via pymupdf4llm |
| `corpus_add_paper` | Add paper to project corpus (idempotent) |
| `corpus_search` | BM25-lite local search over already-fetched papers |
| `corpus_status` | Counts + distributions; **call first** in Stage 2 |
| `corpus_list_papers` | List papers with optional source filter |
| `corpus_get_paper` | Fetch full record by ID |
| `extract_claims` | LLM-based structured claim extraction (5-15 per paper) |
| `verify_citations` | Pre-submission cite verification — blocks hallucinations |
| `self_assess` | Heuristic verdict on whether corpus is sufficient |

## Skills loaded (4)

- **`systematic-review`** (autoloaded) — 9-step PRISMA-inspired workflow
- **`claim-extraction`** — claim schema + extraction protocol
- **`conflict-detection`** — cross-paper conflict + gap detection
- **`citation-verification`** — cite format spec + verification flow

---

## LLM configuration

By default the talent uses **two models** for cost optimization:

| Component | Model | Why |
|---|---|---|
| Main reasoning (90% of calls) | `anthropic/claude-sonnet-4-5` | Long-context + structured reasoning for surveys |
| Claim extraction (per paper) | `openai/gpt-4o-mini` | Cheap structured extraction, ~10× cost reduction |

Both go through OpenRouter so only `OPENROUTER_API_KEY` is needed.

To change the main model, edit `profile.yaml`:

```yaml
llm_model: anthropic/claude-haiku-4-5    # cheaper alternative
# or
llm_model: openai/gpt-4o
```

OMC hot-reloads profile.yaml within a few seconds — no restart needed.

---

## Cost estimate per Stage 2 run

At Claude Sonnet 4.5 prices (~$3/M input, $15/M output) + gpt-4o-mini for extraction:

| Component | Tokens (est.) | Cost |
|---|---|---|
| Main ReAct loop (~30 steps) | 200K in / 30K out | $0.60-1.00 |
| Claim extraction (×30 papers) | 600K in / 60K out | $0.10-0.15 |
| **Total per survey** | | **~$0.70-1.20** |

Academic API calls (arxiv / S2 / OpenAlex / Crossref) are all free.

---

## Local development

```bash
git clone https://github.com/WuizaKaseiyo/literature-surveyor.git
cd literature-surveyor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

28 unit tests cover all tools (mocked APIs) + Pydantic schema. Run live API tests with `pytest -m network` (hits real arxiv/S2/OpenAlex).

### Repo layout

```
literature-surveyor/
├── profile.yaml                    # Talent identity (skill 'literature_surveyor' is the OMC pipeline match key)
├── manifest.json                   # Frontend settings UI schema
├── prompts/talent_persona.md       # Role boundaries + behavioral contract
├── vessel/vessel.yaml              # Per-talent timeout (1800s) + iterations (60)
├── skills/
│   ├── systematic-review/          # autoload: true
│   ├── claim-extraction/
│   ├── conflict-detection/
│   └── citation-verification/
├── tools/
│   ├── manifest.yaml               # custom_tools list
│   └── <name>/                     # one folder per tool group
│       ├── tool.yaml               # OMC tool_registry metadata (id/desc/source)
│       ├── TOOL.md                 # talent-template docs (LLM-readable)
│       └── <name>.py               # @tool implementation(s)
├── schemas/                        # Pydantic LiteratureSurveySchema
├── examples/                       # Reference surveys (TODO: hand-write 2)
└── tests/                          # 28 unit tests
```

---

## Known caveats

1. **`pdf_extract` requires extra deps in OMC venv** — `pip install pymupdf4llm pypdf` (graceful degrade to error if missing)
2. **Corpus directory falls back to `~/.litsurvey_corpus/` if CWD isn't a project workspace** — set `LITSURVEY_CORPUS_DIR` env var to override per-project
3. **OMC `pipeline_engine` task description still mentions a non-existent `submit_result()` tool** — talent_persona.md teaches the LLM to ignore it; full fix requires a PR to OMC itself
4. **OMC tool_registry only refreshes on backend start** — after hire, restart the backend (`./scripts/reset.sh --start`) so the 13 talent tools register
5. **Skill matching is dict-order first-wins** — if you keep both old `00007` and new employee with `literature_surveyor` skill, old wins. Use `stage_assignments` or remove old employee.

See [Memento-Research](https://github.com/Memento-Teams/Memento-Research) for OMC architecture details.

---

## Configuration reference

| Env var | Required? | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | **yes** | Main LLM + claim extraction |
| `S2_API_KEY` | recommended | Semantic Scholar higher rate limits |
| `OPENALEX_MAILTO` | recommended | OpenAlex polite pool |
| `LITSURVEY_CORPUS_DIR` | optional | Override corpus location (default: `<workspace>/corpus/` or `~/.litsurvey_corpus/`) |

---

## Architecture

9 LangChain `@tool` files (exporting 13 tools total — `corpus_store.py` exports 5 sub-tools), 4 folder-based skills, one Pydantic schema, ~3000 LOC including tests. See `prompts/talent_persona.md` for the behavioral contract and `skills/systematic-review/SKILL.md` for the 9-step workflow.

Designed to slot into OMC's `pipeline_engine.py` Stage 2 — receives task description from upstream, writes `stage2_literature_surveyor.md` + `stage2.json` to project workspace, returns to pipeline for critic review.

---

## License

[Talent Market Attribution License (TMAL) v1.0](./LICENSE) — free for commercial use, requires retaining the Citation section below in derivative works.

Citation verification + Semantic Scholar tool design adapted from [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) (Apache-2.0) — see `tools/semantic_scholar_search/semantic_scholar_search.py` for the original copyright notice.

---

## Citation

> **DO NOT REMOVE** — required by the [Talent Market Attribution License](./LICENSE).

This talent was built using the [Talent Market](https://one-man-company.com) template by [Zhengxu Yu](mailto:yuzxfred@gmail.com) / [1mancompany](https://github.com/1mancompany).

```bibtex
@software{talentmarket,
  title  = {Talent Market - AI Agent Marketplace},
  author = {Zhengxu Yu},
  email  = {yuzxfred@gmail.com},
  url    = {https://one-man-company.com},
  year   = {2026}
}
```

If you publish or deploy a talent based on this template, please keep this section intact in your README or equivalent documentation.
