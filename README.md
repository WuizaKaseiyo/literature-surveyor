# Literature Surveyor

Systematic literature review specialist for the [AutoResearch](https://github.com/Memento-Teams/Memento-Research) (OneManCompany) adversarial research pipeline. Designed as Stage 2 — produces evidence-grounded literature surveys with verified citations and identified gaps for downstream Stage 3 idea generation.

> **Talent Market compliant** — packaged per [1mancompany/talent-template](https://github.com/1mancompany/talent-template) v1.

## What it does

- Multi-source academic search (arxiv / Semantic Scholar / OpenAlex)
- PDF → markdown extraction (pymupdf4llm, preserves headings + tables)
- Per-project corpus persistence (BM25-lite local index)
- Structured claim extraction with evidence spans
- Cross-paper conflict detection
- Citation verification — every cite checked against real APIs, hallucinated IDs blocked
- Pydantic-validated structured output (JSON + markdown)

## Why use this instead of asking GPT directly

| Symptom of vanilla LLM | What this talent does |
|---|---|
| Invents arxiv IDs that don't exist | `verify_citations` blocks unverified cites before submitting |
| Cites papers truncated at training cutoff | Live arxiv / Semantic Scholar API |
| Output is unstructured prose | Pydantic schema → JSON + rendered markdown |
| Repeated work on similar topics | Per-project `corpus.jsonl` reusable across runs |
| No way to audit what was searched | Every API call logged, corpus inspectable |

## Install

In your OMC instance, hire via Talent Market or `POST /api/candidates/hire-from-cv` with this repo URL.

After hiring, take over Stage 2 by setting `stage_assignments={"2": "<new_employee_id>"}` when creating a CEO task.

## Local development

```bash
git clone https://github.com/<you>/literature-surveyor.git
cd literature-surveyor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pytest tests/ -v
```

## Configure

Set environment variables (or via OMC settings UI per `manifest.json`):

| Var | Required | Purpose |
|---|---|---|
| `S2_API_KEY` | optional | Semantic Scholar — higher rate limits |
| `OPENALEX_MAILTO` | optional | OpenAlex polite pool (faster) |
| `LITSURVEY_CORPUS_DIR` | optional | Override corpus location (default `<workspace>/corpus/`) |

## Architecture

9 LangChain `@tool` functions, 4 folder-based skills, one Pydantic schema. See `prompts/talent_persona.md` for behavioral contract and `skills/systematic-review/SKILL.md` for the 9-step workflow.

## License

Apache-2.0. See [LICENSE](./LICENSE).

## Citation

Built using the [Talent Market](https://one-man-company.com) template by Zhengxu Yu / [1mancompany](https://github.com/1mancompany).

```
@software{talentmarket,
  title  = {Talent Market - AI Agent Marketplace},
  author = {Zhengxu Yu},
  url    = {https://one-man-company.com},
  year   = {2026}
}
```

Citation verification design inspired by [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) (`tools/semantic_scholar.py`).
