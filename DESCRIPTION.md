# Literature Surveyor

> Systematic literature review specialist — produces evidence-grounded surveys with verified citations.

## Overview

Built for **AutoResearch's adversarial research pipeline** (Stage 2: Literature Survey). Takes a refined research question from Stage 1 and produces:

- **Structured survey JSON** (Pydantic-validated) — taxonomy, findings (with backing `claim_ids`), conflicts, open questions, gaps
- **Human-readable markdown** — rendered from the JSON with optional `> evidence:` footnotes pointing to verbatim source quotes
- **Per-project corpus + shared global pool** — same arxiv paper is fetched + claim-extracted **once across all projects** (opt-in layered mode); each project's view is a thin `refs.jsonl`
- **Citation audit + final-text fact check** — every cite verified against arxiv / Semantic Scholar / OpenAlex; each finding then anchored to its declared extracted claim's verbatim evidence quote
- **Cross-paper conflict detection** — `applies_to_dims` (structured scope) as the cheap join key, LLM judge as the decisive step

## Use Cases

- **Pre-research scoping** — Before writing a grant or starting an experiment, get a 30-paper landscape of a sub-field with identified open questions.
- **Conflict mining** — Find pairs of papers with contradictory findings on the same setup. Useful as a Stage 3 idea generation seed.
- **Reproducibility audit** — Generate a survey of methodologies used for a benchmark; spot which methods are missing baselines.
- **Replacement for vanilla LLM "what's the literature on X"** — Verified, dated, sourced.

## What's Different

| Vanilla LLM | This talent |
|---|---|
| Invents arxiv IDs | `verify_citations` tool blocks pre-submission |
| Cites real but wrong papers | `fact_check_rendered_survey` checks final claims against cited paper text |
| Knowledge cuts off at training | Live arxiv / Semantic Scholar / OpenAlex APIs |
| Free-form prose | Pydantic schema enforces structure |
| Forgets between conversations | Per-project corpus.jsonl persists |

## Demo

(Add screenshots after first real run.)

## Tools Provided

23 LangChain `@tool` functions, all auto-installed on hire:

`arxiv_search`, `semantic_scholar_search`, `openalex_search`, `parallel_multi_search`, `pdf_extract`, `corpus_store` (5 actions: add/search/status/list/get), `extract_claims` (v1 + v2 + reuse short-circuit), `claim_store` (5 actions: search/list_by_paper/get/status/find_evidence), `verify_citations`, `fact_check_rendered_survey` (markdown or survey_json input + claim anchor + multi-cite rollup + low-confidence retry), `detect_conflicts`, `self_assess`, `run_metadata` (3 actions: start/stage_done/finalize).

## Skills Loaded

`systematic-review` (autoloaded — 9-step workflow), `claim-extraction`, `conflict-detection`, `citation-verification`.

---

> **Content Policy** — This description is publicly visible on Talent Market. No illegal / political / explicit content. All external links must point to legitimate, safe resources.
