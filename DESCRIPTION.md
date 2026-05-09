# Literature Surveyor

> Systematic literature review specialist — produces evidence-grounded surveys with verified citations.

## Overview

Built for **AutoResearch's adversarial research pipeline** (Stage 2: Literature Survey). Takes a refined research question from Stage 1 and produces:

- **Structured survey JSON** (Pydantic-validated) — taxonomy, findings, conflicts, open questions, gaps
- **Human-readable markdown** — rendered from the JSON
- **Per-project corpus** (`corpus.jsonl`) — papers + claims persisted for retry / downstream stages
- **Citation audit** — every cite verified against arxiv / Semantic Scholar / OpenAlex; unverified blocked

## Use Cases

- **Pre-research scoping** — Before writing a grant or starting an experiment, get a 30-paper landscape of a sub-field with identified open questions.
- **Conflict mining** — Find pairs of papers with contradictory findings on the same setup. Useful as a Stage 3 idea generation seed.
- **Reproducibility audit** — Generate a survey of methodologies used for a benchmark; spot which methods are missing baselines.
- **Replacement for vanilla LLM "what's the literature on X"** — Verified, dated, sourced.

## What's Different

| Vanilla LLM | This talent |
|---|---|
| Invents arxiv IDs | `verify_citations` tool blocks pre-submission |
| Knowledge cuts off at training | Live arxiv / Semantic Scholar / OpenAlex APIs |
| Free-form prose | Pydantic schema enforces structure |
| Forgets between conversations | Per-project corpus.jsonl persists |

## Demo

(Add screenshots after first real run.)

## Tools Provided

9 LangChain `@tool` functions, all auto-installed on hire:

`arxiv_search`, `semantic_scholar_search`, `openalex_search`, `parallel_multi_search`, `pdf_extract`, `corpus_store` (5 actions: add/search/status/list/get), `extract_claims`, `verify_citations`, `self_assess`.

## Skills Loaded

`systematic-review` (autoloaded — 9-step workflow), `claim-extraction`, `conflict-detection`, `citation-verification`.

---

> **Content Policy** — This description is publicly visible on Talent Market. No illegal / political / explicit content. All external links must point to legitimate, safe resources.
