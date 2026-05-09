---
name: corpus_store
description: Per-project paper corpus with BM25-lite search. 5 actions in one tool group — add, search, status, list, get.
---

# corpus_store

Persistent paper storage for the current project. Each Stage 2 task should `corpus_status` first to check if previous rounds (or earlier in the same session) already gathered relevant papers.

This file exposes 5 separate `@tool` functions:

| Function | Purpose |
|---|---|
| `corpus_status` | Get counts + distributions; **call first** in Stage 2 |
| `corpus_search` | BM25-lite local search — try this BEFORE external APIs |
| `corpus_add_paper` | Add a paper (idempotent on id) |
| `corpus_list_papers` | List papers, optionally filter by source |
| `corpus_get_paper` | Fetch one paper's full record |

## Storage layout

```
<CORPUS_DIR>/
├── papers.jsonl         # one paper per line
├── corpus_index.json    # token freq + doc freq for BM25
└── claims.jsonl         # extracted claims (written by extract_claims tool)
```

## Where CORPUS_DIR resolves to

In priority order:
1. `LITSURVEY_CORPUS_DIR` env var (explicit override)
2. `<CWD>/corpus/` if CWD looks like an OMC project workspace
3. `~/.litsurvey_corpus/` (fallback)

OMC sets CWD to project workspace before invoking tools, so each project gets an isolated corpus by default.

## Paper schema

When calling `corpus_add_paper`, pass:

```python
{
    "id": "2401.12345",                      # required — unique identifier
    "title": "Attention Is All You Need",    # required
    "authors": ["Vaswani", ...],
    "year": 2017,
    "venue": "NeurIPS",
    "abstract": "...",
    "full_text_md": "...",                   # from pdf_extract
    "source_query": "transformer attention", # which search found it
    "source": "arxiv",                       # "arxiv" | "semantic_scholar" | "openalex" | "user_upload"
}
```

Idempotent: re-adding same `id` is a no-op (returns `skipped_duplicate`).

## BM25 details

Standard BM25 with `k1=1.5`, `b=0.75`. Stop words removed. Stemming NOT applied (intentional — academic vocab is precise).

For corpora >100 papers consider migrating to chromadb (future v0.4 feature).
