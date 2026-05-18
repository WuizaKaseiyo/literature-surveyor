---
name: claim_store
description: Per-corpus claim retrieval & BM25 search over claims.jsonl. 5 read-side actions — search, list_by_paper, get, status, find_evidence.
---

# claim_store

Reads the `claims.jsonl` written by `extract_claims`. extract_claims is the
producer; this module is the consumer. Until this tool existed, claims were
write-only data — no tool could look them up at runtime, so the conflict
detector and fact checker had to re-derive everything from raw paper text.

This file exposes 5 separate `@tool` functions:

| Function | Purpose |
|---|---|
| `claim_status` | Total counts, by-type distribution, papers covered |
| `claim_search` | BM25 over all claims (filterable by paper / type) |
| `claim_list_by_paper` | Every claim for a given paper id |
| `claim_get` | One claim by id (`<paper_id>#claim-N`) |
| `claim_find_evidence` | Best-matching claims in a specific paper for a rendered sentence — **the anchor used by fact_check_rendered_survey** |

## Storage layout

```
<CORPUS_DIR>/
├── claims.jsonl         # source of truth (written by extract_claims)
└── claims_index.json    # BM25 index over claim_text + evidence_quote
```

CORPUS_DIR resolution mirrors `corpus_store` (same env var, same workspace
heuristic). The two tools share the same directory.

## Index freshness

`claims_index.json` is rebuilt lazily on every read whenever `claims.jsonl` has
been modified more recently. `extract_claims` does NOT need to know about the
index — appending to claims.jsonl is enough; the next claim_store call rebuilds.

## When to use which

- **Stage 2 step 7 (post-extraction sanity check)**: `claim_search(finding_text)` to confirm at least one extracted claim backs each finding before writing it.
- **Stage 2 step 9 (fact check)**: fact_check_rendered_survey will call `claim_find_evidence(sentence, paper_id)` first; only fall back to BM25 over raw paper text if no match comes back.
- **Conflict detection (planned)**: iterate `claim_list_by_paper` for each paper, build pairwise candidates filtered by `applies_to_dims`.

## BM25 details

Same scoring as `corpus_store` (`k1=1.5`, `b=0.75`, doc length normalization).
Tokens are alphanumeric + apostrophe-aware. Stop words removed. No stemming.
Documents are `claim_text + " " + evidence_quote`.
