---
name: self_assess
description: Decide whether current corpus is sufficient to write the survey. Returns verdict + suggested next action.
---

# self_assess

Heuristic check used in the middle of Stage 2 to decide whether to keep searching, start writing, or give up.

## When to use

- After each round of `parallel_multi_search` + `corpus_add_paper`
- Before deciding to call `extract_claims` on the whole corpus

## Returns one of three verdicts

| Verdict | When | Suggested action |
|---|---|---|
| `insufficient` | < `min_papers` (default 10) OR no full text | `search_more` — try different queries |
| `sufficient` | between min and max | `extract_claims` — start writing |
| `saturated` | >= `max_papers` (default 30) | `extract_claims` — stop searching, prune |

## Why both heuristic and LLM versions

This tool is **purely heuristic** (paper count, year spread, source diversity) — fast and deterministic. For nuanced topical coverage assessment, use an LLM call directly with the corpus contents.

## Example

```python
verdict = self_assess(
    research_question="RLHF degradation in small reasoning models",
    min_papers=10,
    max_papers=30,
)

if verdict["verdict"] == "insufficient":
    # try one more parallel search round
    ...
elif verdict["verdict"] == "sufficient":
    # extract claims from each paper, then write
    ...
elif verdict["verdict"] == "saturated":
    # stop searching, prune to top 30 by relevance
    ...
```

## Tunable thresholds

Override defaults via args:
- `min_papers` — corpus too small below this (default 10)
- `max_papers` — corpus saturated above this (default 30)

## What it doesn't do

- Doesn't judge **topical** sufficiency (whether all sub-topics of RQ are covered) — that needs an LLM call with corpus list
- Doesn't recommend specific queries to add — leave that to the LLM after reading the verdict
