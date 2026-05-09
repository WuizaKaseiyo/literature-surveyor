---
name: arxiv_search
description: Search arxiv.org for papers (free, no API key). Returns structured metadata sorted by relevance.
---

# arxiv_search

Query the arxiv.org export API for preprints.

## When to use

- Stage 2 (Literature Survey) — primary search for ML / AI / CV / NLP preprints
- Need newest work — arxiv covers preprints faster than venues
- Verify a citation actually exists on arxiv

## When NOT to use

- For papers older than ~2010 → prefer `openalex_search`
- For citation graph / "who cites X" → use `semantic_scholar_search`
- For non-CS fields (medicine, biology) → use `openalex_search` or `semantic_scholar_search`

## Cost

- Free
- Rate-limited: ~1 request per 3 seconds. Tool inserts 1s polite-sleep after each call.
- Hard cap: 100 results per call

## Example

```python
arxiv_search(query="RLHF small language models", max_results=20, since="2023-01-01")
```

Returns up to 20 papers from 2023+ matching "RLHF small language models", sorted by arxiv relevance.

## Field-prefix queries

Supports arxiv search syntax:

| Prefix | Meaning | Example |
|---|---|---|
| `ti:` | title contains | `ti:attention` |
| `abs:` | abstract contains | `abs:reasoning` |
| `au:` | author name | `au:vaswani` |
| `cat:` | arxiv category | `cat:cs.CL` |

Combine with AND/OR: `ti:attention AND au:vaswani`.
