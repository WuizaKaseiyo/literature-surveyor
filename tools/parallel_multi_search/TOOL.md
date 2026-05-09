---
name: parallel_multi_search
description: Concurrently query arxiv + Semantic Scholar + OpenAlex. Dedupes by arxiv_id/DOI/title. Returns unified result list.
---

# parallel_multi_search

Single round-trip multi-source search with cross-source deduplication.

## When to use

- Default Stage 2 retrieval — gives broader coverage than any single source
- When you need both preprint freshness (arxiv) AND citation count (Semantic Scholar) AND cross-domain reach (OpenAlex)

## When NOT to use

- For a single specific paper lookup → use the source-specific tool directly
- If `S2_API_KEY` not set AND you want to avoid the 1.5s polite-sleep → omit `"semantic_scholar"` from sources

## Default behavior

```python
parallel_multi_search(query="RLHF small models")
# → queries ["arxiv", "openalex"] in parallel (SS off by default)
```

To include Semantic Scholar (recommended if `S2_API_KEY` is set):

```python
parallel_multi_search(
    query="RLHF small models",
    sources=["arxiv", "semantic_scholar", "openalex"],
    max_results_per_source=30,
)
```

## Output

```json
{
  "results": [
    {"arxiv_id": "...", "title": "...", "authors": [...], ...,
     "found_in_sources": ["arxiv", "openalex"]}    // ← provenance
  ],
  "total_unique": 47,
  "per_source_counts": {"arxiv": 30, "openalex": 25},
  "query": "...",
  "sources_queried": ["arxiv", "openalex"]
}
```

If a source errors, its entry appears in `"errors": {source: reason}` instead of crashing the whole call.

## Dedup logic

Two papers considered duplicates if they share ANY of:
- same `arxiv_id`
- same `doi`
- same normalized title (lowercased, punctuation stripped, whitespace collapsed)

Merged paper preserves all non-empty fields from both, and union of `found_in_sources`.

## Cost

Total wall-clock = max(per-source latency) — sources run concurrently. Typical: 2-4s for 3 sources @ 30 results each.
