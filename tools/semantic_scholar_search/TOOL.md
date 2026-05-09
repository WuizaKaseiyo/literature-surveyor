---
name: semantic_scholar_search
description: Search Semantic Scholar (citation-aware). Sorted by citation count — surfaces influential works first.
---

# semantic_scholar_search

Query the Semantic Scholar Graph API. Strongest signal: **citation count** — results are sorted descending so influential papers surface first.

## When to use

- Find foundational / influential works in a sub-field
- Cross-domain (covers ML, biology, medicine, physics, social sciences)
- When you need DOI / venue info (arxiv only has preprints)
- When you want both arxiv preprint version AND published version

## When NOT to use

- Need preprint freshness (papers ≤ 2 weeks old) → use `arxiv_search`
- Need >100 results in one query → use `openalex_search`

## Configuration

Set `S2_API_KEY` environment variable for higher rate limits. Without it:
- ~100 requests / 5 minutes (very throttling)
- Tool will print warning on first call

Get a free key: https://www.semanticscholar.org/product/api

## Cost

- Free with API key
- Tool inserts 0.5s (with key) / 1.5s (without) polite-sleep per call
- Auto-retries with exponential backoff on 429/5xx

## Adapted from

Originally `ai_scientist/tools/semantic_scholar.py` in
[SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) (Apache-2.0).
Repackaged as LangChain `@tool` and normalized output schema.
