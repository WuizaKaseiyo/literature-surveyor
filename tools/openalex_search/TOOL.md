---
name: openalex_search
description: Search OpenAlex (240M+ works, all fields, open metadata). Best for cross-domain or older papers.
---

# openalex_search

Query OpenAlex — the open-source academic graph (replaces Microsoft Academic Graph).

## When to use

- Cross-domain search (medicine, biology, social sciences, humanities — not just CS)
- Older papers (full historical coverage, not just preprints)
- Need venue + citation metadata
- Filter by open-access only

## When NOT to use

- Bleeding-edge ML preprints (last 2 weeks) → `arxiv_search`
- Influence-weighted ranking → `semantic_scholar_search`

## Configuration

Set `OPENALEX_MAILTO` env to your email — joins OpenAlex's "polite pool" for faster, more stable responses. No API key needed.

## Cost

- Free, no key
- Polite pool: 0.3s polite-sleep per call
- Otherwise: 1.0s polite-sleep per call

## Notable quirks

- OpenAlex stores abstracts as **inverted index** for IP reasons. This tool reconstructs full text automatically.
- ID format: `W123456789` (the "Work" prefix). Strip prefix to use elsewhere.

## Example

```python
openalex_search(
    query="systematic review methodology medical",
    max_results=20,
    since_year=2020,
    open_access_only=True,
)
```
