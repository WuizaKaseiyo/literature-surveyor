---
name: verify_citations
description: Scan markdown for citations and verify each is real (corpus + arxiv/Crossref/Semantic Scholar live check). Final defense against hallucinated cites.
---

# verify_citations

Final pre-submission defense against hallucinated citations.

## When to use

**Always call this immediately before considering Stage 2 output complete.**

If `unverified_count > 0`, you must fix (replace with real cite from corpus, or remove the cite + claim) and re-run until all are verified.

## Recognized cite formats

```
[Author Year, arxiv:2401.12345]
[Author Year, doi:10.1109/TPAMI.2023.1234567]
[Author Year, S2:abcd1234abcd1234abcd]
```

Other formats (numeric `[1]`, plain `[Smith 2024]`, APA `(Smith, 2024)`) are **ignored** — they can't be machine-verified, so don't use them.

## Verification logic

Per unique (kind, id) pair (deduped):

1. Check local corpus first (instant)
2. If miss, hit live API:
   - `arxiv:` → arxiv export API
   - `doi:` → Crossref API (free)
   - `S2:` → Semantic Scholar paper endpoint

## Output

```json
{
  "verified": [
    {"raw": "[Smith et al. 2024, arxiv:2401.12345]",
     "kind": "arxiv", "id": "2401.12345",
     "author_year": "Smith et al. 2024",
     "title": "...", "matched_corpus": true}
  ],
  "unverified": [
    {"raw": "[Wang et al. 2023, arxiv:2308.99999]",
     "kind": "arxiv", "id": "2308.99999",
     "author_year": "Wang et al. 2023",
     "reason": "arxiv ID 2308.99999 not found",
     "context_around": "...80 chars before/after the cite..."}
  ],
  "verified_count": 23,
  "unverified_count": 1,
  "total_cites": 24,
  "verification_method": "local + live API"
}
```

## Edge cases

- **Same cite repeated multiple times** in markdown → verified once (deduped by (kind, id))
- **Network failure** → verification reported as `"reason": "request failed: ..."` — treat as unverified for safety
- **arxiv versioning** (`2401.12345v2`) — use the bare ID without `v*` suffix for best results
- **Unicode in DOI** — automatically URL-encoded
