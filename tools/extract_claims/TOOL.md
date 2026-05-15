---
name: extract_claims
description: Use a cheap LLM to extract 5-15 structured claims from a paper in corpus. Persists to claims.jsonl.
---

# extract_claims

Convert paper text → structured `Claim` records using LLM-based extraction.

## When to use

After `corpus_add_paper` for each paper that has full_text_md (or at minimum a substantial abstract).

## When NOT to use

- For papers with `char_count < 200` (extraction will be poor)
- For surveys / position papers (not enough quantitative claims to extract)

## Cost

Uses `openai/gpt-4o-mini` by default (cheap — about $0.001 per paper). Override via `model` argument.

## Output schema

Each claim:
```python
{
    "id": "<paper_id>#claim-N",
    "paper_id": "...",
    "claim_text": "PPO-RLHF reduces TruthfulQA hallucination from 42% to 29% in Llama-7B",
    "claim_type": "factual",       # factual | methodological | negative_result | conjecture
    "evidence_span": "Section 4.2, Table 3",
    "evidence_quote": "The hallucination rate decreased from 42% to 29%...",
    "source_section": "Section 4.2",
    "confidence": 0.85,            # paper's own confidence
    "applies_to": "models 7B-13B, English only",
    "extracted_at": 1700000000.0,
    "model": "openai/gpt-4o-mini",
}
```

Persisted to `<CORPUS_DIR>/claims.jsonl` (one per line).

## Why a separate file from corpus

Claims are queried separately from papers (e.g., conflict-detection scans claims across papers). JSONL append-only matches OMC's SSOT philosophy.

## LLM resolution order

1. OMC's `make_llm()` if running inside an OMC employee process
2. OpenRouter via `OPENROUTER_API_KEY` env
3. OpenAI direct via `OPENAI_API_KEY` env

If none work, returns `{"error": "LLM returned no parseable claims"}` — callable can decide to fall back or skip.
