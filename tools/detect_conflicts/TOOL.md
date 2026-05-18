---
name: detect_conflicts
description: Cross-paper claim contradiction detector. Uses applies_to_dims as join key, LLM as judge.
---

# detect_conflicts

The "find papers that say opposite things" tool. Reads `claims.jsonl`, joins
candidate pairs via the structured `applies_to_dims` field (filled by
extract_claims v2), and asks an LLM judge whether they actually contradict.

## When to use

After `extract_claims` has populated the corpus. The output is meant to be
inlined into `stage2.json` as `Conflict[]`. Run in step 7 of the
systematic-review workflow.

## Filtering chain (cheap → expensive)

1. **Different paper_id** — same-paper "contradictions" aren't real cross-study disagreements
2. **Not both conjecture** — speculation × speculation is opinion, not contradicting facts
3. **Shared `applies_to_dims` value** — bidirectional substring match on at least one shared key
4. **Token Jaccard on claim_text ≥ `min_topic_jaccard`** (default 0.08) — prevents asking the LLM about totally unrelated claims that happen to share a setting key
5. **Top `max_pairs` by topic overlap** (default 50) — hard cost cap

In a typical 5-paper, 70-claim corpus the cheap filter cuts ~15K naive pairs
down to a few dozen real candidates.

## Levels

| Level | Meaning | Example |
|---|---|---|
| `direct` | Same setup + same metric, opposite numeric/qualitative | A: +5pp MMLU on Llama-7B; B: -3pp MMLU on Llama-7B |
| `methodological` | Same problem, different methods, different conclusions | A: PPO works for X; B: DPO doesn't |
| `scope` | Agree on overlap, differ at extent | A: works at 7B; B: doesn't at 70B |
| `temporal` | Later paper overturns earlier | 2022: works; 2024 replication: doesn't |

## Output

Each conflict matches `schemas.literature_survey_schema.Conflict`:

```python
{
  "id": "conflict-001",
  "level": "direct",
  "claim_a_id": "...", "claim_a_paper_id": "...", "claim_a_text": "...",
  "claim_b_id": "...", "claim_b_paper_id": "...", "claim_b_text": "...",
  "shared_setting": "model_size=7B-13B, dataset=MMLU",
  "description": "A reports +5pp; B reports -3pp on same setup",
  "confidence": 0.85,
  "topic_overlap": 0.42,
}
```

## Why `applies_to_dims` matters

Without structured scope dims, "do these two claims share a setting?" becomes
free-form LLM judgement on every pair → expensive and unreliable. The dim dict
turns it into a cheap key-overlap test, narrowing the funnel to LLM-worthy pairs.
This is exactly what v2's `applies_to_dims` was designed for; without F, the
field is decorative.

## Cost

50 LLM calls × ~500 input tokens each = 25K input tokens → ~$0.01 with DS-V3.
Set `use_llm=False` to get the pair candidates only (no verdicts), useful for
sanity-checking the filter behavior on a new corpus.
