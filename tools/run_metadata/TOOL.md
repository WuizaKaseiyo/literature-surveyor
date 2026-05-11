---
name: run_metadata
description: Per-run provenance tracker. Hashes inputs/prompts, records model identifiers, accumulates per-stage wall-clock timings. Final artifact at <workspace>/run.json.
---

# run_metadata

Three coordinated tools (`run_start`, `run_stage_done`, `run_finalize`) that build a `run.json` file alongside the talent's outputs. The file records *how* this run was produced, so two runs of the same talent can be diffed mechanically.

## Why

A talent run produces `stage2.json` + `stage2_literature_surveyor.md`, but no record of:

- which prompt version drove the LLM (talent_persona.md changes silently between commits)
- which model was called (Claude 4.5 vs an earlier version vs a different provider)
- where time was spent (search vs claim extraction vs verification)
- what the exact input hash was (so you can confirm "two runs got different answers from the same question")

With `run_metadata` enabled, every run drops a small JSON next to its outputs containing this provenance. CEO / downstream tools can `diff run_a.json run_b.json` to localize behavioural changes.

## When to call

| Call | When | Cost |
|---|---|---|
| `run_start` | Once, at the very beginning of the 9-step workflow (Step 0) | Cheap: hashes a handful of `.md` files |
| `run_stage_done` | Once per stage that takes >5s wall-clock | Cheap: append + rewrite ~1KB JSON |
| `run_finalize` | Once, AFTER `verify_citations` passes and outputs are written | Cheap: snapshot + stamp |

## Workspace resolution

`run.json` lands at `<workspace>/run.json` where workspace is resolved in this order:

1. `$LITSURVEY_RUN_DIR` — explicit override
2. parent of `$LITSURVEY_CORPUS_DIR` — workspace where corpus also lives
3. CWD if it contains a `corpus/` subdir
4. `~/.litsurvey_corpus/` as last fallback

## `run.json` schema

```json
{
  "schema_version": "1",
  "run_id": "ab12cd34...",
  "started_at": "2026-05-11T08:00:00+00:00",
  "completed_at": "2026-05-11T08:14:32+00:00",
  "research_question_sha256": "d890bd59...",
  "research_question_preview": "RLHF on small language models (≤7B)...",
  "prompt_hashes": {
    "prompts/talent_persona": "008c6438...",
    "skills/systematic-review": "a7a9e6b0...",
    "skills/claim-extraction": "f9c8a2d1...",
    "skills/conflict-detection": "...",
    "skills/citation-verification": "..."
  },
  "models": {
    "main": "claude-sonnet-4-5",
    "extract": "gpt-4o-mini"
  },
  "stages": [
    {"name": "search",         "elapsed_s": 45.2, "recorded_at": "..."},
    {"name": "filter",         "elapsed_s": 8.1,  "recorded_at": "..."},
    {"name": "pdf_extract",    "elapsed_s": 22.8, "recorded_at": "..."},
    {"name": "claim_extract",  "elapsed_s": 312.5,"recorded_at": "..."},
    {"name": "conflict_detect","elapsed_s": 28.1, "recorded_at": "..."},
    {"name": "verify",         "elapsed_s": 12.0, "recorded_at": "..."},
    {"name": "render",         "elapsed_s": 4.2,  "recorded_at": "..."}
  ],
  "corpus_size_final": 28,
  "output_paths": ["stage2.json", "stage2_literature_surveyor.md"],
  "notes": []
}
```

## Example call sequence

```python
# Step 0 — start of workflow
run_start(
    research_question="RLHF on small language models, methods and bottlenecks",
    main_model="claude-sonnet-4-5",
    extract_model="gpt-4o-mini",
)
# → {"run_id": "...", "run_json_path": "/cwd/run.json", "prompt_hash_count": 5}

# Step 3 — after parallel_multi_search
run_stage_done(stage="search", elapsed_s=45.2, notes="3 queries × 3 sources")

# Step 6 — after extract_claims on the whole corpus
run_stage_done(stage="claim_extract", elapsed_s=312.5)

# Step 9 — after verify_citations passes
run_finalize(output_paths=["stage2.json", "stage2_literature_surveyor.md"])
# → {"run_id": "...", "completed_at": "...", "total_elapsed_s": 432.9, ...}
```

## Diff workflow

Two runs of the same talent on the same research question:

```bash
$ diff <(jq -S . run_a.json) <(jq -S . run_b.json)
```

Differences fall into a few classes:

- `prompt_hashes` differ → talent's behavioural prompts changed between runs
- `models` differ → different LLM picked up
- `research_question_sha256` differs → input changed
- `corpus_size_final` differs → retrieval converged on different paper sets
- `stages` differ in elapsed → performance regression / improvement

Everything else identical = differences in output are pure LLM nondeterminism (sample with `--temperature 0` to remove this dimension).

## What this tool does NOT do

- Does not measure LLM token usage (the @tool functions don't see token counts; need LangChain callbacks for that)
- Does not estimate USD cost (defer to v2, requires token counts)
- Does not hash the corpus content (only its final size)
- Does not block on validation failure (best-effort logging — talent must still call `verify_citations` for hard hallucination check)
