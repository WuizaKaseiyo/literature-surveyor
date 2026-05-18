# End-to-end tests

Two layers:

## Offline (in pytest)

`tests/test_e2e_pipeline.py` — no LLM, no network. Uses the pre-populated
`tests/fixtures/factcheck_fixture/corpus/` and walks the full data plumbing:

```
corpus_status → claim_search → Finding(claim_ids=...) → LiteratureSurveySchema
   → render markdown w/ evidence footnote
   → verify_citations (corpus-only, no API)
   → fact_check_rendered_survey (use_llm=False)
   → assert matched_claim_id ⊆ Finding.claim_ids, blocking_count == 0
```

Runs in 0.2s, runs in CI. Catches plumbing regressions (schema/parser/anchor
flow) but does not catch LLM-side issues.

## LLM-in-loop smoke (manual)

`run_smoke_pipeline.py` — real LLM calls. Costs roughly $0.02-0.10 per run
depending on `--papers N`.

```bash
OPENAI_API_KEY=sk-... python tests/e2e/run_smoke_pipeline.py
OPENAI_API_KEY=sk-... python tests/e2e/run_smoke_pipeline.py --papers 5
OPENAI_API_KEY=sk-... python tests/e2e/run_smoke_pipeline.py --extract-version v1
```

Outputs (under `smoke_runs/<timestamp>/`):

- `stage2.json` — the schema-valid survey JSON
- `stage2_literature_surveyor.md` — the rendered markdown with evidence footnotes
- `corpus/papers.jsonl` + `corpus/claims.jsonl` — fresh corpus
- `citation_audit.json` — `verify_citations` result
- `factcheck_audit.json` — `fact_check_rendered_survey` result
- `summary.md` — top-line numbers + cross-check status

When to run:
- After any change to `extract_claims` (v1 or v2 path)
- After any change to `fact_check_rendered_survey`
- Before a release / before merging the big refactor PR
- Anytime you want a fresh `stage2_literature_surveyor.md` to eyeball

The `summary.md` ends with either `STATUS: ✅` or `STATUS: ⚠` — quick scan to
see if the pipeline still produces a clean survey end to end.
