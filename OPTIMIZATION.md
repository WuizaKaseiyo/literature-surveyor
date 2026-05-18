# Design Notes — Claim-centric Pipeline Overhaul

This document captures the architecture decisions behind the claim-centric
overhaul. It's a design retrospective for reviewers — not a changelog (see
git log for ordered changes) and not a TODO list (in-flight items live in
GitHub issues).

---

## Problems addressed

1. **Claim extraction was truncating.** The single-shot LLM call saw at most
   the first 30K chars of any paper; long papers (Sparks of AGI, 505K chars)
   either silently dropped half their content or failed parse-time entirely.
   `evidence_quote` was LLM-generated, never round-tripped against the source
   — paraphrased quotes were indistinguishable from verbatim ones to
   downstream consumers.

2. **Fact-check re-derived evidence per attribution.** `fact_check_rendered_survey`
   selected context via BM25 over raw paper text for every sentence it
   judged. Claims that had already been extracted (and that carried verbatim
   evidence quotes) were ignored, so the fact-checker repeated work and
   sometimes missed numeric contradictions that the extracted claims had
   already captured.

3. **Corpus storage was per-project; claims were write-only.** Every project
   re-fetched the same arXiv papers and re-ran extraction on them. Extracted
   claims went to `claims.jsonl` but no tool read them back; the conflict
   detector's algorithm was prose in a SKILL.md without a corresponding @tool.

---

## Architecture

### Claim record as a first-class entity

Three new fields on every claim:

- `evidence_quote_verified: bool` — substring/fuzzy-match the quote against
  the paper at extraction time. Downstream consumers can trust verified
  quotes; the rest are flagged in `warnings`.
- `applies_to_dims: dict[str, str]` — structured scope keys
  (`model_size`, `dataset`, `metric`, `regime`, `domain`, `language`).
  Replaces the free-form `applies_to` string for filter-friendly use, with
  the legacy string back-filled from the dict for backward compat.
- `section_path: list[str]` — heading lineage where the claim was extracted.
  Drives the section-distribution metric in `compare_extraction.py`.

`extract_claims version="v2"` is the default. v1 (single-shot, 30K truncated)
is retained for backward compat with a runtime deprecation warning.

### v2 extraction = section split + per-section LLM + quote verify

Markdown headings split the paper into sections (`_split_sections`,
H1–H3). Sections matching `references`/`acknowledgments`/`appendix` are
filtered out. The remaining sections are scored (claim-keyword headings +
length) and the top 8 are sent to the LLM individually with an 8K char cap.
Each returned claim's `evidence_quote` is substring-matched against the
full source (`_quote_in_source`), and the result stamped on the claim.

Per-section budget makes long papers tractable (Sparks now extracts 26
verified claims, was failing entirely in v1) and isolates parse failures
to a single section instead of the whole paper.

### Robust LLM JSON parsing

`_parse_claims_response` runs a tolerance pipeline:
1. Strip ` ```json ` fences.
2. Strict `json.loads` → list / `{"claims": [...]}` / single dict.
3. Salvage: walk the string with brace-depth + string-state tracking,
   extract balanced `{…}` blocks individually, parse each (string-aware so
   braces inside JSON strings don't bias the count).

`_call_llm_with_retry` retries once on empty response (transient API blip);
malformed JSON is not retried (salvage handles it).

`parse_mode` (`strict_array` / `salvaged_N` / `failed`) bubbles up to
talent-visible warnings so a rescued extraction is auditable.

### Claim store — five read-side tools

`tools/claim_store/`: `claim_search`, `claim_list_by_paper`, `claim_get`,
`claim_status`, `claim_find_evidence`. BM25 index over
`claim_text + evidence_quote`. Lazily rebuilt from `claims.jsonl` whenever
the source is newer than the index file. `extract_claims` doesn't need
to know — append-only writes are enough.

`claim_find_evidence` is the one fact_check uses: given a rendered
sentence + paper id, returns the most token-overlapping extracted claim's
verbatim `evidence_quote`, used as the LLM judge's anchor.

### Fact-check anchoring + survey_json fast path

`fact_check_rendered_survey` now resolves attributions via either:
- `markdown` — sentence-level parse of the rendered Stage 2 markdown
  (backward-compatible legacy entry), or
- `survey_json` — `LiteratureSurveySchema`-shaped dict, where each
  `Finding.claim_ids` is honored as a pre-declared anchor (skips
  token-overlap matching).

When an anchor is available (preferred or token-matched), the LLM judge
gets `evidence_quote + claim_text + source_section` as a strong-evidence
block, with the BM25-selected context appended for grounding. Without
an anchor the original BM25-only path runs unchanged.

Per-attribution rollup (`_aggregate_attributions`): a sentence with
multiple cites contributes one item per cite to `items`, plus one
rolled-up entry to `per_attribution`. `blocking_count` is reported at
the attribution level — if any cite supports, the attribution is
non-blocking even when sibling cites are unresolvable.

Low-confidence retry: when the LLM verdict is `unsupported` or
`partially_supported` with `confidence < 0.6`, the judge is rerun with
a 12K (vs 6K) BM25 window. The retry's verdict replaces the original
only if it's stricter (`_STRICTNESS_RANK` lower) — escalations are
taken, relaxations are not.

### Verdict ranking — two ranks, two purposes

Multi-cite rollup and E5 retry comparison have different semantic
requirements; collapsing both onto one rank introduced a latent bug
where `unsupported → contradicted` retries silently kept the weaker
verdict. Two explicit ranks:

- `_ROLLUP_RANK`: `supported > partial > contradicted > unsupported > …`.
  `max(rank)` wins. Among rejection-class verdicts, `contradicted` is
  more *actionable* than `unsupported` for the talent ("the paper
  actively disagrees" vs "the paper doesn't say"), so it surfaces in the
  rollup display.
- `_STRICTNESS_RANK`: `contradicted < unsupported < … < partial < supported`.
  Lower = stricter. Used only by E5 retry — a retry whose verdict is
  stricter replaces the original; equal or relaxed retries keep the
  original.

`tests/test_fact_check_rendered_survey.py::test_rollup_rank_aggregation_matrix`
and `test_strictness_rank_invariants` pin this contract so a future rank
reshuffle can't silently regress E5 escalation.

### Cross-project corpus (opt-in via `LITSURVEY_GLOBAL_CORPUS_DIR`)

When unset (default), tools behave identically to the pre-overhaul
single-tenant flow. When set, layered mode:

- Entities (`papers.jsonl`, `claims.jsonl`, BM25 indices) live in the
  global dir, shared across projects.
- Each project's working dir holds `refs.jsonl` (one row per
  `(paper_id, source_query)` edge) + `project_meta.json` (project_id =
  `sha256(abs_cwd)[:12]`).
- `corpus_search` / `corpus_list_papers` accept `scope=project|global|both`.
- `corpus_add_paper` becomes idempotent at the entity level — a paper
  already in the global pool is referenced, not re-fetched.
- `extract_claims` short-circuits when `claims.jsonl` already has claims
  for `paper_id`, status="reused", zero LLM calls. This is the main
  cost-saver: PDF download and claim extraction are the two most
  expensive pipeline steps, both skipped on cache hit.

Concurrency: an `fcntl.flock` guard protects the `(load → check → write
entity → load index → mutate → save)` sequence in the shared store.
Index writes use a tmp + atomic rename so readers never see a torn JSON.

A one-shot migration script (`tools/corpus_store/migrate_to_global.py`)
promotes a legacy single-tenant corpus into the global pool and builds
the project's `refs.jsonl`. Idempotent (re-running skips already-promoted
entities) and lock-aware (holds the same `.lock` as live agents).

### Findings carry backing claim IDs

`Finding.claim_ids: list[str]` (new schema field) lets the talent
declare which extracted claim(s) support each finding. The rendering
convention includes a verbatim evidence footnote per finding (parsed-and-
ignored by the fact_check cite regex via the `(arxiv:…#claim-N)` parenthesized
form), so human reviewers can verify a survey without opening any PDFs.

Step 7.5 of `systematic-review` requires a `claim_search` call before
each finding is written, and step 9 cross-checks that the
`matched_claim_id` returned by fact_check is a subset of the
finding's declared `claim_ids`.

### Conflict detection — applies_to_dims as the join key

`detect_conflicts` walks all pairs of (different-paper, not-both-conjecture)
claims, filters on shared `applies_to_dims` keys with overlapping value
tokens, then on `claim_text` token Jaccard. The remaining candidates are
sent to an LLM judge with both claims' `claim_text` AND `evidence_quote`
AND the shared setting. Verdict is one of `direct / methodological /
scope / temporal / none`. The temporal level uses a paper_id → year map
loaded from `papers.jsonl` (the previous `claim.get("paper", {}).get("year")`
path was dead code — claim records don't carry a nested paper dict).

Cost control: filter chain typically takes ~9.7K naive pairs (5 papers
× ~140 claims) down to ~11 candidates before any LLM call. The cap
(`MAX_CANDIDATE_PAIRS=200`) is applied *after* sorting by topic overlap
so high-scoring pairs from late in iteration aren't dropped.

---

## Intentionally deferred

These items came up in design / review and were deliberately not built:

- **Strict `claim_type: Literal[...]` validation in the Claim Pydantic.**
  Real DS-V3 output across 248 claims showed 100% adherence to the four
  canonical values. Adding the constraint is pure defense against an
  unobserved problem and would silently drop claims if a future model
  emits "factual" capitalized differently.

- **`search_history.jsonl` for query-level caching.** External search
  results decay quickly (new papers daily). A query-level cache would
  miss new arrivals; the freshness/staleness tradeoff is wrong. Paper-id
  level caching (which we do have) saves the expensive PDF/extract steps
  without freshness risk.

- **Cross-unit numeric equivalence (`30% ≡ 0.30`).** Context-dependent and
  lossy. The LLM judge handles this when needed; the heuristic stays
  unit-strict to avoid false positives on plain-number-vs-percent comparisons.

- **Difference matching for numbers ("improved by 13pp" ≡ "42% → 29%").**
  Complex parser, narrow win. LLM judge handles it; heuristic stays
  comparison-only.

- **In-process LRU cache for BM25 indices, SQLite migration.** Index
  files are small enough that JSON-load-per-call is acceptable below ~1K
  papers. The TOOL.md notes a chromadb/sqlite migration as future work.

- **`recent_projects` list in `corpus_status`.** Would require scanning
  all known project directories; complexity vs value didn't justify.

- **`corpus_promote_to_global(paper_id)` as a standalone @tool.** The
  migration script's batch version covers the same need; runtime promotion
  is rare enough not to need its own @tool surface.

- **Strict shared-read locks on corpus_search / claim_search.** The
  global write lock (`fcntl.LOCK_EX` on `.lock`) prevents reader-vs-writer
  torn writes for the *index* (atomic rename) but `papers.jsonl` /
  `claims.jsonl` append-only files can still produce a partial-last-line
  read during a concurrent append. Lines that fail to parse are silently
  skipped by `_load_papers` / `_load_claims`, so the worst observable
  effect is "the just-appended row isn't visible until the next read."
  Acceptable for the current single-agent dogfood case; shared-read locks
  go in when we hit multi-agent layered mode in anger.

---

## Evaluation harness (local-only, not in commits)

`tests/eval/compare_extraction.py` and `tests/eval/compare_factcheck.py`
produce side-by-side baseline reports under `tests/eval/reports/`.
`tests/e2e/run_release_gate.py` is the full LLM-in-loop release gate (29
asserted checks across 9 phases). All three are git-ignored — they're
dev tooling, not shipped code. Reports / run artifacts under
`smoke_runs/`, `release_gate_runs/`, `reports/` are gitignored too;
regenerated on demand from the same fixtures (`tests/fixtures/golden_papers/`,
`tests/fixtures/factcheck_fixture/`).
