---
name: fact_check_rendered_survey
description: Final attribution fact-checker for rendered literature surveys. Checks whether each sentence-level claim is supported by its cited corpus paper.
---

# fact_check_rendered_survey

Use this after `verify_citations` and before final submission.

`verify_citations` answers: "Does this cited paper or ID exist?"

`fact_check_rendered_survey` answers: "Does the cited paper actually support the sentence that cites it?"

## What It Checks

The tool parses the final Markdown into sentence-level attribution pairs:

```text
claim sentence + citation -> cited paper full_text_md/abstract -> verdict
```

It uses deterministic parsing with backward attribution: when a citation appears at the end of a paragraph, the citation is applied to preceding uncited claim sentences in that same paragraph. This matches common LLM report style.

## Verdicts

| Verdict | Meaning |
|---|---|
| `supported` | Cited source supports the concrete claim |
| `partially_supported` | Source supports the topic or part of the claim, but the wording should be softened |
| `unsupported` | Source is related but does not support the specific claim |
| `contradicted` | Source contradicts the claim |
| `source_irrelevant` | Source does not address the same topic |
| `source_not_in_corpus` | Citation exists in text but is not in local corpus |
| `no_source_text` | Paper exists but has neither full text nor abstract |
| `judge_error` | Evaluator failed |

## Example

```python
audit = fact_check_rendered_survey(
    markdown=read("stage2_literature_surveyor.md"),
    use_llm=True,
    judge_model="openai/gpt-4o-mini",
)
```

## Gate Policy

- `contradicted_count > 0`: must fix before submission
- `unsupported_count > 0`: delete, rewrite, or replace citation
- `source_irrelevant_count > 0`: replace citation
- `source_not_in_corpus_count > 0`: add/fetch paper or remove citation
- `partial_count > 0`: soften wording, e.g. "shows" -> "suggests"

## Per-cite vs per-attribution (E3)

When a sentence has multiple citations (`claim X [a][b]`), the per-cite items
list contains one row per cite. The canonical gate, however, is the
**per-attribution** rollup: if any cite supports the attribution, the
attribution is non-blocking (the talent just added an extra reference).

- `blocking_count`: number of attributions whose rolled-up verdict is non-supportive (act on this)
- `blocking_cite_count`: legacy per-cite count (for audit; usually higher)
- `per_attribution`: rolled-up view with `sub_results` listing each cite

This means a sentence with [supported, source_not_in_corpus] → blocking=0,
blocking_cite=1. Talent can drop the unresolvable cite without rewriting the
sentence.

When no LLM client is available, the tool falls back to a conservative heuristic using source-text overlap and numeric consistency. This is less precise than an LLM judge but catches many number mismatches and misattributions.
