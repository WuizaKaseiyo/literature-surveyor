# Examples

Standard reference surveys for few-shot prompting. The talent's `systematic-review` SKILL references these as quality benchmarks.

## TODO (Day 9 of plan)

Hand-write 2 high-quality reference surveys:

- `survey_rlhf_reasoning.md` + `.json` — exemplar with ~30 papers, 5 taxonomy nodes, 8 gaps
- `survey_long_context.md` + `.json` — second exemplar in a different sub-field

Each pair: the JSON is a `LiteratureSurveySchema` instance, the markdown is rendered from it. Both are loaded by the talent at runtime as few-shot examples.

Until these are filled in, the talent runs without few-shot — quality should still be acceptable thanks to SKILL.md instructions, but few-shot examples typically improve consistency by 20-30%.
