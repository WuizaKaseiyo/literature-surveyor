# Fact-check fixture

Fixture for `tests/eval/compare_factcheck.py`. Compares
`fact_check_rendered_survey` with `use_extracted_claims=True` (PR-3 path) vs
`False` (BM25-only baseline) on the same rendered markdown.

## Files

- `stage2_rendered.md` — a hand-crafted Stage 2 survey with 18 attributions:
  8 supportable, 4 partial-quality, 6 deliberately broken (wrong numbers /
  scope overreach / fabricated content)
- `corpus/papers.jsonl` — the 5 fixture papers (same as
  `tests/fixtures/golden_papers/papers.jsonl`)
- `corpus/claims.jsonl` — 70 claims extracted from those 5 papers by the v1
  baseline run (DeepSeek-V3, sample dir
  `tests/eval/reports/extraction_20260517_191650_samples/`)

## Expected verdict ground truth (manual ground truth)

The "intentional mistakes" section of the markdown is the negative-test set. A
correct fact-checker should mark each as `unsupported` or `contradicted`:

| Citation | Claim | Expected | Why |
|---|---|---|---|
| 2305.11206 | "preferred in 65% of comparisons" | unsupported | Paper says 43% |
| 2305.18290 | "DPO requires 4x less compute at every scale" | unsupported | Not in paper |
| 2305.18290 | "DPO works equivalently at 70B+ parameters" | unsupported | Paper studies up to 6B |
| 2307.09288 | "Llama 2 emitted ~1000 tCO2eq" | unsupported | Paper reports 539 |
| 2307.09288 | "Llama 2 pretrained on 5T tokens" | unsupported | Actually 2T |
| 2305.15717 | "imitation closes the gap when scaled" | contradicted | Paper concludes the opposite |
| 2303.12712 | "GPT-4 has human-level consciousness" | unsupported | Paper makes no such claim |

The remaining ~8 supported + 4 partial attributions should land as `supported`
or `partially_supported`.

## Regenerating claims.jsonl

```bash
python -c "
import json
from pathlib import Path
out = []
for s in Path('tests/eval/reports/extraction_20260517_191650_samples').glob('*__v1.json'):
    d = json.load(open(s))
    out.extend(d['claims'])
with open('tests/fixtures/factcheck_fixture/corpus/claims.jsonl', 'w') as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
"
```

Don't regenerate the markdown — that's the fixed input. Regenerate claims only
if extract_claims behavior changes.
