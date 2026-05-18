# Stage 2 Literature Survey — Alignment & Imitation Limits (FIXTURE)

## Research question

How effective are different post-training alignment recipes (SFT, RLHF, DPO,
imitation) at narrowing the capability gap between open and proprietary LLMs?

## Findings

### Supervised fine-tuning on small, curated datasets

Strongly curated supervised fine-tuning sets can produce instruction-following
quality competitive with much heavier RLHF pipelines.
Responses from a 65B model trained on only 1,000 examples are preferred or
considered equivalent to GPT-4 in 43% of human comparisons
[Zhou et al. 2023, arxiv:2305.11206].
LIMA also outperforms RLHF-trained DaVinci003 in the same evaluation
[Zhou et al. 2023, arxiv:2305.11206].

### Preference optimization beyond PPO

DPO matches or improves response quality versus PPO-based RLHF in summarization
and single-turn dialogue while removing the need for an explicit reward model
[Rafailov et al. 2023, arxiv:2305.18290].
DPO is also more stable and lighter-weight to train than RLHF
[Rafailov et al. 2023, arxiv:2305.18290].
The reported experiments cover models up to 6B parameters across sentiment
modulation, summarization, and single-turn dialogue tasks
[Rafailov et al. 2023, arxiv:2305.18290].

### Pretraining and scale

Llama 2 70B improves results on MMLU and BBH compared to Llama 1 65B
[Touvron et al. 2023, arxiv:2307.09288].
Llama 2 also adopts a longer context window than Llama 1
[Touvron et al. 2023, arxiv:2307.09288].

### The limits of imitating proprietary LLMs

When crowd raters evaluate imitation models trained on outputs from a stronger
model such as ChatGPT, their early impressions appear competitive with the
target system
[Gudibande et al. 2023, arxiv:2305.15717].
However, targeted automatic evaluations reveal that imitation closes little or
none of the underlying capability gap
[Gudibande et al. 2023, arxiv:2305.15717].

### General intelligence claims for GPT-4

The authors argue that the early GPT-4 they evaluated exhibits more general
intelligence than previous AI models across mathematics, coding, vision, and
medicine
[Bubeck et al. 2023, arxiv:2303.12712].

## Caveats and unsupported claims (intentional mistakes for fact-check evaluation)

The following paragraphs contain intentional errors — wrong numbers, scope
overreach, or fabrications. A well-behaved fact-checker should mark each as
`unsupported`, `partially_supported`, or `contradicted`.

LIMA is preferred to GPT-4 in 65% of human comparisons
[Zhou et al. 2023, arxiv:2305.11206].

DPO requires 4x less compute than PPO at every model scale
[Rafailov et al. 2023, arxiv:2305.18290].

DPO is shown to work equivalently on models with 70B or more parameters
[Rafailov et al. 2023, arxiv:2305.18290].

Pretraining Llama 2 emitted approximately 1000 tCO2eq
[Touvron et al. 2023, arxiv:2307.09288].

Llama 2 was pretrained on 5 trillion tokens
[Touvron et al. 2023, arxiv:2307.09288].

Imitation completely closes the gap with ChatGPT on factual reasoning when data
is scaled up
[Gudibande et al. 2023, arxiv:2305.15717].

The Sparks of AGI authors confirm that GPT-4 has emerged human-level
consciousness
[Bubeck et al. 2023, arxiv:2303.12712].
