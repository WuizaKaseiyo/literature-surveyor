---
name: claim-extraction
description: How to extract structured claims from a paper
autoload: false
---

# Claim Extraction Protocol

每篇论文抽 **5-15 个 claim**。少了说明 paper 相关性弱（考虑剔除）；多了说明你在抽噪声。

## Claim Schema

每个 claim 必须包含：

| 字段 | 说明 | 例 |
|---|---|---|
| `claim_text` | 论文主张的事实 / 关系（一句话） | "RLHF reduces hallucination by 30% on TruthfulQA in 7B models" |
| `claim_type` | factual / methodological / negative_result / conjecture | factual |
| `evidence_span` | 原文支持位置（章节 + 表/图编号） | "Section 4.2, Table 3" |
| `evidence_quote` | 支持该 claim 的短原文摘录，用于最终 fact check | "The hallucination rate decreased..." |
| `source_section` | evidence 所在章节/标题 | "Section 4.2" |
| `confidence` | 论文自身宣称的统计置信度（0-1） | 0.85 |
| `applies_to` | 适用范围限定（模型规模 / 数据集 / 领域） | "models 7B-13B, English only" |

## Claim Types

- **factual** — 数值结果、性能数字、效应大小
- **methodological** — 算法/loss/训练流程改进
- **negative_result** — "X 不 work"，"Y 与预期相反"
- **conjecture** — 作者基于结果的推断（不是直接证据）

注意区分 factual 和 conjecture —— "我们的方法 work" 是 factual；"这暗示更大模型也会受益" 是 conjecture。

## 抽取流程

```python
paper = corpus_get_paper(paper_id)
text = paper["full_text_md"] or paper["abstract"]
# 调 LLM with structured output → list of Claim
# 检查每个 claim 是否带 evidence_span + evidence_quote
# 写回 corpus
extract_claims(paper_id)  # 工具内部完成上述
```

## 反例

❌ "This paper proposes an interesting method." — 不是 claim，是评价
❌ "RLHF improves models." — 太泛，没有 applies_to / numbers
❌ "Future work could explore X." — 这是 future_work，进 OpenQuestion 而不是 claim

✓ "PPO-based RLHF on Llama-7B reduces TruthfulQA hallucination from 42% to 29% (Table 2)." — 有数字、有范围、有 evidence_span

## 与 Conflict Detection 的衔接

每个 claim 的 `applies_to` 字段是 conflict detection 的关键 —— 两篇 paper 在**同一 applies_to**下给出**矛盾的 claim_text** = conflict。

如果你抽 claim 时不写 `applies_to`，conflict detection 会失效。

## 与 Finding Rendering 的衔接

每个 claim 的 `id` 后面会被 finding 的 `claim_ids` 引用 —— `claim_search()` 是
talent 在写 finding 前必跑的查询。`evidence_quote` 会作为 evidence footnote 出现
在最终 markdown 的 finding 下面，让人读 review 不用打开 PDF：

```markdown
DPO matches PPO on summarization [Rafailov et al. 2023, arxiv:2305.18290].

> evidence: "matches or improves response quality in summarization and single-turn dialogue"
> — Section 6, Table 1 (arxiv:2305.18290#claim-3)
```

所以 **`evidence_quote` 必须是原文 verbatim**（v2 抽取期已自动 substring 回校；
v1 凭 LLM 复述，回校率约 95%）。复述失真的 quote 在 footnote 里就是噪声。
