---
name: conflict-detection
description: Cross-paper conflict and gap detection
autoload: false
---

# Conflict & Gap Detection

文献综述最有价值的输出是 **conflicts** 和 **gaps** —— 它们是 Stage 3 idea generation 的弹药。

## Conflict（跨 paper 矛盾）

定义：两个 claim 在**同一 setting 下**给出**互相反驳**的结论。

### 用 `detect_conflicts` 工具，不要手写

```python
result = detect_conflicts(
    paper_ids=None,            # 默认扫所有；可限定到 Stage 2 本批 paper
    use_llm=True,               # 关闭则只返回 candidate pair（dry-run）
    model="openai/gpt-4o-mini",
    level_filter=None,          # 可选：只要 "direct" / "methodological" 等
    min_topic_jaccard=0.08,     # claim_text token Jaccard 下限
    max_pairs=50,               # LLM judge 调用次数硬上限
)
# result["conflicts"]: list[Conflict dict]
# result["candidates_screened"]: 经 filter 的候选 pair 数
# result["candidates_judged"]: 实际过 LLM 的对数
```

工具内部 filter chain：
1. 不同 paper_id
2. 不是双 conjecture（speculation × speculation 不算矛盾）
3. `applies_to_dims` 至少有一个 key 双方都填且值有重叠（双向 substring）
4. claim_text token Jaccard ≥ `min_topic_jaccard`
5. 按 topic overlap 排序，前 `max_pairs` 进 LLM judge

**前置依赖**：`extract_claims version="v2"` 必须填 `applies_to_dims`。v1 的
free-form `applies_to` 字符串不能用，filter 会全部跳过。

### Conflict 的 4 个等级

| 等级 | 说明 | 例 |
|---|---|---|
| direct | 同 setup 同 metric 数字相反 | A: "+5pp", B: "-3pp" on MMLU 7B |
| methodological | 同问题不同方法得出不同结论 | A 用 PPO 得到 X, B 用 DPO 得到非 X |
| scope | 不同范围下不同结论（不是真冲突，但值得报告） | A: "在 7B 上 work", B: "在 70B 上不 work" |
| temporal | 后来的 paper 推翻早期 paper | 2022 年：work；2024 年复现：不 work |

direct 和 methodological 是 Stage 3 最值钱的；scope 次之；temporal 单独标注。

### 输出 Conflict Schema

`detect_conflicts` 返回的字典直接匹配 `schemas.literature_survey_schema.Conflict`，
可以直接灌进 `stage2.json` 的 `conflicts: []`：

```json
{
  "id": "conflict-001",
  "level": "direct",
  "claim_a_id": "2305.18290#claim-7",
  "claim_a_paper_id": "2305.18290",
  "claim_a_text": "...",
  "claim_b_id": "2401.99999#claim-3",
  "claim_b_paper_id": "2401.99999",
  "claim_b_text": "...",
  "shared_setting": "model_size=7B, method=PPO, dataset=MMLU",
  "description": "claim_a reports +5pp; claim_b reports -3pp on same setup",
  "confidence": 0.85,
  "topic_overlap": 0.42
}
```

## Gap（未解决的开放问题）

定义：在 corpus 里**多次出现**但**没有 paper 解决**的问题。

### 来源（按可信度排序）

1. **作者自承的 limitations** —— 多篇 paper 在 Limitations 章节提同一个问题
2. **Future Work 段提及但没人后续做** —— 检查后续 paper 是否引用了
3. **Conflict 揭示的 gap** —— 上面 detected conflicts 暗示需要决定性实验
4. **覆盖空白** —— taxonomy 某个分支只有 0-1 篇 paper

### Gap Schema

```json
{
  "id": "gap-001",
  "title": "...",
  "description": "...",
  "evidence": [
    {"paper_id": "...", "quote": "Authors note this is unclear ..."},
    ...
  ],
  "actionability": "high" | "medium" | "low",  // Stage 3 能否变成 idea
  "estimated_difficulty": "low" | "medium" | "high"
}
```

`actionability: high` 的 gap 应优先 — 它们是 Stage 3 最容易转成 idea 的。

## Open Question vs Gap

- **Open Question** = paper 自承的待答问题（直接来自 limitations / future work）
- **Gap** = 你（综述者）总结的、跨 paper 的**模式**

每个 gap 通常关联多个 open_questions。

## 输出建议

- 5-10 个 conflicts（目标 conflicts.length / corpus.size ≈ 0.2）
- 8-15 个 open_questions
- 4-8 个 gaps（actionability=high 至少 2 个）

少于这个量 → 你抽 claim 不够细 / setting 标注不够准；多了 → 你在重复或编造。
