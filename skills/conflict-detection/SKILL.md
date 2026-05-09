---
name: conflict-detection
description: Cross-paper conflict and gap detection
autoload: false
---

# Conflict & Gap Detection

文献综述最有价值的输出是 **conflicts** 和 **gaps** —— 它们是 Stage 3 idea generation 的弹药。

## Conflict（跨 paper 矛盾）

定义：两个 claim 在**同一 setting 下**给出**互相反驳**的结论。

### 检测算法

```python
for claim_a in all_claims:
    for claim_b in all_claims:
        if claim_a.paper_id == claim_b.paper_id:
            continue
        # 同 setting 检查
        if not _same_setting(claim_a.applies_to, claim_b.applies_to):
            continue
        # 反驳检查（用 LLM 判，不能纯字符串）
        if _contradicts(claim_a.claim_text, claim_b.claim_text):
            yield Conflict(
                claim_a_id=claim_a.id,
                claim_b_id=claim_b.id,
                description="...",
                resolution_attempts=[],  # 后续 paper 是否解决了？
            )
```

### Conflict 的 4 个等级

| 等级 | 说明 | 例 |
|---|---|---|
| direct | 同 setup 同 metric 数字相反 | A: "+5pp", B: "-3pp" on MMLU 7B |
| methodological | 同问题不同方法得出不同结论 | A 用 PPO 得到 X, B 用 DPO 得到非 X |
| scope | 不同范围下不同结论（不是真冲突，但值得报告） | A: "在 7B 上 work", B: "在 70B 上不 work" |
| temporal | 后来的 paper 推翻早期 paper | 2022 年：work；2024 年复现：不 work |

direct 和 methodological 是 Stage 3 最值钱的；scope 次之；temporal 单独标注。

### 输出 Conflict Schema

```json
{
  "id": "conflict-001",
  "level": "direct" | "methodological" | "scope" | "temporal",
  "claim_a": {"paper_id": "...", "text": "..."},
  "claim_b": {"paper_id": "...", "text": "..."},
  "shared_setting": "...",
  "description": "...",
  "stage3_hint": "可让 Stage 3 验证哪种条件下哪边正确"
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
