---
name: systematic-review
description: PRISMA-inspired systematic literature review workflow — autoloaded
autoload: true
---

# Systematic Review Workflow

每次接到 Stage 2 任务必走这 9 步。**不要跳步**。

## 1. 评估现状

```
corpus_status()
read('stage1_topic_refiner.md')   # 拿到精炼的研究问题 (RQ)
```

如果 corpus 里已有 ≥ 10 篇相关 paper（用 corpus_search 验证），可以直接进 step 6。

## 2. 拆解 query（multi-aspect）

将 RQ 拆成 **3-5 组互补 query**：
- 主题词 + 同义词
- 关键技术 + 缩写
- 应用领域限定

例：RQ = "RLHF degradation in small reasoning models"
- query 1: "RLHF small language models reasoning"
- query 2: "reward model overfitting distillation"
- query 3: "instruction tuning capability degradation"

## 3. 并行多源检索

```
parallel_multi_search(query, sources=["arxiv", "semantic_scholar", "openalex"])
```

每组 query 用一次。3-5 组 query × 30 results = 候选池约 100-150 篇（去重后通常 60-90 篇）。

如果 `S2_API_KEY` 未配置，去掉 `semantic_scholar` 避免限速。

## 4. 筛选（漏斗）

候选池 → 30 篇内。两轮：

**Round 1（标题 + abstract）**：
- ✓ 与 RQ 直接相关（不是擦边）
- ✓ 近 5 年（除非奠基性工作 — citation count > 100 例外）
- ✓ method 描述足够支持你的下游 stage 引用

**Round 2（如果还 > 30 篇，按 cite count 排序取 top 30）**

排除：
- ✗ 纯 position paper / opinion piece
- ✗ workshop 且 0 引用 + 发表 > 2 年（噪声）
- ✗ retracted papers

## 5. 取全文 + 入库

对筛后每篇：
```
md = pdf_extract(pdf_url)
corpus_add_paper({
    "id": arxiv_id_or_doi,
    "title": ..., "authors": [...], "year": ..., "venue": ...,
    "abstract": ..., "full_text_md": md["markdown"],
    "source_query": "<which query found it>",
})
```

PDF 解析失败的 paper：用 abstract 凑合（注明 `full_text_available: false`）。

## 6. 抽取 claim

```
for paper_id in corpus_list_papers():
    extract_claims(paper_id)
```

每篇 5-15 个结构化 claim。少于 5 → paper 相关性可能不够，考虑剔除。多于 15 → 你在抽噪声。

加载 `skills/claim-extraction/SKILL.md` 看抽取协议（`load_skill('claim-extraction')`）。

## 7. 找冲突 + gap

```
load_skill('conflict-detection')
```

按 SKILL 指引扫所有 claim 找：
- **conflicts** — 跨 paper 矛盾（同一 setup 不同结论）
- **gaps** — 各 paper 自承的 limitations / future work，且无后继研究

## 8. 输出双格式

**先**生成 `stage2.json`（严格 LiteratureSurveySchema）：

```json
{
  "research_question": "...",
  "search_strategy": {
    "queries": [...], "sources": [...], "time_window": "...",
    "inclusion_criteria": [...], "exclusion_criteria": [...]
  },
  "corpus_summary": {
    "total_papers": 27, "year_distribution": {...}, "venue_distribution": {...}
  },
  "taxonomy": [...],
  "methods_landscape": [...],
  "findings": [{"text": "...", "cites": ["arxiv:2401.XXXXX"]}, ...],
  "conflicts": [...],
  "open_questions": [...],
  "gaps": [...],
  "suggested_directions": [...]
}
```

**然后**用 schema 渲染人读 markdown 写到 `stage2_literature_surveyor.md`。

## 9. 自检（必须）

```
result = verify_citations(read('stage2_literature_surveyor.md'))
```

如果 `result.unverified_count > 0`：
1. 看哪些 cite 失败
2. 删掉这些 cite + 对应 claim，或者用 corpus_search 找替代
3. 重新 verify
4. 直到 unverified_count == 0

## 严禁

- 编造 arxiv_id 或 cite text
- 跳过 step 9 的 verify_citations
- 输出超过 30 篇 paper（不是越多越好）
- 报告里出现没 cite 的 claim
- 跟 critic 辩论 — critic reject = 改

## 范例

参考 `examples/`（如果存在）—— 标杆 survey 的长度、结构、引用密度。
