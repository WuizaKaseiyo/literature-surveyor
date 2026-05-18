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

**Layered 模式提示**：如果 `corpus_status()` 返回 `"mode": "layered"`，先用
`corpus_search(query, scope="global")` 看 global 池（其他项目沉淀的）能不能
覆盖你的 RQ 一部分 —— `corpus_add_paper` 把它们引入到本项目时**不会重抓 PDF
也不会重抽 claim**（extract_claims 自动短路复用 global claims，省钱省时间）。
然后再决定还要补搜哪些。

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

## 7.5 编写 finding 前先用 claim 背书（强制）

**每条 finding 写进 stage2.json 之前**，必须先用 `claim_search` 找到至少 1 条
（建议 1-3 条）extracted claim 在背书它。否则就是"凭印象写"，下游 fact_check
会抓到。

```python
# 草拟 finding 文本（脑内）：
# "DPO matches or improves response quality vs PPO-based RLHF on summarization"

backing = claim_search(
    query="DPO PPO summarization response quality",
    paper_ids=["2305.18290"],   # 你打算挂的那篇
    top_k=3,
)
if not backing["results"]:
    # 没 claim 背书 → 不能写这条 finding
    # 选项 A：corpus 里这篇 paper 确实没说这个 → 删掉 finding 或换论据
    # 选项 B：claim 抽取漏了 → 跑 extract_claims(paper_id, version="v2")
    pass

claim_ids_backing = [r["claim"]["id"] for r in backing["results"][:3]]
# 后面写进 Finding.claim_ids
```

**判断标准**：只用 `claim_search.score` 看相关性还不够，**必须人读一眼 claim_text
确认它真的在说同一件事**。score 高但话题不同的也存在（特别是同一篇 paper 内）。

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
  "findings": [
    {
      "text": "DPO matches PPO-based RLHF in summarization quality.",
      "cites": [{"paper_id": "2305.18290", "cite_text": "Rafailov et al. 2023, arxiv:2305.18290"}],
      "claim_ids": ["2305.18290#claim-3", "2305.18290#claim-7"],
      "confidence": 0.85
    }
  ],
  "conflicts": [...],
  "open_questions": [...],
  "gaps": [...],
  "suggested_directions": [...]
}
```

`claim_ids` **不能为空** —— step 7.5 你已经查过，每条 finding 至少 1 条 claim 背书。

**然后**用 schema 渲染人读 markdown 写到 `stage2_literature_surveyor.md`。

### Markdown 渲染约定

每个 finding 一段话 + cite，**强烈建议**紧跟一个 evidence footnote 块（blockquote
格式），引用 backing claim 的 `evidence_quote` 原文。

```markdown
DPO matches or improves response quality versus PPO-based RLHF on
summarization and single-turn dialogue [Rafailov et al. 2023, arxiv:2305.18290].

> evidence: "matches or improves response quality in summarization and single-turn dialogue"
> — Section 6, Table 1 (arxiv:2305.18290#claim-3)
```

footnote 格式约束：
- 用 `> evidence:` 起头（fact_check 的 parser 会忽略 blockquote 里的 cite，避免重复打分）
- 用 claim 的 `evidence_quote` 原文（带引号），后面 `—` + `source_section` + `(paper_id#claim-N)`
- 一条 finding 配一条最强的 evidence；多 backing 时挑 confidence/相关性最高的那条

这一步让人读 review 时不用打开 PDF 就能验证。`claim_ids` 写在 JSON 里给下游
（fact_check / 后续 stage / 跨项目复用）；evidence quote 出现在 markdown 里给人读。

## 9. 自检（必须）

```
cite_result = verify_citations(read('stage2_literature_surveyor.md'))

# 推荐：用 stage2.json 走 fast path，跳过 markdown 解析的不确定性，且
# Finding.claim_ids 会被直接当 anchor（更确定，不靠 token-overlap 猜）
fact_result = fact_check_rendered_survey(
    survey_json=read_json('stage2.json'),
    use_llm=True,
)

# 退路：仅有 markdown（外部 review 场景）
# fact_result = fact_check_rendered_survey(
#     markdown=read('stage2_literature_surveyor.md'),
#     use_llm=True,
# )
```

如果 `cite_result.unverified_count > 0`：
1. 看哪些 cite 失败
2. 删掉这些 cite + 对应 claim，或者用 corpus_search 找替代
3. 重新 verify
4. 直到 unverified_count == 0

如果 `fact_result.blocking_count > 0`：
1. 对每条 `unsupported` / `contradicted` / `source_irrelevant` 找到对应句子
2. 删除、改写、或换成真正支持该句的 citation
3. `partially_supported` 必须降级措辞（如 "shows" → "suggests"）
4. 重新运行 `fact_check_rendered_survey`，直到 blocking_count == 0

**交叉校验 `matched_claim_id`**：每条 `fact_result.items[i].matched_claim_id`
（fact_check 真正用来判 verdict 的那条 claim）应该出现在对应 finding 的
`claim_ids` 里。如果不在：

- `matched_claim_id == ""` 但 finding 的 `claim_ids` 不空 → 你 step 7.5 挂的
  claim 跟 fact_check 实际找到的不一致，要么 claim 没真在背书要么挂错了
- `matched_claim_id` 不在 `claim_ids` 里 → fact_check 找到了**别的** claim 在背书；
  补进 finding 的 `claim_ids`
- 都对得上 → 数据闭环正确

## 严禁

- 编造 arxiv_id 或 cite text
- 跳过 step 9 的 verify_citations
- 跳过最终文本的 fact_check_rendered_survey
- 输出超过 30 篇 paper（不是越多越好）
- 报告里出现没 cite 的 claim
- 报告里出现 citation 真实但不支持该句的 claim
- 跟 critic 辩论 — critic reject = 改

## 范例

参考 `examples/`（如果存在）—— 标杆 survey 的长度、结构、引用密度。
