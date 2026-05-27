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
    extract_claims(paper_id)   # version="v3" 是默认
```

`extract_claims` v3 默认每篇返回 **8 条高信号 claim**（meta_pass → 段内 grounded
抽取 → 全局 rerank → 确定性 hard filter）。返回的每条 claim 都满足：

- `evidence_quote_verified=True`（quote 是源文连续片段）
- `applies_to_dims` 至少含 1 个非空键（model_size / dataset / domain / language / regime / metric）
- `saliency_type ∈ {empirical_finding, evaluation_result, method_proposed, limitation}`
  —— noise saliency 类（background / methodology_footnote / self_promotion /
  cited_other_work）已被 hard_filter 丢掉，不会进库
- `contribution_idx`：≥0 表示 ground 到 meta_pass 列出的某条 contribution；-1
  表示是 limitation 或 methodology 局部内容

**如何判断 claim 是否够用**：

- 返回 < 5 条 → 这篇 paper 可能信号弱（短 paper / 偏综述 / 主题与 RQ 偏远），考虑剔除
- 返回 == 8 条 → 正常，rerank 选了 top-K
- 返回 > 8 条 → 通常因为 force=True 重抽且原来已有更多。正常 e2e 不会出现

百科全书式长 paper（70+ 页，如 Llama 2 技术报告）的 8 条上限可能偏紧 —— 这时可
考虑显式跑 `extract_claims(paper_id, force=True, version="v3")` 加上人工二次筛选，
或者在 finding 阶段补一次 `claim_search` 拉同篇其它角度的 claim。

加载 `skills/claim-extraction/SKILL.md` 看抽取协议的设计细节（`load_skill('claim-extraction')`）。

## 7. 找冲突 + gap

```
load_skill('conflict-detection')
detect_conflicts(use_llm=True)
```

按 SKILL 指引扫所有 claim 找：

- **conflicts** — 跨 paper 矛盾。v3 抽取后 `applies_to_dims` 接近 100% 填充，
  detect_conflicts 用它作为 "shared setting" 的 join key 找候选对。LLM judge 在 4
  个 level 上判：`direct` / `methodological` / `scope` / `temporal`。
- **gaps** — 各 paper 自承的 limitations / future work，且无后继研究。优先看
  `saliency_type=="limitation"` 的 claim。

**0 conflict 是合法结果**：method-additive 的语料（如 RAG 这种"加方法"领域）
往往不会产生直接对线。这种情况在 markdown 里**显式说明**而不是省略
Conflicts 节（见 §8.2.7）。

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

## 7.7 综合规则（finding / taxonomy / methods 写之前都读这一节）

### 7.7.1 Cross-paper finding 至少占 1/3

Survey ≠ "每篇 paper 各列一条 highlight"。**5-8 条 finding 中，至少 1/3 必须是
cross-paper synthesis**（同一条 finding 引用 2-3 篇 paper，揭示 converging
evidence / shared limitation / contested result / methodological pattern）。

**好 cross-paper finding 模板**：

> "Multiple benchmark studies report that reference-based metrics correlate
> poorly with human judgments for long-form RAG output (2407.13998, 2410.23000,
> 2506.20051)." ← converging evidence 模式

> "Perplexity-based and paraphrase-based defenses both fail against
> interpretable corpus poisoning, while gradient-based attacks are reliably
> mitigated (2512.24268, 2502.00306)." ← shared limitation 模式

**好 single-paper finding**（节制使用，仅当 corpus 里没有 peer 可比时）：

> "Collab-RAG outperforms black-box and white-box retrieval baselines by 14.2%
> and 6.6% respectively on complex QA tasks (2504.04915)." ← 独特 headline 结果

### 7.7.2 SCOPE DISCIPLINE（防 scope inflation）

Finding text **不得**含未在 cited paper extracted claim 中出现的实体、数字、
方法名、qualifier、claim 成分。具体反例：

- ❌ 写 "especially for X and Y" 但 claim 只支持 X
- ❌ 写 "outperforms baseline B" 但 claim 没说
- ❌ 组合两篇 paper 的数字到一句里（"X reaches 0.88 and Y achieves 85%"），
  除非两个数字都在你引用的 claim 里
- ❌ 把"63.5% ComponentLevel"拆成两条 finding 分别讲 32.5% 和 31% —— 除非
  paper 原文确实把 63.5% 拆出 32.5+31 这两个数字独立列出
- ⚠️ 源 claim 用 hedging ("may", "suggests", "observed in some cases")，finding
  不要升级成 "shows" / "demonstrates"

如果 claim 只覆盖你 draft finding 的一部分 → **窄化 finding 匹配实际证据**。
紧的 finding 永远胜过宽的。

### 7.7.3 Taxonomy / Methods landscape / Gaps 覆盖率要求

**Coverage rule**：corpus 里每篇 paper 必须出现在 `{taxonomy, methods_landscape, gaps}`
至少一个 section。漏掉的 paper 是 coverage failure，audit 时会显示出来。

- **Taxonomy buckets**：3-6 buckets，每个 bucket 3-6 篇 paper。bucket 之间允许
  论文重叠（一篇可在多个 bucket 出现），但要倾向于每篇有一个 primary bucket。
- **Methods landscape**：3-5 个 method，每个 method 的 `representative_paper_id`
  **跨 method 必须 unique**（不能两个 method 用同一 rep paper）。每个 method 的
  `paper_ids` 列 2-4 篇。
- **Gaps**：3-5 个 gap，每个 gap 的 `evidence_paper_ids` 列 1-4 篇。gap 必须是
  真的研究空白，不是 limitation 的复述。
- **Open questions**：**≥3 条，最多 6 条**。如果你只想到 1-2 条 → 从你写的 gaps
  里派生（每条 gap 都隐含 1 个 open question）。每条锚 1-2 篇 `raised_by_paper_ids`。
- **Suggested directions**：3-5 条**具体研究路径**，不只是 gap 的复述。

## 8. 输出双格式

### 8.1 stage2.json（严格 LiteratureSurveySchema）

```json
{
  "research_question": "...",
  "search_strategy": {
    "queries": [...], "sources": [...], "time_window": "...",
    "inclusion_criteria": [...], "exclusion_criteria": [...]
  },
  "corpus_summary": {
    "total_papers": 19, "year_distribution": {...}, "venue_distribution": {...}
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

### 8.2 Markdown 必备结构（按顺序）

`stage2_literature_surveyor.md` 必须**按下面这个顺序**包含 12 节。每节标题用 H2
（`## ...`），缺一节都不算合格。

```
# Stage 2: Literature Survey — <one-line research question>

*Generated YYYY-MM-DD · N papers · M claims · model `<llm>` · run `<short_id>`*

## Abstract
## Research question
## Search strategy
## Corpus summary
## Findings              ← 按主题分组成 H3 子节
## Key numerical results ← 表格，无数字时省略
## Taxonomy
## Methods landscape
## Conflicts             ← 即使 0 conflict 也写（见 8.2.7）
## Open questions
## Gaps
## Suggested directions
## Audit summary
## References
## About this survey
```

下面逐节细则。

#### 8.2.1 Header + metadata strip

H1 标题用 RQ 的第一行（去掉句末标点）。**紧跟一行 metadata strip**，斜体格式，
含 4-5 个字段。

```markdown
# Stage 2: Literature Survey — What are the dominant failure modes of RAG in long-form QA

*Generated 2026-05-27 · 19 papers · 134 claims · model `deepseek-chat` · run `34b263d6`*
```

#### 8.2.2 Abstract（4-6 句 prose）

放在 `## Research question` 之前。**必须覆盖所有主要 finding theme**——如果
findings 里既讲 problem 也讲 mitigation，abstract 不能只讲一面。

模板：
1. 1 句话重述 RQ
2. 每个主要主题 1 句话 lead sentence（取该主题最强 finding 的核心结论）
3. 收尾："The survey synthesizes N papers spanning YYYY–YYYY."

写 prose 不要列表。avoid 机械连接词（"Across the corpus / Beyond that /
Further / Finally"）—— 用变化的学术过渡。

#### 8.2.3 Research question / Search strategy / Corpus summary

把 stage1 给的 RQ 原样贴出。Search strategy 列 queries + sources + time window。
Corpus summary 至少 `total papers` + `with full text`，建议附 `year distribution`
（如果 corpus_status 提供）。

#### 8.2.4 Findings —— 按主题分 H3 子节

不要 7 条 finding 平铺。用 taxonomy bucket 把 finding 分组成 H3 子节，每个 H3
下放属于该主题的 finding。

**主题分组规则**：每条 finding 归到 taxonomy bucket 里 `paper_ids` 与 finding
`cites[].paper_id` 重叠最多的那个。**平局时**：
1. 用 finding text 的 lexical signal —— 含 `defense`/`repair`/`reranker`/
   `refinement`/`mitigation` → 倾向 Mitigation Strategies / Adversarial Defenses 那类
   bucket；含 `failure`/`hallucination`/`error` → 倾向 Failure 那类；含
   `benchmark`/`evaluator`/`metric` → 倾向 Evaluation 那类
2. 若仍平局，倾向**当前已分配 finding 数较少的 bucket**（避免堆积到第一个 bucket）

**每条 finding 的写法**：

```markdown
<Finding sentence>. [Smith et al. 2024, arxiv:NNNN.NNNNN; Doe et al. 2025, arxiv:MMMM.MMMMM]

> evidence: "<verbatim quote from the most-supporting claim>"
> — <section ref> (<paper_id>#<claim-N>)
```

约束：

- finding 句末必须有 `[author year, arxiv:id]` 形式 cite。cross-paper finding 用
  `;` 分隔多个 cite：`[A et al. 2024, arxiv:1; B et al. 2024, arxiv:2]`
- 紧跟 1 行 blockquote evidence —— 用 `> evidence:` 起头（fact_check parser 会
  跳过 blockquote 里的 cite，避免重复打分）
- evidence quote 必须是源文 verbatim 子串；从 finding 的 `claim_ids` 里**挑能
  支撑 finding 主要断言的那一条**（不要无脑取 claim_ids[0]）。具体规则按优先级：
  - **(a) 数字必须对齐**：如果 finding 含具体数字（`+8.3 F1`、`14.7%`、`0.54`），
    evidence quote 必须含这些数字中至少一个的精确出现。打分 =
    `shared_numbers × 10 + token_jaccard`，**有数字命中的 claim 永远赢过无数字命中的**
  - **(b) 主断言 vs hedging 分清**：finding 形如 "X achieves substantial
    improvements ... **but** performance is sensitive to ..." 时，前半（主断言）
    优先，evidence quote 应支撑 "substantial improvements" 那部分，**不要选只
    支撑 "but" 后半 hedging/limitation 的 quote**。除非整个 finding 主断言就是
    一条 limitation
  - **(c) 多数字 finding**：finding 出现 N 个数字（如 "+14.7% recall, +12.5%
    groundedness"），优先选能 cover 最多数字的 claim 的 evidence quote。无法
    一条 cover 全部时挑数字最多的那条
- evidence quote 长度 ≤ 280 字符，**截断必须按词边界**（不能在单词中间砍，结尾加 `…`）
- evidence_quote 如果是 markdown 表格行（`X|Y|Z`），用 `(numeric values from a
  table — see §<section> of the source)` 替代原文

#### 8.2.5 Key numerical results 表（findings 含数字时）

`## Findings` 之后立刻一张紧凑表，每条 finding 一行（如果该 finding 含数字）：

```markdown
## Key numerical results

| Subject | Result | Paper |
|---|---|---|
| F1-score on HotpotQA | +8.3, +25.8% | 2510.22344 |
| page recall on ASQA | 14.7%, 12.5% | 2410.08623 |
| attack success rate (ASR) on FiQA | 1% | 2512.24268 |
```

- **Subject** 列：从 backing claim 的 `applies_to_dims` 拼 "metric on dataset"
  （二者都有时）；若都没，取 finding text 数字前的 60 字符上下文
- **Result** 列：finding text + evidence_quote 里所有 result-shaped 数字，按出现
  顺序逗号分隔，最多 4 个。"result-shaped" = 含小数点或百分号或带 `+`/`-` 号；
  排除 GPT-4 里的 `-4`、章节号 `5.1` 这类裸数
- **Paper** 列：finding 的第一个 cite paper_id

#### 8.2.6 Taxonomy / Methods landscape

按 §7.7.3 的 coverage rule 写。每个 H3 bucket 标题 + 描述 + `Papers: pid1, pid2, ...`。

Methods 用：

```markdown
### <Method name>

<2-3 句 description>

Pros: <pro 1>; <pro 2>
Cons: <con 1>; <con 2>
Representative: <paper_id>
```

**所有 method 的 representative_paper_id 跨 method 必须 distinct**。

#### 8.2.7 Conflicts（**即使 0 也写**）

非空时按 conflict level 列出。

**0 conflict 必须显式说明**（不要省略整节）：

```markdown
## Conflicts

No direct contradictions detected. Pairwise comparison screened <N> candidate
claim pairs (shared scope + topical overlap ≥ 0.08); <M> were LLM-judged and
all classified `not_contradicted`. Treat absence as method-additive corpus
behavior, not as proof of consensus.
```

`N` 和 `M` 从 `detect_conflicts` 返回的 `candidates_screened` 和 `candidates_judged`。

#### 8.2.8 Open questions / Gaps / Suggested directions

按 §7.7.3 数量要求和质量要求。Open question 每条尾部 `(pid1, pid2)`。Gaps 每个
带 `_(actionability: high|medium|low, difficulty: low|medium|high)_` 标签。

#### 8.2.9 Audit summary（在 References 之前）

**fact_check 完成后**写。让读者看到 attribution 验证结果——不能给读者"全部 supported"
的假印象。

```markdown
## Audit summary

- **Attribution audit**: 8/10 fully supported, 1/10 partially supported,
  1/10 unsupported, 0/10 contradicted.
- Findings to scrutinize: `finding-0001` (partially_supported),
  `finding-0005` (unsupported).
- See `run.json` and `stages/13_fact_check.json` for per-claim verdicts.
```

如果 attribution_audit 全部 supported 也写出来（"11/11 fully supported"），不省略。

#### 8.2.10 References

末尾 numbered list，**按首作者姓氏字母序**。每条：

```
[N] <First Author>, <Second Author>, ... (Year). <Title>. arxiv:NNNN.NNNNN.
```

3 个以上作者用 `, et al.`。从 `corpus_list_papers` 的 `authors` 字段拿数据（v3 已
保证 authors 不为空，见 commit 59311b6）。

**收录范围**：扫 markdown 全文，凡是出现 `arxiv:NNNN.NNNNN` 或 paper_id（如
`2604.00865` 不带前缀）的，**全部都要进 refs**。具体包括所有这些位置出现的
paper_id：

- Findings 的 inline cite 和 evidence footnote
- Taxonomy 各 bucket 的 `Papers: ...` 行
- Methods landscape 的 `Representative:` 和 `paper_ids`
- Gaps 的 `Evidence: ...` 行
- Open questions 末尾的 `(pid1, pid2)`

**只排除**真没在 markdown 出现的 corpus paper（这种情况是 coverage failure，应
回头补到至少一个 section，见 §7.7.3）。

典型情况：corpus 19 篇 → markdown 引到 18-19 篇 → refs 列 18-19 条。如果你的
refs 只有 8 条而 findings 引了 9 篇，说明你**漏看了 taxonomy / methods / gaps**
里出现的另外 10 篇 paper_id，必须补全。

#### 8.2.11 About this survey（最末节）

紧凑表格，元数据 transparency：

```markdown
## About this survey

| | |
|---|---|
| Pipeline | literature_surveyor v<version> |
| Generated | <ISO timestamp from run.json.completed_at> |
| Run ID | `<short_id>` (full id in `run.json`) |
| Models | main `<model>` · extraction `<extract_model>` |
| Corpus | <N> papers |
| Cost | $<X.XXXX> |
| Elapsed | <Ns> |
| Prompt provenance | <skill@hash, skill@hash, ...> |

This survey was generated by an automated multi-stage pipeline (search →
ingest → extract → synthesize → fact-check). Findings have been quote-verified
against source papers (see audit summary above). Treat as a research aid, not
an authoritative review.
```

数据从 `run.json` + 各 stage 的 `elapsed_s` / `llm_call.cost` 求和。

### 8.3 写作风格

- 句子要带具体数字、数据集名、方法名 —— 抽象 finding（"X 显著改善 Y"）扣分
- 跨论文 finding 要先点 shared pattern 再列论文，不要把多篇数字硬拼一句
- evidence quote 出现时保留原 markdown formatting（如 `+8.3%`），但去掉
  `**bold**` 之类的样式符号

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
- **省略 §8.2 列出的任何一节**（即使 Conflicts 为 0、Key numerical results 没数字、
  Audit summary 全 supported，也要把节标题和"无内容"说明写出来——transparency 优先）
- **cite_text 写成 `"Author et al. YYYY"` 占位符**（必须用 corpus 真作者名；
  `corpus_list_papers` 已自带 `authors` 字段）
- **从同一条 claim 数字拆出多条 finding**（如 paper 说 63.5% 是 retrieval+generation
  错误之和，不能拆成两条各报 32.5% / 31% 的 finding 除非论文真把这两个数字独立列出）
- **5-8 条 finding 全是 single-paper**（必须 ≥1/3 cross-paper，见 §7.7.1）
- **corpus 有 paper 完全没在 taxonomy/methods/gaps 任何一节出现**（coverage failure，见 §7.7.3）

## 范例

参考 `examples/`（如果存在）—— 标杆 survey 的长度、结构、引用密度。
