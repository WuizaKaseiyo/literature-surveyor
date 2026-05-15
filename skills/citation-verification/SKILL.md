---
name: citation-verification
description: Pre-submission citation verification protocol
autoload: false
---

# Citation Verification

提交 stage2 之前**必须**调 `verify_citations` 和 `fact_check_rendered_survey`。

`verify_citations` 检查 citation 对象是否真实存在；`fact_check_rendered_survey`
检查最终 markdown 中每个句子级 claim 是否真的被它旁边的 citation 支持。

## 流程

```python
md_text = read('stage2_literature_surveyor.md')
cite_result = verify_citations(md_text)
fact_result = fact_check_rendered_survey(md_text, use_llm=True)

if cite_result["unverified_count"] == 0 and fact_result["blocking_count"] == 0:
    # 提交
    pass
else:
    # 修复
    for cite in cite_result["unverified"]:
        # 选项 A：在 corpus 中找替代真实 paper
        alt = corpus_search(cite["context_around"])
        if alt:
            replace_in_md(cite["raw"], format_cite(alt[0]))
        # 选项 B：删除该 cite + 对应 claim
        else:
            remove_claim_with_cite(cite["raw"])
    for item in fact_result["items"]:
        if item["verdict"] in ["unsupported", "contradicted", "source_irrelevant"]:
            rewrite_or_remove_claim(item["claim_text"])
        elif item["verdict"] == "partially_supported":
            soften_wording(item["claim_text"])
    # 重新 verify
    cite_result = verify_citations(read('stage2_literature_surveyor.md'))
    fact_result = fact_check_rendered_survey(read('stage2_literature_surveyor.md'), use_llm=True)
```

## 引用格式（统一）

输出 markdown 里所有 cite 必须用以下三种之一，便于 verify_citations 正则识别：

| 形式 | 例 | 适用 |
|---|---|---|
| arxiv | `[Smith et al. 2024, arxiv:2401.12345]` | arxiv preprint |
| DOI | `[Lee et al. 2023, doi:10.1109/TPAMI.2023.1234567]` | 期刊/会议正式发表 |
| Semantic Scholar | `[Park et al. 2024, S2:abcd1234]` | 没有 arxiv/DOI 但 SS 收录 |

不要用：
- ❌ `[Smith et al.]` — 没 ID 无法验证
- ❌ `[1]` 数字脚注 — 不知道指哪个 paper
- ❌ `(Smith, 2024)` — APA style 但没 ID

## verify_citations 返回结构

```json
{
  "verified": [
    {"raw": "[Smith et al. 2024, arxiv:2401.12345]", "id": "2401.12345",
     "source": "arxiv", "title": "...", "matched_corpus": true}
  ],
  "unverified": [
    {"raw": "[Wang et al. 2023, arxiv:2308.99999]",
     "reason": "arxiv ID 2308.99999 not found",
     "context_around": "...100 chars before/after..."}
  ],
  "verified_count": 23,
  "unverified_count": 1,
  "verification_method": "live API + corpus cross-check"
}
```

## fact_check_rendered_survey 返回结构

```json
{
  "supported_count": 20,
  "partial_count": 3,
  "unsupported_count": 1,
  "contradicted_count": 0,
  "source_irrelevant_count": 0,
  "source_not_in_corpus_count": 0,
  "blocking_count": 1,
  "items": [
    {
      "claim_text": "RLHF reduces hallucination by 30% in 7B models.",
      "citation_raw": "[Smith et al. 2024, arxiv:2401.12345]",
      "paper_id": "2401.12345",
      "verdict": "unsupported",
      "evidence_quote": "...",
      "explanation": "The source is about RLHF but does not contain the 30% value."
    }
  ]
}
```

## 失败处理优先级

如果 unverified > 5：很可能你之前没用 search_* 工具就开始写。回到 step 5 重新爬 paper。

如果 unverified 1-5：逐个修复（替代或删除）。

如果 verify_citations 或 fact_check_rendered_survey 工具自身报错：报 `verification_skipped: <reason>` 在 stage2.json 里，但**不要直接提交** — 这是 critic 必 reject 的情况。
