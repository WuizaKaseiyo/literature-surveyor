# Golden papers fixture

固定的 paper 集，用于 v1/v2 抽取对比（`tests/eval/compare_extraction.py`）。

## 文件

- `papers.jsonl` — 每行一个 paper 记录，schema 与 corpus `papers.jsonl` 完全一致（见 `tools/corpus_store/TOOL.md`）

## 推荐组成（5 篇）

为了覆盖不同情况，建议挑：

1. **短 workshop paper**（≤ 8 页，full_text_md ~ 25K chars） — 验证 v2 不会因为过度切分浪费 token
2. **中等 conference paper**（10–15 页，~40–60K chars）— 主流情况
3. **长 technical report**（25+ 页，~100K+ chars）— v1 的 30K 截断在这里丢最多东西，v2 章节切分应该改善最明显
4. **method-heavy paper**（algorithm / loss 改进，弱实验）— 测 methodological claim 抽取
5. **negative-result / contradicting paper**（声明"X 不 work"）— 测 claim_type 分布是否合理

## 怎么填进来

OMC 环境里有 `arxiv_search` + `pdf_extract` 工具。一个简单的 bootstrap：

```python
from tools.arxiv_search.arxiv_search import arxiv_search
from tools.pdf_extract.pdf_extract import pdf_extract

ids = ["2401.XXXXX", "2403.XXXXX", ...]  # 你选的 5 篇
for aid in ids:
    meta = arxiv_search.invoke({"query": f"id:{aid}", "limit": 1})["results"][0]
    text = pdf_extract.invoke({"pdf_url": meta["pdf_url"]})["markdown"]
    paper = {**meta, "full_text_md": text, "source": "arxiv"}
    # append to papers.jsonl
```

或者从已有项目的 corpus 拷 5 行进来。

## Schema 要求

每行至少要有：

- `id` — arxiv id / doi / s2 id，唯一
- `title`
- `full_text_md` — 完整 markdown 正文（不是 abstract）。长度 > 5000 chars，否则 v1/v2 区别不出来
- `abstract` — 摘要
- `year`, `authors`, `venue` — 可选但建议填

## 不要把这个 jsonl 提到 git LFS

5 篇大概几 MB，常规 git 完全 ok。但不要扩到 50 篇 —— eval 跑一次成本就成问题了。
