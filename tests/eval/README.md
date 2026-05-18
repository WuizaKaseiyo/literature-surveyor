# Eval harness

不是单元测试 —— 是用来跨版本对比抽取/事实核查产物的离线脚本。要 LLM API key 才能跑。

## compare_extraction.py

对比 `extract_claims` 不同 version 在同一份 fixture 上的产物。

```bash
# 用默认 fixtures（tests/fixtures/golden_papers/papers.jsonl）
python tests/eval/compare_extraction.py

# 指定 fixtures 和要对比的版本
python tests/eval/compare_extraction.py --versions v1 v2 --model openai/gpt-4o-mini
```

需要 `OPENAI_API_KEY` 或 `OPENROUTER_API_KEY`。

输出：`tests/eval/reports/extraction_<timestamp>.md`，git tracked，每改一次 prompt / 切分逻辑跑一遍。

## 指标

每个 paper × 每个 version：

- `n_claims` — 抽到的 claim 数
- `section_dist` — 按 section bucket 分桶（intro/method/experiment/result/...）
- `evidence_verified_rate` — `evidence_quote` 能在 `full_text_md` 里找到的比例（exact substring + 4-gram overlap fallback）
- `applies_to_filled_rate` — `applies_to` 或 `applies_to_dims` 非空的比例
- `type_dist` — claim_type 分布
- `approx_output_chars` — 输出大小代理（token 成本）

聚合：每个 version 的均值 + 散点（paper 长度 vs claim 数）。

## reports/

时间戳命名，永久保留。改 prompt 前后的报告并排看曲线，比肉眼读 claim 列表靠谱。
