# Literature Surveyor 优化清单

三条主线：
1. **Claim 抽取**：从一次性截断 → 章节感知 + quote 回校 + 结构化 scope
2. **Fact check**：从 BM25 重新选证据 → 优先用已抽 claim 的 evidence_quote 当 anchor
3. **落库管理**：建 global corpus（跨项目复用）+ 把 claims 从只写变成一等数据（可检索、给 fact check 用、给渲染用）

落地建议顺序：**D + E1 先动**（收益大改动小）→ 再做 A/B/C（global corpus + claim 重写）→ 最后 F/G/H 串起来。

---

## A. 数据层 / Schema

### A1. `schemas/literature_survey_schema.py`
- [ ] `Finding` 增 `claim_ids: list[str]`（finding 直接挂支撑 claim）
- [ ] `AttributionAuditItem` 增 `matched_claim_id: str`、`matched_claim_quote: str`
- [ ] `Conflict` 增 `claim_a_id`/`claim_b_id`（现在只有 paper_id + text，丢了 claim 维度）
- [ ] 新增 `ClaimRecord` Pydantic 模型，字段：`id, paper_id, claim_text, claim_type(Literal), evidence_span, evidence_quote, evidence_quote_verified, source_section, section_path, confidence, applies_to_dims(dict), applies_to(legacy str), extracted_at, model`
- [ ] `extract_claims.py` 里本地的 `Claim` 改为复用上面的 `ClaimRecord`（停止再写本地副本）

### A2. Corpus 分层存储（`tools/corpus_store/corpus_store.py`）—— ✅ 完成

**实施状态**：opt-in via `LITSURVEY_GLOBAL_CORPUS_DIR`。不设 = legacy 单租户行为完全不变；设了 = 启用全局/项目分层。所有 80 个老测试 0 改动通过 + 17 个新测试（12 layered behavior + 5 migration）+ 1 个 layered-mode e2e 通过。


**设计原则**：paper 和 claim 是实体，全局池里唯一存一份；项目目录只存"引用清单"（refs.jsonl，每行一条边记录）。pdf_extract 和 extract_claims 是流水线最贵的两步，跨项目复用主要省的就是这两步。

**已锁定的设计决策**：
- **project_id = `sha256(absolute_cwd)[:12]`**，自动写进 `project_meta.json`。换工作目录 = 换项目身份（OMC workspace 路径稳定，这条假设成立）
- **BM25 IDF 用 global 统计**，搜索分数跨项目可比，小语料 IDF 噪声不会爆
- **保留 `search_history.jsonl`**，记录每次 query 和命中的 paper_id 列表（debug + 回溯都要用）

**目录布局**：
- [x] `$LITSURVEY_GLOBAL_CORPUS_DIR`（默认 `~/.litsurvey_corpus_global/`）：`papers.jsonl`, `claims.jsonl`, `corpus_index.json`, `claims_index.json`, `.lock`
- [x] `$LITSURVEY_CORPUS_DIR` 项目目录：`refs.jsonl`, `project_meta.json`
- [ ] `search_history.jsonl` 未做（design 里说留，目前未需）
- [ ] global `meta.json`（version/last_compacted）未做

**路径解析**：
- [x] `_global_dir()`、`_project_dir()`、`_in_layered_mode()` 三个函数加好
- [x] 项目目录无 `LITSURVEY_CORPUS_DIR` + 不像 workspace 时返回 None（自动 fallback global-only 单租户）
- [x] 所有 4 个工具（corpus_store / extract_claims / claim_store / fact_check_rendered_survey / verify_citations）的 `_corpus_dir()` 都加了 layered 分支

**实体 schema**：
- [x] `refs.jsonl` 行结构：`{paper_id, source_query, found_via, added_at, kept}`（按 design 草稿；search_round/relevance_note 暂未填）
- [x] 同 paper_id 允许多条 ref 记录（test_add_idempotent_global_but_appends_ref_for_distinct_query 验证）
- [x] `project_meta.json`：`{project_id (sha256(cwd)[:12]), project_path, created_at, retired_paper_ids}`
- [ ] `papers.jsonl` 仍保留 `source_query` 字段（移到 refs 是 nice-to-have，目前留着无害）

**索引策略**：
- [x] **不维护 per-project 索引**：项目内 search = 查 global 索引 + 按 refs 过滤；test_search_scope_project_filters_to_refs 验证
- [ ] 进程内 LRU 缓存 + mtime 检查（暂未做，每次 load 一次 JSON；< 1000 paper 时 acceptable）
- [ ] ≥ 1000 paper 时迁 sqlite（TOOL.md 留 marker）

### A3. 并发写锁 —— ✅ 完成
- [x] 封装 `_global_write_lock()` context manager（fcntl flock，legacy mode 无锁直 yield）
- [x] `corpus_add_paper` 的 (load → check → write paper → load index → update → save) 整段被锁包住

### A4. 索引版本号
- [ ] `corpus_index.json` 增 `version: 2`
- [ ] 启动时检查 version mismatch 自动 rebuild

---

## B. `corpus_store` 工具改造

### B1. `corpus_add_paper` —— ✅ 完成
- [x] 默认写 global + 在 project `refs.jsonl` append 一条 ref
- [x] 返回 layered 时 `global_status` / `project_status` + `corpus_size`（project）+ `global_corpus_size`
- [x] 同 paper 不同 `source_query` 各自 append 一条 ref（test_add_idempotent_global_but_appends_ref_for_distinct_query 验证）
- [ ] 单独的 `scope` 参数没加（layered 模式语义就是"双写"，legacy 模式只写一处；scope 在 search/list 上才有意义）

### B2. `corpus_search` —— ✅ 完成
- [x] 新参 `scope: Literal["project","global","both"] = "both"`
- [x] 结果每条加 `from_project: bool`（layered 模式下）
- [x] 返回值增加 `project_corpus_size` / `global_corpus_size`

### B3. `corpus_status` —— ✅ 完成
- [x] 返回 `corpus_size`（project）+ `global_corpus_size` + `claims_count`（project）+ `global_claims_count`
- [x] 增 `mode: "layered"` 标签 + `global_dir` / `project_dir` 路径
- [ ] `recent_projects: list[str]` 未做（要扫多项目目录，复杂度高 vs 价值低）

### B4. 新工具 + 迁移 —— ✅ 完成
- [x] `tools/corpus_store/migrate_to_global.py` 一次性脚本（dry-run / 幂等 / .bak 备份），5 个测试覆盖
- [ ] `corpus_promote_to_global(paper_id)` @tool 未单独做（迁移脚本就是它的批量版本；runtime 通过 corpus_add_paper 也能 promote 单篇）

---

## C. `tools/extract_claims/extract_claims.py` 重写

### C1. 章节切分
- [ ] 新增 `_split_sections(full_text_md) -> list[{section_path, heading, text}]`，按 markdown heading 解析
- [ ] **删除** `MAX_INPUT_CHARS = 30000` 全局截断
- [ ] 改为每个 section block 限长（建议 8K），超长 section 再按段落切

### C2. 按章节单独抽
- [ ] 循环每个 section block，LLM prompt 限 1–4 claim/section
- [ ] 注入 `section_path` 到产物字段
- [ ] 跳过 `^(references|acknowledg|appendix|related work)` 章节
- [ ] 总数软目标 5–15，`MIN_CLAIMS_PER_PAPER=3` 兜底

### C3. evidence_quote 回校
- [ ] 实现 `_quote_in_source(quote, full_text_md)`：
  - 先 exact substring（normalize 空白 + unicode）
  - 再退化 `difflib.SequenceMatcher.ratio() >= 0.85` 的模糊匹配
- [ ] 不通过 → `evidence_quote_verified = False`，记 warning，**保留**但下游可降权

### C4. `applies_to` 结构化
- [ ] prompt 要求返回 `applies_to_dims` dict：`model_size / dataset / domain / language / regime / metric`
- [ ] 保留 free-form `applies_to` string 做向后兼容（自动拼出）
- [ ] 这是 conflict detection 的 join key，**强约束非空**

### C5. `claim_type` 严格化 —— 🛑 故意不做（audit 后决定）
- 实测 DS-V3 在 248 个 real claim 上 **100% 守 canonical** 4 个值，0 变体
- 加 Literal 是纯防御；对**今天**的 LLM 行为 0 改善
- 未来真观察到模型乱来再加，避免为理论 bug 加代码

### C6. Retry / partial parse —— ✅ 完成
- [x] 共享 `_parse_claims_response(raw)` helper：strict array → strict claims-key → strict single dict → salvage `{…}` blocks（string-aware）
- [x] `_salvage_json_objects(raw)` 处理 prose 前缀 / max_tokens 截断 / 字符串内含 `{}`
- [x] `_call_llm_with_retry`：仅空响应触发，max 1 retry，0.3s backoff（不无限重试）
- [x] v1 + v2 调用点改造，parse_mode (`strict_array` / `salvaged_N` / `failed` / `empty`) 透出到 warnings
- [x] 13 个新 LLM-free 测试覆盖所有 parse 路径 + retry 行为（mock sleep 避免拖慢）

### C7. 写入 global
- [ ] `claims.jsonl` 写 global 而不是 project
- [ ] 更新 `claims_index.json`（BM25 over `claim_text + evidence_quote`）

---

## D. `tools/claim_store/` 从零写起（目前空目录）

### D1. 新文件 `tools/claim_store/claim_store.py`
- [x] `claim_search(query, paper_ids=None, claim_type="", top_k=10)` —— BM25 over claim 内容，可过滤
- [x] `claim_list_by_paper(paper_id)`
- [x] `claim_get(claim_id)`
- [x] `claim_status()` —— 总数、按 type 分布、`unverified_quote_count`
- [x] `claim_find_evidence(claim_text, paper_id, top_k=3)` —— 给 markdown 句子，在该 paper 的 claim 集合里找最相近的

### D2. Meta
- [x] `tools/claim_store/TOOL.md`
- [x] `tools/claim_store/tool.yaml`
- [x] `tools/manifest.yaml` 注册全部 5 个 @tool

---

## E. `tools/fact_check_rendered_survey/` 增强

### E1. 先查 claim 再回退 BM25（最高优先级）
- [x] `_check_claim_against_paper` 流程：
  1. 读 `claims.jsonl`，token-overlap (Jaccard ≥ 0.12, ≥3 共享 token) 找最匹配 claim → 用 `matched.evidence_quote` 作为 LLM context，记 `matched_claim_id`
  2. 没命中再走原 `_select_context(BM25)`
- [x] LLM judge prompt 里"extracted claim 文本 + evidence_quote + section"作为强证据 block，BM25 段落作为补充 context
- [x] 公开 `@tool` 加 `use_extracted_claims: bool = True` 参数，便于和 BM25-only baseline 做对比
- [x] 每个 item 透出 `matched_claim_id` + `matched_claim_quote`
- [x] schema `AttributionAuditItem` 同步加这两个字段
- [x] 4 个新测试（anchor 命中 / 命中失败回退 / 无 claims 文件无回归 / 显式关闭 anchor）+ 5 个老测试全过
- [x] **效果验证**（factcheck_20260517_201149 报告）：
  - blocking_count: anchored=7, baseline=5（**多抓 2 个错引用**）
  - LIMA 65% vs 43% case: anchored `contradicted`, baseline `partially_supported`（最 gold 的一例——baseline 把数字硬错软化为 partial 漏过）
  - DPO stability: anchored `partially_supported`, baseline `supported`（baseline 把相近词混淆为支持）
  - elapsed 几乎相同（29.0s vs 28.9s），anchor hit 72%（13/18）
  - 顺带修复 `_llm_judge` 对字符串 confidence（DS-V3 返回 `"high"`）crash 的 bug，加 `_parse_confidence`

### E2. 数字匹配放宽 —— ✅ 完成（差值匹配延后）
- [x] 单位归一：`%` ≡ `percent` ≡ `pct`；`K/M/B/T` 大小写；保留有符号
- [x] 容差：`abs(a-b)/max(|a|,|b|) <= 0.05` 同单位内匹配
- [ ] ~~跨单位等价（30% ≡ 0.30）~~ 故意不做：lossy + 上下文相关，留给 LLM judge
- [ ] 差值匹配（"reduced by 13pp" ≡ `42% → 29%`）—— 延后，复杂度高、可用 LLM judge 兜住
- [x] `_extract_numbers_with_unit` + `_numbers_match` 新 helper；`_heuristic_judge` 切换；老 `_numbers` 保留为 deprecated 但不删
- [x] 5 个新测试（容差命中 / percent 词形 / 单位错配 / 容差外仍 fail / extract helper 单位表）

### E3. 多 cite 聚合 —— ✅ 完成
- [x] 按 `attribution_id` 聚合 cite，最终 verdict = 按 supported > partial > contradicted > unsupported > ... rank 取 best
- [x] `blocking_count` 改为按聚合后的 attribution 统计；旧的 per-cite 数字保留为 `blocking_cite_count`（backward compat）
- [x] 输出额外 `per_attribution: [{attribution_id, claim_text, final_verdict, final_matched_claim_id, sub_results}]`
- [x] 单 cite 细节仍在 `items` 里保留
- [x] 3 个新测试（多 cite 一对一错 → non-blocking / 多 cite 全错 → blocking / 单 cite backward compat）+ 老 fact_check 测试 0 改动通过

### E4. `source_not_in_corpus` 主动回填（可选）
- [ ] 新参 `auto_fetch_missing: bool = False`
- [ ] 命中 missing → arxiv/s2/openalex 搜 → pdf_extract → corpus_add_paper(global) → 重判
- [ ] 默认 False（避免静默改 state）

### E5. 扩窗重试 —— ✅ 完成
- [x] LLM verdict in {unsupported, partially_supported} 且 confidence < 0.6 时，BM25 窗口扩到 12K chars / 24 sentences 再问一次
- [x] 两次取**更严**的（rank min）；retry 更宽松时**保留原始**（不放松判断）
- [x] `@tool` 加 `expand_on_low_confidence: bool = True` 参数；items 透出 `expanded_retry ∈ {"","took_stricter","kept_original","retry_failed"}`
- [x] `_select_context` 参数化 `max_chars` + `max_sentences`（兼容原 6K/12s 默认）
- [x] 5 个新单测（retry-flips / retry-kept-original / 高置信不 retry / supported 不 retry / flag 关闭不 retry）—— 都 mock LLM judge 无外部依赖

### E6. 接 stage2.json —— ✅ 完成
- [x] 新参 `survey_json: dict = None`；与 `markdown` 互斥（survey_json 优先）
- [x] `attributions_from_survey_json` helper：每个 Finding → 1 attribution，cite_text 解析出 (kind, id)
- [x] `preferred_claim_id` 通过 `_check_claim_against_paper` 直接定位 claim_id，**跳过 token-overlap 匹配**
- [x] 输出加 `input_mode ∈ {"survey_json", "markdown", "empty"}`，调用方可知道走的哪条路
- [x] markdown 入口完全保留作为兜底，老 caller 一行不动
- [x] 6 个新测试覆盖 survey_json 主路径 / preferred claim 优先 / 无 preferred 时 fallback / 双 input survey 优先 / 空 findings / 仅 markdown

---

## F. 新工具 `tools/detect_conflicts/` —— ✅ 完成

- [x] 读 global `claims.jsonl`（A2 layered 路径已统一），可按 `paper_ids` 限定 project scope
- [x] 用 `applies_to_dims` 做 join key（bidirectional substring match）— v2 dims 终于派上用场
- [x] LLM judge per pair；conjecture × conjecture 自动跳过
- [x] 输出匹配 `schemas.literature_survey_schema.Conflict` 的 dict 列表，含四级判定
- [x] 入参：`paper_ids=None, use_llm=True, model=..., level_filter=None, min_topic_jaccard=0.08, max_pairs=50`
- [x] `tools/manifest.yaml` 注册
- [x] `use_llm=False` 模式返回 candidates 不调 LLM（cheap dry-run）
- [x] 14 个新单测全过（pair filter + share_setting + topic_overlap + tool-level）
- [x] **实测验证**（5 篇 fixture paper × 140 v2 claims）：
  - 9,730 naive pairs → **11 candidates**（3 个数量级压缩，filter 有效）
  - 11 LLM judges → 0 conflicts（5 篇 paper 主题互补不矛盾，无假阳性）
  - 注入 2 条人造矛盾 claim（同 dim 设定，结果相反）→ LLM **正确判 `direct` level, confidence=0.95**，自然语言描述精准

---

## G. 渲染期把 claim 串进去

### G1. SKILL.md 流程修改
- [x] `Finding` schema 加 `claim_ids: list[str]` 字段（A1 推后到此处一起做）
- [x] `skills/systematic-review/SKILL.md` 新增 **step 7.5**：每条 finding 写进 stage2.json 前必跑 `claim_search` 找 backing claim，记录 `claim_ids`；硬约束 `claim_ids` 非空
- [x] Step 8 JSON 示例改为带 `claim_ids` 的完整 Finding
- [x] Step 9 加 `matched_claim_id` ↔ `claim_ids` 交叉校验流程（fact_check 实际命中的 claim 应在 finding 声明的 claim_ids 里）
- [x] Step 9 fact_check 调用可改传 `survey_json=...`（E6 fast path）—— 工具已支持；SKILL.md 待补示例

### G2. Markdown 渲染
- [x] finding 后 evidence footnote 强约定：`> evidence: "..." — Section X (arxiv:XXXX#claim-N)`
- [x] `claim-extraction/SKILL.md` 同步：`evidence_quote` 必须 verbatim（v2 抽取期已回校；v1 凭 LLM 复述 ~95% pass rate）
- [x] 2 个新 schema 测试（claim_ids 默认空 / 接受 list）+ 73 个老测试全过
- [x] **Evidence footnote 格式断言被验证**：`test_parser_ignores_evidence_footnote` 确认 `> evidence: "..." (arxiv:XXX#claim-N)` 不会被 fact_check parser 重复算 cite（圆括号 + blockquote 段尾无 cite 会被丢）
- [x] **端到端闭环验证**：`test_e2e_finding_claim_ids_match_factcheck_matched_claim_id` 跑 Finding(claim_ids=...) → 渲染 markdown → fact_check → `matched_claim_id` 落回 `Finding.claim_ids`；step 9 交叉校验流程的 happy path 是机器可验证的
- **77 个测试全过**（73 老 + 4 新：2 schema + 2 footnote/e2e）

---

## H. Manifest / Skill 同步

### H1. `tools/manifest.yaml`
- [ ] `claim_search, claim_list_by_paper, claim_get, claim_status, claim_find_evidence`
- [ ] `detect_conflicts`
- [ ] `corpus_promote_to_global`

### H2. `skills/conflict-detection/SKILL.md`
- [ ] 算法段改为调 `detect_conflicts` 工具说明
- [ ] 强调 `applies_to_dims` 是 join key，dim 缺失的 claim 不参与

### H3. `skills/claim-extraction/SKILL.md`
- [ ] 加 `applies_to_dims` 表格化字段示例
- [ ] 加 `evidence_quote` 必须原文可回校的硬约束
- [ ] section-aware extraction 样例 prompt

### H4. `skills/systematic-review/SKILL.md`
- [ ] Step 1：global ≥ 10 篇匹配可跳过 step 3–5（充分复用）
- [ ] Step 8：`claim_ids` 填写要求
- [ ] Step 9：`survey_json` 入参

---

## I. 测试（`tests/`）

- [ ] `test_extract_claims_section_split.py` —— mock LLM，验证按 section 切分 + 数量分布
- [ ] `test_extract_claims_quote_verify.py` —— 故意错的 quote → `verified=False`
- [ ] `test_claim_store.py` —— add → search → list → get 全链路
- [ ] `test_corpus_global_project.py` —— 两项目共享 global，project refs 隔离
- [ ] `test_fact_check_uses_claims.py` —— 验证 fact_check 优先命中 claim 而非 BM25
- [ ] `test_fact_check_multi_cite_aggregation.py` —— 句尾两 cite 一支持一不支持，attribution 判 supported
- [ ] `test_detect_conflicts.py` —— 同 dim 矛盾的两 claim → conflict 输出

---

## J. 落地顺序

1. A1 schema 拓展（兼容字段，无破坏）
2. A2 + A3 + B：corpus 分层 + 锁 + 工具改造（旧路径 fallback）
3. B4 + 迁移脚本：现有数据灌 global
4. C：extract_claims 重写，旧 jsonl 用 migration 补字段
5. **D：claim_store 三个读工具上线**（最小改动最大收益的入口）
6. **E1 + E3：fact_check 接 claim + 多 cite 聚合**（第二批最高 ROI）
7. E2 / E4 / E5 / E6：fact_check 其余增强
8. F：detect_conflicts 工具
9. G/H：prompt + skill 同步，挂 `Finding.claim_ids` 串起来
10. I：测试随每段一起加

---

## 备注

- 当前 `tools/claim_store/` 是空目录（只有 `__pycache__`），`claims.jsonl` 写入后**完全没有任何工具读它**，这是最大的死数据问题。
- 当前 corpus 强绑 CWD，同一 arxiv paper 在 N 个项目里被重复抓 + 重复抽 claim，pdf_extract 和 extract_claims 是流水线最贵的两步，纯浪费。
- `fact_check_rendered_survey` 已经实现了句子级 attribution 解析（这部分很好），主要问题是没接已抽的 claim 当 anchor，全靠 BM25 重新找证据。
