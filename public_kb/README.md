# 公共评审知识库

公共标准检索的无证据处理规则见 [`retrieval_no_evidence_policy.md`](retrieval_no_evidence_policy.md)，可复用判定模块见 [`public_evidence_policy.py`](public_evidence_policy.py)。

本目录保存清洗后的论文评审标准，不保存用户论文，也不保存针对具体论文生成的问答。

## 当前数据源（仅格式审查）

- `degree_format_standards.jsonl`：学位论文格式与写作规范，来源为《学位论文的格式检查（2026）》。
- NeurIPS 2026 仅保留 `venues/neurips_2026_standards.jsonl` 中的 7 条纯格式规则。
- ICML Reviewer Instructions 已用于科研论文内容评价（Soundness、Presentation、Significance、Originality等13条）；它不是作者格式规范，格式库仍需 Author Instructions 或 Call for Papers 本地副本。
- `venue_format_template.json`：新增期刊或会议格式要求时使用的空白配置模板。
- `venues/neurips_2026.json`：NeurIPS 2026 投稿格式库配置。只保留 Paper Format Instructions 中的 7 条纯格式规则；匿名、代码数据、LLM、预印本、回复和资金披露等非格式内容不进入该库。
- `venues/icml_2026_pending.json`：收到的 ICML Reviewer Instructions 不含作者论文格式要求，因此不导入 RAGFlow。需补充保存 ICML 2026 Author Instructions 和 Call for Papers 页面。

## 离线原始网页

- `venues/icml Reviewer Instructions 2026.html` 与同名 `_files` 目录组成 ICML 2026 Reviewer Instructions 的完整离线副本。
- `venues/neurips主赛道手册2026.html` 与同名 `_files` 目录组成 NeurIPS 2026 Main Track Handbook 的完整离线副本。
- HTML 与配套 `_files` 目录必须保存在同一目录下；移动、复制或备份时应成对处理，否则样式、公式、图标和脚本可能无法加载。
- 清洗后的 JSONL/TXT 用于 RAGFlow 检索，离线 HTML 用于来源追溯和人工核验，两者不要互相替代。

## 安全与来源约束

- 标准证据只能说明评价依据，不能证明论文实际完成了某项工作。
- 学位论文内容标准已重新启用，但来源是课程论文评分标准，标记为 `provisional_course_rubric`，不能当作所有学位论文的正式统一评分规范。

## RAGFlow 当前配置

当前启用的数据集：

- `public_degree_format_2026`：已绑定 `BAAI/bge-m3@silicon@SILICONFLOW`，29 条格式标准已导入并通过检索验证；同时保留原始 PDF `degree_format_source_2026.pdf` 供查看。
- `public_research_format_neurips_2026`：已绑定 `BAAI/bge-m3@silicon@SILICONFLOW`，7 条纯格式标准已导入并通过检索验证。
- `public_research_content_icml_2026`：已绑定 `BAAI/bge-m3@silicon@SILICONFLOW`，13 条 ICML 内容评价标准已导入并通过检索验证。
- `public_degree_content_2026`：已绑定 `BAAI/bge-m3@silicon@SILICONFLOW`，10 条临时学位论文内容标准已导入；原始来源为《计算机系统结构课程论文评分标准(2026-05-15)》。

实际 ID 保存在 `manifest.json`。公共服务账号与现有用户论文库账号属于不同 RAGFlow 租户，后端应分别使用用户论文库 Key 和公共库 Key。

## 逐库检索测试

`ragflow_retrieval_test.py` 内置了五个知识库的中文测试问题，当前统一使用相似度阈值 `0.2`：

- 学位论文内容：`0.2`
- ICML 科研论文内容：`0.2`
- NeurIPS 格式：`0.2`
- 用户论文：`0.2`
- 学位论文格式：`0.2`

这些阈值随逐库检索请求发送，不会被 RAGFlow Chat 的单一全局阈值覆盖。脚本不保存 API Key；单租户可设置 `RAGFLOW_API_KEY`，公共库和用户库分属不同租户时分别设置 `RAGFLOW_PUBLIC_API_KEY`、`RAGFLOW_USER_API_KEY`。

```powershell
$env:RAGFLOW_API_KEY = "你的 RAGFlow API Key"

# 查看知识库、阈值和内置问题，不连接 RAGFlow
python public_kb/ragflow_retrieval_test.py --list

# 运行一个知识库的全部内置问题
python public_kb/ragflow_retrieval_test.py --kb icml_content

# 运行自定义中文问题
python public_kb/ragflow_retrieval_test.py --kb user_paper --question "去掉L_gen后结果下降多少？"

# 运行五个知识库的全部测试
python public_kb/ragflow_retrieval_test.py --all
```

## 公共库独立检索脚本

四个脚本分别固定绑定一个公共知识库，相似度阈值均为 `0.2`。由于 RAGFlow 0.26.4 在包含原始来源和手工 Chunk 的多文档数据集上使用 `doc_ids` 可能错误返回空结果或内部错误，独立公共库脚本固定限定 `dataset_id`，并对返回 Chunk 的文档 ID 或预期索引文档名再次强制过滤。文档名校验用于兼容手工 Chunk 的内部 `doc_id` 与文档 API ID 不一致。低分结果会被再次过滤，其余结果按 `rerank_score`、`similarity`、`score` 的优先级降序排列，并返回证据摘录式答案。未提供 `--question` 时会运行该库的全部内置问题。

```powershell
$env:RAGFLOW_PUBLIC_API_KEY = "你的公共库 RAGFlow API Key"

python public_kb/degree_content_search.py --question "学位论文摘要需要包含哪些内容？"
python public_kb/icml_content_search.py --question "ICML如何评价论文的技术可靠性？"
python public_kb/neurips_format_search.py --question "NeurIPS论文是否需要检查清单？"
python public_kb/degree_format_search.py --question "英文缩写第一次出现时怎么写？"

# 不连接 RAGFlow，只检查绑定的数据集、文档和内置问题
python public_kb/icml_content_search.py --list

# 返回结构化答案和证据
python public_kb/degree_format_search.py --question "参考文献格式有什么要求？" --json
```

答案由检索到的原始标准片段直接组成，不依赖 RAGFlow Chat 或额外聊天模型，适合在模型供应商尚未配置完成时独立验证每个公共库。正式系统接入时应继续保留论文证据与公共标准证据分库检索，并由工作流负责最终综合。

需要重建或恢复导入时执行：

```powershell
python public_kb/public_kb_import.py `
  --source public_kb/degree_format_standards.jsonl `
  --dataset-id 13470cac831c11f1966fe5529e1d1cda `
  --document-id 17ae2c44831c11f1966fe5529e1d1cda
```

脚本会记录断点并跳过已经导入的标准；目标数据集没有 Embedding 模型时会拒绝导入，避免建立不可检索的 Chunk。
