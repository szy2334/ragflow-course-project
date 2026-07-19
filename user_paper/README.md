# 用户论文 PDF 摄取流水线

该目录实现以下阶段：

1. `mineru_clean.py`：调用 MinerU，或读取已有 MinerU 产物，生成稳定的业务 Block 和媒体对象。
2. `baidu_ocr.py`：按图片、图表、表格调用百度 OCR，并保存原始响应与归一化结果。
3. `second_clean.py`：生成正文、摘要、公式、表格、图片和参考文献等结构化 Chunk。
4. `ragflow_import.py`：以 manual Chunk 方式导入 RAGFlow，保存业务 Chunk 与 RAGFlow Chunk 映射。
5. `run_pipeline.py`：串联上述阶段。

## 一次配置，直接运行

在本目录执行：

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# 编辑 .env，只填写本机的 MINERU_TOKEN、BAIDU_OCR_API_KEY、
# BAIDU_OCR_SECRET_KEY 和 RAGFLOW_API_KEY。
python .\run_pipeline.py --print-config
python .\run_pipeline.py
```

`.env` 包含 PDF 路径、输出目录、MinerU/百度 OCR/RAGFlow 地址、数据集名称和严格 OCR 开关。命令行参数仍可覆盖 `.env` 中的默认值，例如：

```powershell
python .\run_pipeline.py --pdf .\another-paper.pdf --run-name another-paper
```

`--print-config` 只显示配置路径、普通设置和每个凭据是否已配置，绝不会显示密钥内容。`.env` 中的相对路径以该 `.env` 所在的 `user_paper` 目录为基准，因此可从仓库根目录或本目录执行脚本。

`.env` 已被 Git 忽略，禁止提交真实密钥。`.env.example` 仅含字段名和可安全提交的默认值。

`USER_PAPER_STRICT_SPECIALIZED=true` 要求百度表格识别 V2 与 PaddleOCR-VL 均成功；任一专项接口失败都会终止流程，不会进入普通 OCR 或 MinerU 表格降级路径。`USER_PAPER_FORCE_OCR=true` 会重新调用百度 OCR；只想复用已有 OCR 结果时改为 `false`。

更换 RAGFlow 的聊天模型不需要重新运行本流水线；只有 PDF、OCR、Chunk 规则或索引内容发生变化时才需要重新运行。

主要产物：

```text
01_mineru_clean/document.json
01_mineru_clean/blocks.jsonl
01_mineru_clean/media_objects.jsonl
02_baidu_ocr/ocr_results.jsonl
03_chunks/chunks.jsonl
03_chunks/ragflow_chunks.jsonl
03_chunks/quality_report.json
04_ragflow/chunk_mapping.jsonl
04_ragflow/import_summary.json
```

## QA 执行与评分

`qa_execute.py` 会读取 `.env` 中 `USER_PAPER_QA_GOLDEN` 指向的 `golden.json`，直接调用
RAGFlow 的检索接口，并将命中的 Chunk 引用写到 `USER_PAPER_QA_RESULTS`。它从
`USER_PAPER_QA_CHUNK_MAPPING` 自动识别当前论文的数据集与文档 ID；可按需在 `.env` 用
`USER_PAPER_QA_DATASET_ID` 和 `USER_PAPER_QA_DOCUMENT_ID` 覆盖。

```powershell
python .\qa_execute.py
python .\qa_baseline.py
```

若问题为中文而论文 Chunk 以英文为主，在 `.env` 设置
`USER_PAPER_QA_CROSS_LANGUAGES=English`，脚本会向 RAGFlow `/retrieval`
请求传入 `cross_languages: ["English"]`。该设置或检索参数变更后，即使
`USER_PAPER_QA_RESUME=true` 也不会复用旧结果。

`golden.json` 是人工审核后的评测题与预期事实/证据。执行器只生成可复现的检索结果
`results.json`；评分器在该模式下只判定检索与引用命中，不评判 LLM 答案。
