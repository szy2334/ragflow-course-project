# UMAP Chunk 结构优化报告

## 结论

已修复 `paper_metadata` Chunk 混入首页正文的问题。该 Chunk 现在只保留题名与文件名；摘要与正文继续由独立的 `abstract` / `paragraph` Chunk 承载。

本次未修改 RAGFlow 的线上知识库、检索参数或既有导入映射。

后续已按需求补充脚本侧的跨语言检索支持；同样未调用线上检索或重建索引。

## 问题与根因

旧实现将论文标题、文件名与首页前三个段落拼入同一个元数据 Chunk。UMAP 实测中，该 Chunk 包含 1,871 个字符，并包含摘要起始句 `Multimodal emotion recognition based on physiological signals`。

因此，中文内容型问题（例如“论文要解决的核心问题是什么？”）在跨语言关键词检索不足时，可能误召回以“论文标题”开头的元数据 Chunk，而不是独立摘要 Chunk。

## 实施内容

- 修改 `second_clean.py` 的 `build_metadata_chunk`：不再读取或拼接首页段落。
- 元数据文本增加明确作用域：`仅用于题名、文件信息等书目查询`。
- 写入可追溯标记：
  - `scope: bibliographic_only`
  - `excluded_content: page_one_paragraphs`
- 将本地 `retrieval_weight` 从 `0.9` 调整为 `0.2`，表达低优先级意图。请注意：现有 RAGFlow 手工 Chunk 导入接口没有使用该字段，它不会单独改变线上排序。
- 同步修改 `data_pipeline/user_paper_agent_gateway/second_clean.py`，避免两个入口的 Chunk 规则漂移。

## 验证结果

| 检查项 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 元数据 Chunk 字符数 | 1,871 | 141 |
| 含摘要起始句 | 是 | 否 |
| `paper_metadata` 角色 | 是 | 是 |
| 标题保留 | 是 | 是 |
| 溯源字段 | 无 | `bibliographic_only` |

已通过 Python AST 语法解析与构造函数断言验证。两个入口目录中的 `second_clean.py` 文件哈希一致。

## 后续上线注意事项

1. 使用更新后的 `second_clean.py` 重新生成 Chunk。
2. 不要向含旧 Chunk 的同一 RAGFlow 文档直接追加导入；应先删除旧文档或导入新文档/知识库，避免旧的混合元数据 Chunk 继续被召回。
3. 元数据 Chunk 内容变化会生成新的 `source_chunk_id`。标题类黄金用例及新的 RAGFlow `chunk_mapping.jsonl` 应在重新导入后一起更新。
4. 完成索引替换后，使用 `--no-resume` 重跑 QA 基线，再继续调整多语种 Embedding、向量权重、阈值和 Reranker。

## 跨语言检索脚本支持（补充）

`qa_execute.py` 现支持 `.env` 参数 `USER_PAPER_QA_CROSS_LANGUAGES` 和命令行参数
`--cross-languages`。对于中文提问、英文论文，配置为：

```text
USER_PAPER_QA_CROSS_LANGUAGES=English
```

执行器会向 RAGFlow `/retrieval` 发送：

```json
{"cross_languages": ["English"]}
```

检索结果文件现在会记录完整的 `retrieval_config`。当跨语言设置、候选数、阈值或向量权重发生变化时，旧结果不会被 `--resume` 错误复用。

已用模拟 HTTP 会话验证请求体包含 `cross_languages: ["English"]`，并使用 `--dry-run` 验证当前 UMAP 配置会生效。

## 相关问题检索基线（补充）

当前阶段的基线只覆盖论文内可回答的问题，不再包含“作者家庭住址”之类的无关问题。`answerable` 字段仍保留在黄金数据中作为描述信息，但不会影响检索或答案评分；RAGFlow 返回近邻 Chunk 不会因此被判为失败。

黄金集由 8 个扩展为 16 个相关问题，新增覆盖：模态缺失成因、两阶段训练流程、专家与门控结构、实现超参数、对比方法分类、预训练作用、总预训练损失和 `L_gen` 消融结果。所有新增用例的预期源 Chunk 均已在当前 `03_chunks/chunks.jsonl` 中验证存在。
