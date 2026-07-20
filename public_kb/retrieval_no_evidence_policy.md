# 公共评审库“无有效证据”处理规则

本规则用于公共评审标准检索。它与“公共库服务不可用”是两种不同情况。

## 结果状态

| 状态 | 含义 | 是否允许确定评分 |
| --- | --- | --- |
| `found` | 检索到满足范围和相关性阈值的标准证据 | 允许，但仍需绑定论文证据 |
| `no_evidence` | 公共库正常工作，但没有匹配的有效标准证据 | 不允许 |
| `unavailable` | RAGFlow、网络或公共库账号不可用 | 不允许 |

## 有效证据判定

一条公共标准只有同时满足以下条件才算有效：

1. 存在 `standard_id` 和 `standard_version`；
2. 标准正文非空；
3. 匹配当前请求的 `paper_type`、`rule_type`、`venue_code` 和 `dimension`；
4. 检索或重排分数达到当前检索配置的最低阈值。

低于阈值的片段、缺少标准元数据的片段、其他期刊或论文类型的片段，都不能作为有效公共标准证据。

## 无证据时的返回

公共库服务正常但有效结果为零时，返回成功结果，并设置：

```json
{
  "source_scope": "public_standard",
  "status": "no_evidence",
  "items": [],
  "warning_code": "PUBLIC_KB_NO_EVIDENCE",
  "score_allowed": false,
  "degraded": true
}
```

用户可见提示：

> 公共评审库未检索到与当前问题匹配的有效标准证据，无法依据公共标准作出确定评价或评分。当前只能基于论文原文进行有限分析。

评价流程可以继续，但必须标注“仅基于论文证据”；评分流程不得生成确定的标准化分数。`PUBLIC_KB_UNAVAILABLE` 仅用于服务不可用，不能代替 `PUBLIC_KB_NO_EVIDENCE`。

