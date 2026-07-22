import type { PaperFailure, PaperStatus, PaperUnderstanding, PaperView } from '@/api/contracts'

export const ingestionStages = [
  { key: 'uploaded', label: '接收' },
  { key: 'mineru_parsing', label: '版面解析' },
  { key: 'ocr_processing', label: '内容补全' },
  { key: 'cleaning', label: '二次清洗' },
  { key: 'quality_check', label: '入库校验' },
  { key: 'understanding', label: '论文理解' },
  { key: 'ready', label: '可阅读' },
] as const

type IngestionStageKey = (typeof ingestionStages)[number]['key']

const stageLabels: Record<IngestionStageKey, string> = Object.fromEntries(
  ingestionStages.map((stage) => [stage.key, stage.label]),
) as Record<IngestionStageKey, string>

export function ingestionStageKey(status: PaperStatus, failure?: PaperFailure | null): IngestionStageKey {
  const value = status === 'failed' ? failure?.stage : status
  if (value === 'indexing') return 'quality_check'
  return value && value in stageLabels ? value as IngestionStageKey : 'uploaded'
}

export function ingestionStageLabel(status: PaperStatus, failure?: PaperFailure | null) {
  return stageLabels[ingestionStageKey(status, failure)]
}

export function isUnderstandingUnavailable(understanding?: PaperUnderstanding | null) {
  return understanding?.status === 'unavailable'
}

export function ingestionFailureMessage(failure?: PaperFailure | null) {
  if (!failure) return '论文处理未完成，请查看任务状态后重试。'
  if (failure.error_code === 'PAPER_INGEST_FAILED') {
    return `处理在“${ingestionStageLabel('failed', failure)}”阶段意外中断，系统未返回更具体的诊断信息。请从该阶段重试；若仍失败，请提供下方错误编号。`
  }
  return failure.message || `“${ingestionStageLabel('failed', failure)}”阶段未完成。`
}

export function paperOverview(paper: PaperView) {
  if (paper.status === 'failed') return ingestionFailureMessage(paper.failure)
  const summary = paper.summary_markdown || paper.understanding?.paper_summary
  if (summary) return markdownPreview(summary)
  if (isUnderstandingUnavailable(paper.understanding)) {
    return paper.understanding?.message || '论文已完成结构化入库，可以检索原文和章节；模型理解与摘要暂不可用。'
  }
  if (paper.status === 'ready') return '论文已完成结构化入库，可以开始阅读、检索、总结或发起格式审查。'
  return '论文正在完成结构化入库，完成后即可开始阅读。'
}

function markdownPreview(markdown: string, limit = 260) {
  const plain = markdown
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
  return plain.length > limit ? `${plain.slice(0, limit - 1).trimEnd()}…` : plain
}
