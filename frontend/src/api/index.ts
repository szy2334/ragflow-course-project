import type { AxiosRequestConfig } from 'axios'
import { http, newRequestId, request } from './http'
import { demo } from './demo'
import type { AdminRecord, AnswerDetailView, AuthView, ChatMessageView, ChatSessionView, FormatProfileView, FormatReviewView, MetricsOverviewView, PageData, PaperSectionView, PaperUploadBatchView, PaperView, ReadingReportView, TaskAccepted, TaskView, UserView } from './contracts'

function createDemoPaperBlob() {
  const stream = 'BT\n/F1 24 Tf\n72 720 Td\n(Demo paper preview) Tj\nET\n'
  const objects = [
    '1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n',
    '2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n',
    '3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n',
    `4 0 obj\n<< /Length ${stream.length} >>\nstream\n${stream}endstream\nendobj\n`,
    '5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n',
  ]
  const header = '%PDF-1.4\n'
  let offset = header.length
  const offsets = objects.map((object) => {
    const start = offset
    offset += object.length
    return start
  })
  const xref = `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.map((start) => `${String(start).padStart(10, '0')} 00000 n \n`).join('')}`
  const trailer = `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${offset}\n%%EOF\n`
  return new Blob([header, ...objects, xref, trailer], { type: 'application/pdf' })
}

const get = <T>(url: string, params?: Record<string, unknown>) => request<T>({ url, method: 'GET', params })
const post = <T>(url: string, data?: unknown, config?: AxiosRequestConfig) => request<T>({ ...config, url, data, method: 'POST', headers: { ...config?.headers, 'Idempotency-Key': newRequestId() } })
const put = <T>(url: string, data: unknown, config?: AxiosRequestConfig) => request<T>({ ...config, url, data, method: 'PUT', headers: { ...config?.headers, 'Idempotency-Key': newRequestId() } })
const encodeRelativePath = (path: string) => path.split('/').map(encodeURIComponent).join('/')

export const api = {
  register: (data: { email: string; password: string; display_name: string }) => post<AuthView>('/auth/register', data),
  login: (data: { email: string; password: string }) => Promise.resolve(demo.login(data.email, data.password) ?? post<AuthView>('/auth/login', data)),
  me: () => demo.active() ? Promise.resolve(demo.user()) : get<UserView>('/auth/me'),

  listPapers: (params: Record<string, unknown>) => demo.active() ? Promise.resolve(demo.papers()) : get<PageData<PaperView>>('/papers', params),
  getPaper: (paper_id: string) => demo.active() ? Promise.resolve(demo.paper(paper_id)) : get<PaperView>(`/papers/${paper_id}`),
  getPaperFile: (paper_id: string) => demo.active() ? Promise.resolve(createDemoPaperBlob()) : http.get<Blob>(`/papers/${paper_id}/file`, { params: { disposition: 'inline' }, responseType: 'blob' }).then((response) => response.data),
  listReferencePaperRuns: () => get<{ items: Array<{ name: string; file_count: number }> }>('/reference-papers/runs'),
  getReferencePaperFile: (relativePath: string) => http.get<Blob>(`/reference-papers/runs/${encodeRelativePath(relativePath)}`, { responseType: 'blob' }).then((response) => response.data),
  listSections: (paper_id: string, params?: Record<string, unknown>) => demo.active() ? Promise.resolve(demo.sections(paper_id)) : get<PageData<PaperSectionView>>(`/papers/${paper_id}/sections`, params),
  getTask: (task_id: string) => demo.active() ? Promise.resolve(task_id === 'demo-export' ? { ...demo.task(task_id), result: { download_url: 'data:text/plain;charset=utf-8,%E7%9F%A5%E9%98%85%E6%BC%94%E7%A4%BA%E6%8A%A5%E5%91%8A' } } : demo.task(task_id)) : get<TaskView>(`/tasks/${task_id}`),
  uploadPapers: (files: File[]) => {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    return demo.active() ? Promise.resolve(demo.upload()) : post<PaperUploadBatchView>('/papers', form)
  },
  reparsePaper: (paper_id: string, data: { parser_name?: string; force: boolean }) => demo.active() ? Promise.resolve(demo.accepted('demo-reparse', paper_id)) : post<TaskAccepted>(`/papers/${paper_id}/retry`, { ...data, stage: 'mineru_parsing' }),
  retryPaper: (paper_id: string, data: { stage: 'mineru_parsing' | 'ocr_processing' | 'cleaning' | 'quality_check' | 'understanding'; force: boolean }) => demo.active() ? Promise.resolve(demo.accepted('demo-retry', paper_id)) : post<TaskAccepted>(`/papers/${paper_id}/retry`, data),
  deletePaper: (paper_id: string) => demo.active() ? Promise.resolve({ paper_id, deleted_at: new Date().toISOString(), cleanup_task_id: 'demo-cleanup' }) : request<{ paper_id: string; deleted_at: string; cleanup_task_id: string } | Record<string, never>>({ url: `/papers/${paper_id}`, method: 'DELETE', headers: { 'Idempotency-Key': newRequestId() } }),

  listSessions: (params?: Record<string, unknown>) => demo.active() ? Promise.resolve(demo.sessions()) : get<PageData<ChatSessionView>>('/sessions', params),
  createSession: (data: { title?: string; paper_ids: string[] }) => demo.active() ? Promise.resolve(demo.session()) : post<ChatSessionView>('/sessions', data),
  listMessages: (session_id: string, params?: Record<string, unknown>) => demo.active() ? Promise.resolve(demo.messages()) : get<PageData<ChatMessageView>>(`/sessions/${session_id}/messages`, params),
  askQuestion: (session_id: string, data: { question: string; paper_ids?: string[] }) => demo.active() ? Promise.resolve(demo.accepted('demo-workflow', demo.answer.message_id)) : post<TaskAccepted>(`/sessions/${session_id}/messages`, { ...data, stream: true }),
  cancelWorkflow: (message_id: string, reason?: string) => demo.active() ? Promise.resolve({ ...demo.task(message_id), status: 'cancelled' as const }) : post<TaskView>(`/messages/${message_id}/cancel`, { reason }),
  getAnswerDetail: (message_id: string) => demo.active() ? Promise.resolve(demo.answerDetail()) : get<AnswerDetailView>(`/messages/${message_id}/details`),
  sendFeedback: (message_id: string, data: { feedback_type: 'like' | 'dislike' | 'issue'; reason?: string; tags: string[] }) => demo.active() ? Promise.resolve({ feedback_id: 'demo-feedback', message_id, feedback_type: data.feedback_type }) : post(`/messages/${message_id}/feedback`, data),

  createAnalysis: (paper_id: string, kind: 'summary' | 'method' | 'experiment', data: { question?: string; force_refresh: boolean }) => demo.active() ? Promise.resolve(demo.accepted('demo-analysis', paper_id)) : post<TaskAccepted>(`/papers/${paper_id}/analyses/${kind}`, data),
  comparePapers: (data: { paper_ids: string[]; dimensions: string[]; question?: string }) => demo.active() ? Promise.resolve(demo.accepted('demo-comparison')) : post<TaskAccepted>('/paper-comparisons', data),
  createReport: (data: { paper_ids: string[]; session_id?: string; template_key: string; title: string }) => demo.active() ? Promise.resolve(demo.accepted('demo-report', 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')) : post<TaskAccepted>('/reading-reports', data),
  getReport: (report_id: string) => demo.active() ? Promise.resolve(demo.report(report_id)) : get<ReadingReportView>(`/reading-reports/${report_id}`),
  exportReport: (report_id: string, format: 'markdown' | 'pdf' | 'docx') => demo.active() ? Promise.resolve(demo.accepted('demo-export')) : post<TaskAccepted>(`/reading-reports/${report_id}/exports`, { format }),

  listFormatProfiles: () => demo.active() ? Promise.resolve(demo.formatProfiles()) : get<{ items: FormatProfileView[] }>('/format-profiles'),
  createFormatReview: (data: { paper_id: string; format_profile_id: string; rule_ids: string[] }) => demo.active() ? Promise.resolve(demo.accepted('demo-format-review', 'cccccccc-cccc-4ccc-8ccc-cccccccccccc')) : post<TaskAccepted>('/format-reviews', data),
  getFormatReview: (format_review_id: string) => demo.active() ? Promise.resolve(demo.formatReview()) : get<FormatReviewView>(`/format-reviews/${format_review_id}`),

  listAdmin: (resource: 'model-configs' | 'prompt-templates' | 'retrieval-configs' | 'knowledge-bases' | 'datasets', params?: Record<string, unknown>) => demo.active() ? Promise.resolve(demo.admin(resource)) : get<PageData<AdminRecord>>(`/admin/${resource}`, params),
  upsertAdmin: (resource: 'model-configs' | 'prompt-templates' | 'retrieval-configs', id: string, data: AdminRecord, version?: number) => put<AdminRecord>(`/admin/${resource}/${id}`, { value: data }, { headers: { 'If-Match': version ? String(version) : '*' } }),
  getMetrics: (params: Record<string, unknown>) => demo.active() ? Promise.resolve(demo.metrics()) : get<MetricsOverviewView>('/admin/metrics/overview', params),
  getWorkflowTrace: (workflow_run_id: string) => get<AdminRecord>(`/admin/workflow-runs/${workflow_run_id}`, { include_events: true, include_evidences: true }),
  createEvaluation: (data: { dataset_id: string; split: string; experiment_type: string; model_config_id?: string; sample_limit?: number; random_seed?: number }) => demo.active() ? Promise.resolve(demo.accepted('demo-evaluation')) : post<TaskAccepted>('/admin/evaluation-runs', data),
}
