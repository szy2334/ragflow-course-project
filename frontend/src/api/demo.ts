import { getAccessToken } from './http'
import type { AdminRecord, AnswerDetailView, AnswerView, AuthView, ChatMessageView, ChatSessionView, Claim, EvidenceItem, FormatProfileView, FormatReviewView, MetricsOverviewView, PageData, PaperSectionView, PaperUploadBatchView, PaperView, ReadingReportView, TaskAccepted, TaskView, UserView } from './contracts'

export const DEMO_ACCOUNT = { email: 'demo@zhiyue.local', password: 'Demo@2026' }
const DEMO_TOKEN = 'zhiyue_demo_access_token'
const now = '2026-07-17T08:00:00Z'

const user: UserView = { user_id: '11111111-1111-4111-8111-111111111111', email: DEMO_ACCOUNT.email, display_name: '演示研究员', role: 'user', is_active: true, created_at: now }
const understanding = { answerable: true, paper_summary: '本文将检索、分析和证据核验组合为可审计的科研论文阅读流程。', missing_information: [], facts: [{ claim: '主要实验报告 Evidence F1 提升 4.2 个点。', evidence_ids: ['P1'], evidence_status: 'explicit' as const, confidence: 0.88 }] }
const papers: PaperView[] = [
  { paper_id: '22222222-2222-4222-8222-222222222222', owner_id: user.user_id, title: 'Evidence-Aware Scientific Reading', authors: ['Lin Chen', 'Ming Zhao'], abstract: '本文提出以证据溯源为核心的科研论文阅读流程，将检索、智能体分析和结论核验合并为可审计工作流。', language: 'en', publication_year: 2026, doi: '10.1000/example.2026.001', file_name: 'evidence-aware-reading.pdf', file_size_bytes: 2_438_901, page_count: 12, status: 'ready', parse_progress: 1, index_status: 'not_indexed', understanding, active_index_version: null, created_at: now, updated_at: now },
  { paper_id: '23232323-2323-4232-8232-232323232323', owner_id: user.user_id, title: 'Multi-Agent Retrieval for Academic Question Answering', authors: ['Yue Wang', 'Hao Li'], abstract: '研究多智能体协同检索在学术问答中的分工与冲突消解机制。', language: 'en', publication_year: 2025, doi: null, file_name: 'multi-agent-retrieval.pdf', file_size_bytes: 1_954_782, page_count: 9, status: 'ready', parse_progress: 1, index_status: 'not_indexed', understanding, active_index_version: null, created_at: now, updated_at: now },
]
const sections: PaperSectionView[] = [
  { section_id: 's-1', paper_id: papers[0].paper_id, parent_section_id: null, section_title: '1 Introduction', section_level: 1, section_order: 1, page_start: 1, page_end: 2, text: 'Scientific reading requires traceable evidence.', content_type: 'text', content_role: 'section_parent' },
  { section_id: 's-2', paper_id: papers[0].paper_id, parent_section_id: null, section_title: '3 Method', section_level: 1, section_order: 2, page_start: 3, page_end: 6, text: 'The workflow coordinates retrieval and verification agents.', content_type: 'text', content_role: 'paragraph' },
  { section_id: 's-3', paper_id: papers[0].paper_id, parent_section_id: null, section_title: '4 Experiments', section_level: 1, section_order: 3, page_start: 7, page_end: 10, text: 'Evidence F1 improved by 4.2 points.', content_type: 'table', content_role: 'table_overview' },
]
const session: ChatSessionView = { session_id: '44444444-4444-4444-8444-444444444444', user_id: user.user_id, title: 'Evidence-Aware Scientific Reading', paper_ids: [papers[0].paper_id], last_message_at: now, created_at: now }
const evidence: EvidenceItem = { evidence_id: '77777777-7777-4777-8777-777777777777', source_type: 'paper', paper_id: papers[0].paper_id, document_id: `local:${papers[0].paper_id}`, chunk_id: '88888888-8888-4888-8888-888888888888', section_title: '4 Experiments', page_number: 8, quote: 'The proposed method improves evidence F1 by 4.2 points.', retrieval_score: 0.91, source_uri: `paper://${papers[0].paper_id}/page/8#chunk=88888888-8888-4888-8888-888888888888`, content_type: 'text' }
const claim: Claim = { claim_id: '99999999-9999-4999-8999-999999999999', text: '该方法在主要数据集上的 Evidence F1 提升 4.2 个点。', verdict: 'supported', confidence: 0.88, evidence_ids: [evidence.evidence_id], reason: '实验章节原文直接报告了该数值。' }
const answer: AnswerView = { message_id: '56565656-5656-4656-8656-565656565656', session_id: session.session_id, task_id: '33333333-3333-4333-8333-333333333333', route_type: 'fact', answer: '## 核心结论\n\n该方法通过把检索、分析和证据核验放入同一工作流，减少了不可追溯的结论。\n\n在主要数据集上，Evidence F1 提升 **4.2 个点**。\n\n该结论已由实验章节原文支持。', claims: [claim], evidences: [evidence], confidence: 0.84, is_refusal: false, refusal_reason: null, warnings: [], completed_at: '2026-07-17T08:20:30Z' }
const accepted = (task_id: string, resource_id: string | null = null): TaskAccepted => ({ task_id, message_id: resource_id, status: 'pending', status_url: `/api/v1/tasks/${task_id}`, stream_url: resource_id ? `/api/v1/messages/${resource_id}/events` : null, resource_id })
const task = (task_id: string, resource_id: string | null = null): TaskView => ({ task_id, task_type: 'paper_qa', status: 'succeeded', progress: 1, stage: 'completed', resource_id, result: null, error: null, created_at: now, started_at: now, completed_at: now })
const page = <T>(items: T[]): PageData<T> => ({ page: 1, page_size: 100, total: items.length, items })
const neuripsFormatProfile: FormatProfileView = { format_profile_id: 'bd7aa26a-df6f-4314-8752-cfcbed6ea096', profile_key: 'neurips_2026', name: 'NeurIPS 2026 投稿格式', version: '2026.1', description: '对应 RAGFlow 中的 NeurIPS 2026 Main Track 格式规范库。', rules: [{ rule_id: 'NEURIPS2026-FORMAT-ORDER-001', title: 'PDF内容建议顺序', description: '主PDF应按正文、参考文献、附录支持性文本和论文检查清单的顺序组织。' }, { rule_id: 'NEURIPS2026-FORMAT-TEMPLATE-001', title: '使用当年LaTeX模板', description: '投稿应使用NeurIPS 2026年度LaTeX样式文件。' }, { rule_id: 'NEURIPS2026-FORMAT-STYLE-001', title: '不得通过边距或字体规避限制', description: '不得缩短页边距、减小字体或以其他方式规避版式或页数限制。' }], is_active: true, created_at: now, updated_at: now }
const degreeFormatProfile: FormatProfileView = { format_profile_id: 'f9f4d714-557e-41e3-8edf-9bc3c295ec98', profile_key: 'degree_thesis_2026', name: '学位论文格式（2026）', version: '2026.1', description: '对应 RAGFlow 中的学位论文格式检查（2026）知识库。', rules: [{ rule_id: 'DEGREE-FORMAT-LAYOUT-002', title: '中文段落缩进两个汉字', description: '中文正文段落开头应统一缩进两个汉字。' }, { rule_id: 'DEGREE-FORMAT-EXPR-002', title: '控制句子长度', description: '应避免单句过长导致逻辑关系不清。' }, { rule_id: 'DEGREE-FORMAT-FORMULA-003', title: '公式符号全文统一', description: '公式变量名称、符号和字体样式应全文统一。' }], is_active: true, created_at: now, updated_at: now }
const formatReview = (): FormatReviewView => ({ format_review_id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', paper_id: papers[0].paper_id, format_profile: { format_profile_id: neuripsFormatProfile.format_profile_id, profile_key: neuripsFormatProfile.profile_key, name: neuripsFormatProfile.name, version: neuripsFormatProfile.version }, selected_rule_ids: neuripsFormatProfile.rules.map((item) => item.rule_id), status: 'succeeded', summary_markdown: '## 格式审查摘要\n\n已基于所选规范完成结构化核对；无法由文本证据确定的项目已标记为人工核对。', items: [{ rule_id: 'NEURIPS2026-FORMAT-ORDER-001', rule_title: 'PDF内容建议顺序', result: 'compliant', severity: 'info', finding: '解析结果可识别正文与参考文献结构。', suggestion: null, page_numbers: [1], paper_evidences: [], standard_evidences: [] }, { rule_id: 'NEURIPS2026-FORMAT-TEMPLATE-001', rule_title: '使用当年LaTeX模板', result: 'needs_manual_check', severity: 'medium', finding: '当前演示解析结果不包含可核验的LaTeX模板信息。', suggestion: '请人工确认源文件及模板版本。', page_numbers: [], paper_evidences: [], standard_evidences: [] }], error: null, created_at: now, completed_at: now })

export const demo = {
  active: () => import.meta.env.DEV && getAccessToken() === DEMO_TOKEN,
  login: (email: string, password: string): AuthView | null => {
    if (!import.meta.env.DEV || email !== DEMO_ACCOUNT.email || password !== DEMO_ACCOUNT.password) return null
    return { user, token: { access_token: DEMO_TOKEN, token_type: 'bearer', access_expires_at: '2026-07-17T20:00:00Z', refresh_expires_at: '2026-07-24T08:00:00Z' } }
  },
  user: () => user,
  papers: () => page(papers),
  paper: (id: string) => papers.find((item) => item.paper_id === id) ?? papers[0],
  sections: (id: string) => page(sections.filter((item) => item.paper_id === id)),
  upload: () => ({ items: [{ paper_id: papers[0].paper_id, file_name: papers[0].file_name, status: 'ready' as const, task_id: 'demo-upload-task' }] }) as PaperUploadBatchView,
  task,
  sessions: () => page([session]),
  session: () => session,
  messages: () => page<ChatMessageView>([
    { message_id: 'user-message-1', session_id: session.session_id, role: 'user', content: '这个方法的主要实验结果是什么？', task_id: null, status: 'succeeded', confidence: null, created_at: now },
    { message_id: answer.message_id, session_id: session.session_id, role: 'assistant', content: answer.answer, task_id: answer.task_id, status: 'succeeded', confidence: answer.confidence, created_at: answer.completed_at },
  ]),
  accepted,
  answerDetail: (): AnswerDetailView => ({ answer, workflow_run: { workflow_run_id: '66666666-6666-4666-8666-666666666666', task_id: answer.task_id, session_id: session.session_id, task_type: 'reading_workflow', status: 'succeeded', planned_agents: ['controller', 'paper_understanding', 'synthesis'], confidence: answer.confidence, started_at: now, completed_at: answer.completed_at, metrics: { latency_ms: 30000, input_tokens: 2400, output_tokens: 620, model_config_id: null, retry_count: 0 } }, agent_results: [] }),
  report: (id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'): ReadingReportView => ({ report_id: id, user_id: user.user_id, title: 'Evidence-Aware Scientific Reading 阅读报告', paper_ids: [papers[0].paper_id], status: 'succeeded', content_markdown: '# 阅读报告\n\n## 核心结论\n\n该方法在主要数据集上的 Evidence F1 提升 4.2 个点。\n\n## 证据说明\n\n结论对应实验章节的持久化证据，可定位到第 8 页。', claims: [claim], evidence_ids: [evidence.evidence_id], created_at: now, completed_at: now }),
  formatProfiles: () => ({ items: [neuripsFormatProfile, degreeFormatProfile] }),
  formatReview,
  admin: (resource: string): PageData<AdminRecord> => page([{ name: resource === 'model-configs' ? 'primary_generation' : resource === 'datasets' ? 'qasper' : '科研论文库', status: 'ready', version: 1, model_type: 'generation', provider: 'openai_compatible', model_name: 'demo-model', dataset_name: 'qasper', dataset_version: 'official_2026_01', license_name: 'dataset_license', paper_count: 2 }]),
  metrics: (): MetricsOverviewView => ({ request_count: 1280, question_count: 320, token_input: 1200000, token_output: 260000, estimated_cost: '42.38000000', latency_p50_ms: 12500, latency_p95_ms: 48000, error_rate: 0.018, retrieval_metrics: { empty_rate: 0.04, mean_top_score: 0.78 }, workflow_metrics: { success_rate: 0.95, refusal_rate: 0.12 }, time_range: { start_time: '2026-07-16T00:00:00Z', end_time: now, interval: 'hour' } }),
  evidence,
  answer,
}
