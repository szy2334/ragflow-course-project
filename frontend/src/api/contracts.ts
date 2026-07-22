/**
 * Minimal contract surface used before `npm run api:generate` is available.
 * It mirrors the design document's snake_case names; generated OpenAPI types
 * should replace these imports once the backend publishes /api/v1/openapi.json.
 */
export type TaskStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type PaperStatus = 'uploaded' | 'mineru_parsing' | 'ocr_processing' | 'cleaning' | 'quality_check' | 'understanding' | 'indexing' | 'ready' | 'failed'
export type IndexStatus = 'not_indexed' | 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'stale'
export type ClaimVerdict = 'supported' | 'refuted' | 'insufficient_evidence' | 'conflicting_evidence'
export type RouteType = 'fact' | 'explain' | 'follow_up' | 'out_of_scope'
export type FeedbackType = 'like' | 'dislike' | 'issue'
export type AgentName = 'controller' | 'intent_router' | 'paper_understanding' | 'answer_generator' | 'synthesis'
export type EventType = 'status' | 'delta' | 'citation' | 'review_summary' | 'unit_started' | 'unit_progress' | 'unit_validated' | 'unit_unverifiable' | 'unit_failed' | 'unit_skipped' | 'synthesis_started' | 'synthesis_completed' | 'final' | 'error'

export interface ApiResponse<T> {
  code: string
  message: string
  data?: T
  details?: Record<string, unknown>
  request_id: string
  timestamp: string
}

export interface PageData<T> { page: number; page_size: number; total: number; items: T[] }
export interface TokenView { access_token: string; token_type: string; access_expires_at: string; refresh_expires_at: string }
export interface UserView { user_id: string; email: string; display_name: string; role: 'user' | 'admin'; is_active: boolean; created_at: string }
export interface AuthView { user: UserView; token: TokenView }

export interface TaskAccepted { task_id: string; message_id?: string | null; status: TaskStatus; status_url: string; stream_url: string | null; resource_id: string | null }
export interface TaskView {
  task_id: string; task_type: string; status: TaskStatus; progress: number; stage: string; resource_id: string | null
  result: Record<string, unknown> | null; error: { code: string; message: string; details?: Record<string, unknown> } | null
  created_at: string; started_at: string | null; completed_at: string | null
}

export interface PaperFailure {
  stage: Exclude<PaperStatus, 'uploaded' | 'ready' | 'failed'>
  error_code: string
  message: string
  retryable: boolean
}

export interface PaperView {
  paper_id: string; owner_id: string; title: string; authors: string[]; abstract: string | null; language: string | null
  publication_year: number | null; doi: string | null; file_name: string; file_size_bytes: number; page_count: number | null
  status: PaperStatus; parse_progress: number; index_status: IndexStatus; quality_status?: 'pending' | 'ready' | 'failed' | null; understanding: PaperUnderstanding | null; summary_markdown?: string | null; summary_status?: 'pending' | 'ready' | 'unavailable' | 'failed'; failure?: PaperFailure | null; active_index_version: number | null
  created_at: string; updated_at: string
}
export interface PaperUnderstanding {
  status?: 'ready' | 'unavailable'
  answerable?: boolean
  paper_summary?: string | null
  missing_information?: string[]
  facts?: Array<{ claim: string; evidence_ids: string[]; evidence_status: 'explicit' | 'directly_inferred' | 'missing'; confidence: number }>
  reason?: string
  message?: string
}
export interface PaperUploadItem { paper_id: string; file_name: string; status: PaperStatus; task_id: string | null; duplicate: boolean }
export interface PaperUploadBatchView { items: PaperUploadItem[] }
export interface PaperSectionView { section_id: string; paper_id: string; parent_section_id: string | null; section_title: string; section_level: number; section_order: number; page_start: number; page_end: number; text: string; content_type: string; content_role: string }
export interface ChatSessionView { session_id: string; user_id: string; title: string; paper_ids: string[]; last_message_at: string | null; created_at: string }
export interface ChatMessageView { message_id: string; session_id: string; role: 'user' | 'assistant' | 'system'; content: string; task_id: string | null; status: TaskStatus | null; confidence: number | null; answer?: AnswerView | null; created_at: string }

export interface EvidenceItem {
  evidence_id: string; source_type: 'paper'; paper_id: string; document_id: string; chunk_id: string; section_title: string | null; page_number: number | null
  quote: string; retrieval_score: number; source_uri: string
  content_type: 'text' | 'figure' | 'table' | 'figure_caption' | 'formula' | 'metadata' | 'reference'; content_role?: string | null; object_id?: string | null; parent_chunk_id?: string | null; metadata?: Record<string, unknown>
}
export interface Claim { claim_id: string; text: string; type?: 'positive' | 'negative'; verdict: ClaimVerdict; confidence: number; evidence_ids: string[]; reason: string }
export interface AgentMetrics { latency_ms: number; input_tokens: number; output_tokens: number; model_config_id: string | null; retry_count: number; estimated_cost?: string | null; currency?: string | null }
export interface AgentResult { agent_name: AgentName; status: TaskStatus; summary: string; claims: Claim[]; evidence_ids: string[]; confidence: number; warnings: string[]; metrics: AgentMetrics }
export interface AnswerView { message_id: string; session_id: string; task_id: string; route_type?: RouteType; answer: string; claims: Claim[]; evidences: EvidenceItem[]; confidence: number; evidence_sufficient?: boolean; evidence_gap_reason?: string | null; is_refusal: boolean; refusal_reason: string | null; warnings: string[]; completed_at: string }
export interface WorkflowRunView { workflow_run_id: string; task_id: string; session_id: string | null; task_type: string; status: TaskStatus; planned_agents: AgentName[]; confidence: number | null; started_at: string | null; completed_at: string | null; metrics: AgentMetrics }
export interface AnswerDetailView { answer: AnswerView; workflow_run: WorkflowRunView; agent_results: AgentResult[] }

export interface StreamEvent {
  event_id: string; event_type: EventType; task_id: string; message_id?: string | null; session_id: string | null; agent_name: AgentName | null; sequence: number; timestamp: string
  data: Record<string, unknown>
}
export interface ReadingReportView { report_id: string; user_id: string; title: string; paper_ids: string[]; status: TaskStatus; content_markdown: string | null; claims: Claim[]; evidence_ids: string[]; created_at: string; completed_at: string | null }
export interface FormatRule { rule_id: string; title: string; description: string }
export interface FormatProfileView { format_profile_id: string; profile_key: string; name: string; version: string; venue_id: string; description: string | null; allowed_submission_modes: string[]; is_active: boolean; created_at: string; updated_at: string }
export interface AdminFormatProfileView extends FormatProfileView {
  ragflow_dataset_id: string
  retrieval_query: string
  shared_document_id: string
  mode_document_mapping: Record<string, string>
  rule_manifest: Array<Record<string, unknown>>
  configuration_issues: string[]
}
export interface FormatProfileCreateInput {
  profile_key: string
  name: string
  version: string
  description?: string | null
  ragflow_dataset_id: string
  retrieval_query: string
  venue_id?: string | null
  allowed_submission_modes: string[]
  shared_document_id: string
  mode_document_mapping: Record<string, string>
  rules: Array<Record<string, unknown>>
  is_active: boolean
}
export interface FormatReviewItemView { unit_id?: string | null; unit_position?: number | null; source_stage?: string; rule_id: string; rule_title: string; category: string; aspect: string; result: 'compliant' | 'non_compliant' | 'unverifiable' | 'not_applicable'; severity: 'info' | 'low' | 'medium' | 'high'; evidence_status: 'complete' | 'incomplete'; finding: string; suggestion: string | null; page_numbers: number[]; paper_evidences: Array<Record<string, unknown>>; standard_evidences: Array<Record<string, unknown>>; annotation: Record<string, unknown> }
export interface FormatReviewUnitView { unit_id: string; unit_position: number; unit_kind: string; title: string; page_range: number[]; status: string; expected_rule_ids: string[]; retrieved_rule_ids: string[]; not_applicable_rule_ids: Array<Record<string, unknown>>; coverage: { expected_rule_ids?: string[]; retrieved_rule_ids?: string[]; missing_rule_ids?: string[]; complete?: boolean }; unit_cycle_count: number; retry_budget_remaining: number; last_retry_reason: string | null; event_sequence: number; findings: Array<Record<string, unknown>> }
export interface FormatReviewView { format_review_id: string; paper_id: string; format_profile: Pick<FormatProfileView, 'format_profile_id' | 'profile_key' | 'name' | 'version'>; submission_mode: string; selected_rule_ids: string[]; status: TaskStatus; summary_markdown: string | null; coverage_report: { covered_categories?: string[]; missing_categories?: string[]; missing_rule_ids?: string[]; expected_rule_ids?: string[]; unallocated_rule_ids?: string[] }; synthesis_status: string; units: FormatReviewUnitView[]; annotations: Record<string, unknown>; items: FormatReviewItemView[]; error: { code: string; message: string } | null; created_at: string; completed_at: string | null }
export interface MetricsOverviewView { request_count: number; question_count: number; token_input: number; token_output: number; estimated_cost: string; latency_p50_ms: number; latency_p95_ms: number; error_rate: number; retrieval_metrics: Record<string, number>; workflow_metrics: Record<string, number>; time_range: Record<string, string> }

export type AdminRecord = Record<string, unknown>
export interface ApiErrorBody { code?: string; message?: string; details?: Record<string, unknown>; request_id?: string }
