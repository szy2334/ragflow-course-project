<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { AlertTriangle, CheckCircle2, CircleAlert, ClipboardCheck, FileSearch, MapPin, PanelRightClose, PanelRightOpen, Play, Trash2 } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import MarkdownContent from '@/components/MarkdownContent.vue'
import StatusPill from '@/components/StatusPill.vue'
import FormatEvidencePreview from '@/components/FormatEvidencePreview.vue'
import { api } from '@/api'
import { ApiError, getAccessToken } from '@/api/http'
import type { FormatProfileView, FormatReviewHistoryItem, FormatReviewItemView, FormatReviewUnitView, FormatReviewView, StreamEvent } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'
import { formatChinaDateTime } from '@/utils/dateTime'

const route = useRoute()
const router = useRouter()
const workspace = useWorkspaceStore()
const profiles = ref<FormatProfileView[]>([])
const selectedPaperId = ref('')
const selectedProfileId = ref('')
const selectedSubmissionMode = ref('')
const review = ref<FormatReviewView | null>(null)
const reviewHistory = ref<FormatReviewHistoryItem[]>([])
const historyOpen = ref(true)
const deletingHistoryId = ref('')
const loading = ref(true)
const submitting = ref(false)
const error = ref('')
const liveMessageByUnit = ref<Record<string, string>>({})
const previewTarget = ref<{ paperId: string; page: number; bbox: [number, number, number, number]; rotation: number; aspect: string } | null>(null)
let streamAbort: AbortController | null = null

const readyPapers = computed(() => Object.values(workspace.papersById).filter((paper) => paper.status === 'ready'))
const selectedProfile = computed(() => profiles.value.find((profile) => profile.format_profile_id === selectedProfileId.value) ?? null)
const submissionModes = computed(() => selectedProfile.value?.profile_key === 'neurips_2020' ? ['general'] : selectedProfile.value?.allowed_submission_modes ?? [])
const reviewGroups = computed(() => {
  const groups = new Map<string, FormatReviewItemView[]>()
  for (const item of review.value?.items ?? []) groups.set(item.category, [...(groups.get(item.category) ?? []), item])
  return [...groups.entries()].map(([category, items]) => ({ category, items }))
})
const reviewUnits = computed(() => [...(review.value?.units ?? [])].sort((left, right) => left.unit_position - right.unit_position))
const resultCounts = computed(() => {
  const counts = { non_compliant: 0, compliant: 0, unverifiable: 0 }
  for (const item of review.value?.items ?? []) {
    if (item.result in counts) counts[item.result as keyof typeof counts] += 1
  }
  return counts
})

function configureSubmissionMode() {
  const modes = submissionModes.value
  if (!modes.includes(selectedSubmissionMode.value)) selectedSubmissionMode.value = modes[0] ?? ''
}
function modeLabel(mode: string) {
  return mode === 'general' ? '通用' : mode === 'initial_submission' ? '匿名初稿' : mode === 'camera_ready' ? '终稿' : mode.replace(/_/g, ' ')
}
function profileLabel(profile: Pick<FormatProfileView, 'name' | 'profile_key' | 'version'>) {
  if (!/\?{2,}/.test(profile.name)) return profile.name
  const repairedNames: Record<string, string> = {
    neurips_2020: 'NeurIPS 2020 投稿格式',
    neurips_2026: 'NeurIPS 2026 投稿格式',
    degree_thesis_2026: '学位论文格式',
  }
  return repairedNames[profile.profile_key] ?? `${profile.profile_key} 格式规范`
}
function resultLabel(result: FormatReviewItemView['result']) {
  return result === 'compliant' ? '符合' : result === 'non_compliant' ? '不符合' : result === 'not_applicable' ? '不适用' : '无法可靠判断'
}
function evidenceQuote(evidence: Record<string, unknown>) { return String(evidence.quote ?? evidence.text ?? '未提供引用内容') }
function evidencePage(evidence: Record<string, unknown>) {
  const page = evidence.page_number ?? evidence.page
  return typeof page === 'number' ? page : null
}
function annotationFor(item: FormatReviewItemView) {
  const source = Object.keys(item.annotation ?? {}).length ? item.annotation : item.paper_evidences[0]
  const bbox = source?.bbox
  const page = source?.page ?? source?.page_number
  if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every((value) => typeof value === 'number') || typeof page !== 'number') return null
  return { page, bbox: bbox as [number, number, number, number], rotation: typeof source.page_rotation === 'number' ? source.page_rotation : 0 }
}
function openEvidence(item: FormatReviewItemView) {
  const annotation = annotationFor(item)
  if (!annotation || !review.value) return
  previewTarget.value = { paperId: review.value.paper_id, aspect: item.aspect || item.rule_title, ...annotation }
}

function queryValue(value: unknown) {
  return typeof value === 'string' ? value : ''
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const [_, profileResult, historyResult] = await Promise.all([workspace.loadPapers({ status: 'ready' }), api.listFormatProfiles(), api.listFormatReviews()])
    profiles.value = profileResult.items
    reviewHistory.value = historyResult.items
    const reviewId = queryValue(route.query.reviewId)
    if (reviewId) {
      const restoredReview = await api.getFormatReview(reviewId)
      review.value = restoredReview
      selectedPaperId.value = restoredReview.paper_id
      selectedProfileId.value = restoredReview.format_profile.format_profile_id
      selectedSubmissionMode.value = restoredReview.submission_mode
      return
    }
    selectedPaperId.value = queryValue(route.query.paperId) || readyPapers.value[0]?.paper_id || ''
    selectedProfileId.value = profiles.value[0]?.format_profile_id ?? ''
    configureSubmissionMode()
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法加载可用格式规范。'
  } finally { loading.value = false }
}

async function loadReviewHistory() {
  try {
    reviewHistory.value = (await api.listFormatReviews()).items
  } catch {
    // History should not block starting or viewing a review.
  }
}

async function openReviewHistory(item: FormatReviewHistoryItem) {
  try {
    const restoredReview = await api.getFormatReview(item.format_review_id)
    review.value = restoredReview
    selectedPaperId.value = restoredReview.paper_id
    selectedProfileId.value = restoredReview.format_profile.format_profile_id
    selectedSubmissionMode.value = restoredReview.submission_mode
    error.value = ''
    await router.replace({ query: { ...route.query, paperId: restoredReview.paper_id, reviewId: restoredReview.format_review_id } })
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法加载历史格式审查记录。'
  }
}

async function deleteReviewHistory(item: FormatReviewHistoryItem) {
  if (item.status === 'pending' || item.status === 'running') return
  if (!window.confirm(`删除“${profileLabel(item.format_profile)}”的格式审查记录？此操作无法撤销。`)) return
  deletingHistoryId.value = item.format_review_id
  try {
    await api.deleteFormatReview(item.format_review_id)
    reviewHistory.value = reviewHistory.value.filter((entry) => entry.format_review_id !== item.format_review_id)
    if (review.value?.format_review_id === item.format_review_id) {
      review.value = null
      await router.replace({ query: { ...route.query, reviewId: undefined } })
    }
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法删除格式审查记录。'
  } finally {
    deletingHistoryId.value = ''
  }
}

async function startReview() {
  if (!selectedPaperId.value || !selectedProfile.value || !selectedSubmissionMode.value) {
    error.value = '请选择论文、格式规范和投稿模式。'
    return
  }
  submitting.value = true
  error.value = ''
  review.value = null
  liveMessageByUnit.value = {}
  try {
    const task = await api.createFormatReview({
      paper_id: selectedPaperId.value,
      format_profile_id: selectedProfile.value.format_profile_id,
      submission_mode: selectedSubmissionMode.value,
    })
    if (!task.resource_id) throw new Error('服务端未返回格式审查标识。')
    await router.replace({ query: { ...route.query, paperId: selectedPaperId.value, reviewId: task.resource_id } })
    review.value = await api.getFormatReview(task.resource_id)
    const completed = await consumeReviewEvents(task.resource_id, task.stream_url, task.task_id)
    if (!completed) {
      const finished = await workspace.pollTask(task.task_id)
      if (finished?.status !== 'succeeded') throw new Error(finished?.error?.message ?? '格式审查未完成。')
    }
    review.value = await api.getFormatReview(task.resource_id)
  } catch (cause) {
    error.value = cause instanceof ApiError || cause instanceof Error ? cause.message : '无法完成格式审查。'
  } finally {
    submitting.value = false
    void loadReviewHistory()
  }
}

function updateUnitFromEvent(event: StreamEvent) {
  if (!review.value || !event.event_type.startsWith('unit_')) return
  const data = event.data
  const unitId = typeof data.unit_id === 'string' ? data.unit_id : ''
  if (!unitId) return
  const current = review.value.units.find((unit) => unit.unit_id === unitId)
  const pageRange = Array.isArray(data.page_range) ? data.page_range.filter((value): value is number => typeof value === 'number') : current?.page_range ?? []
  const next: FormatReviewUnitView = {
    unit_id: unitId,
    unit_position: typeof data.unit_position === 'number' ? data.unit_position : current?.unit_position ?? Number.MAX_SAFE_INTEGER,
    unit_kind: typeof data.unit_kind === 'string' ? data.unit_kind : current?.unit_kind ?? 'body_section',
    title: typeof data.title === 'string' ? data.title : current?.title ?? '审查块',
    page_range: pageRange,
    status: typeof data.status === 'string' ? data.status : current?.status ?? 'running',
    expected_rule_ids: current?.expected_rule_ids ?? [],
    retrieved_rule_ids: current?.retrieved_rule_ids ?? [],
    not_applicable_rule_ids: current?.not_applicable_rule_ids ?? [],
    coverage: typeof data.coverage === 'object' && data.coverage !== null ? data.coverage as FormatReviewUnitView['coverage'] : current?.coverage ?? {},
    unit_cycle_count: typeof data.unit_cycle_count === 'number' ? data.unit_cycle_count : current?.unit_cycle_count ?? 0,
    retry_budget_remaining: typeof data.retry_budget_remaining === 'number' ? data.retry_budget_remaining : current?.retry_budget_remaining ?? 1,
    last_retry_reason: typeof data.last_retry_reason === 'string' ? data.last_retry_reason : current?.last_retry_reason ?? null,
    event_sequence: typeof data.event_sequence === 'number' ? data.event_sequence : event.sequence,
    findings: Array.isArray(data.findings) ? data.findings as Array<Record<string, unknown>> : current?.findings ?? [],
  }
  review.value = { ...review.value, units: [...review.value.units.filter((unit) => unit.unit_id !== unitId), next].sort((left, right) => left.unit_position - right.unit_position) }
  if (typeof data.message === 'string') liveMessageByUnit.value = { ...liveMessageByUnit.value, [unitId]: data.message }
}

async function consumeReviewEvents(reviewId: string, streamUrl: string | null, taskId: string) {
  void taskId
  if (!streamUrl) return false
  streamAbort?.abort()
  streamAbort = new AbortController()
  let afterSequence = 0
  let reconnects = 0
  while (reconnects < 3 && !streamAbort.signal.aborted) {
    try {
      const url = `${streamUrl}${streamUrl.includes('?') ? '&' : '?'}after_sequence=${afterSequence}`
      const response = await fetch(url, { headers: { Authorization: `Bearer ${getAccessToken() ?? ''}` }, credentials: 'include', signal: streamAbort.signal })
      if (!response.ok || !response.body) throw new Error('SSE 连接不可用')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (!streamAbort.signal.aborted) {
        const { done, value } = await reader.read()
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
        const chunks = buffer.split(/\r?\n\r?\n/)
        buffer = chunks.pop() ?? ''
        for (const chunk of chunks) {
          const payload = chunk.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('')
          if (!payload) continue
          const event = JSON.parse(payload) as StreamEvent
          if (event.sequence <= afterSequence) continue
          afterSequence = event.sequence
          updateUnitFromEvent(event)
          if (event.event_type.startsWith('unit_') || event.event_type === 'synthesis_completed') review.value = await api.getFormatReview(reviewId)
          if (event.event_type === 'final') return true
          if (event.event_type === 'error') throw new Error(String(event.data.message ?? '格式审查任务失败。'))
        }
        if (done) break
      }
      if (streamAbort.signal.aborted) return false
    } catch {
      if (streamAbort.signal.aborted) return false
      reconnects += 1
      if (reconnects >= 3) break
      await new Promise((resolve) => window.setTimeout(resolve, reconnects * 700))
      continue
    }
    reconnects += 1
  }
  return false
}

watch(selectedProfileId, configureSubmissionMode)
onMounted(load)
onBeforeUnmount(() => streamAbort?.abort())
</script>

<template>
  <section class="page review-page">
    <PageHeader eyebrow="格式合规审查" title="审查论文格式" description="基于所选投稿场所、规范版本和投稿模式完成论文版面综合检查。">
      <button class="secondary-button" @click="router.push('/papers')"><FileSearch :size="18" />返回阅读</button>
    </PageHeader>

    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="skeleton-list"><div v-for="item in 3" :key="item" class="skeleton-row" /></div>
    <template v-else>
      <div class="review-layout" :class="{ 'history-collapsed': !historyOpen }">
        <div class="review-main">
      <section class="review-config card-surface">
        <div class="config-step"><span>01</span><div><h2>选择论文</h2><p>仅可审查已完成解析与理解的本地 PDF。</p></div></div>
        <select v-model="selectedPaperId" class="review-select" aria-label="选择论文"><option v-for="paper in readyPapers" :key="paper.paper_id" :value="paper.paper_id">{{ paper.title }}</option></select>

        <div class="config-step"><span>02</span><div><h2>选择格式规范</h2><p>规范数据集和规则文档由服务端固定，不会暴露给客户端。</p></div></div>
        <select v-model="selectedProfileId" class="review-select" aria-label="选择格式规范"><option v-for="profile in profiles" :key="profile.format_profile_id" :value="profile.format_profile_id">{{ profileLabel(profile) }} · {{ profile.version }}</option></select>
        <p v-if="selectedProfile?.description" class="profile-description">{{ selectedProfile.description }}</p>

        <div class="config-step"><span>03</span><div><h2>选择投稿模式</h2><p>{{ selectedProfile?.profile_key === 'neurips_2020' ? 'NeurIPS 2020 规则清单适用于通用模式。' : '系统将检索通用规则和当前投稿模式规则，并完成完整性检查。' }}</p></div></div>
        <select v-model="selectedSubmissionMode" class="review-select" aria-label="选择投稿模式" :disabled="selectedProfile?.profile_key === 'neurips_2020'"><option v-for="mode in submissionModes" :key="mode" :value="mode">{{ modeLabel(mode) }}</option></select>
        <button class="primary-button full-width" :disabled="submitting || !readyPapers.length || !profiles.length" @click="startReview"><Play :size="18" />{{ submitting ? '正在分块审查…' : '开始综合审查' }}</button>
      </section>

      <section v-if="review" class="review-result card-surface">
        <div class="result-title"><div><p class="eyebrow">{{ profileLabel(review.format_profile) }} · {{ review.format_profile.version }} · {{ modeLabel(review.submission_mode) }}</p><h2>格式审查结果</h2></div><StatusPill :status="review.status" /></div>
        <div class="result-summary"><span><CircleAlert :size="17" />{{ resultCounts.non_compliant }} 项不符合</span><span><CheckCircle2 :size="17" />{{ resultCounts.compliant }} 项符合</span><span><AlertTriangle :size="17" />{{ resultCounts.unverifiable }} 项无法可靠判断</span></div>
        <MarkdownContent :content="review.summary_markdown || '审查块正在执行，阶段结果会实时出现。'" />
        <p v-if="review.coverage_report.missing_categories?.length" class="coverage-warning"><AlertTriangle :size="16" />未完整检索的规范类别：{{ review.coverage_report.missing_categories.join('、') }}</p>

        <section v-if="reviewUnits.length" class="review-units" aria-label="审查块实时进度">
          <div class="finding-group-heading"><h3>审查块进度</h3><span>{{ reviewUnits.length }} 个块 · 汇总{{ review.synthesis_status === 'completed' ? '完成' : '等待中' }}</span></div>
          <article v-for="unit in reviewUnits" :key="unit.unit_id" class="review-unit" :class="unit.status">
            <div><strong>{{ unit.title }}</strong><span>{{ unit.unit_kind }} · 第 {{ unit.page_range[0] ?? '?' }}–{{ unit.page_range[1] ?? '?' }} 页</span></div>
            <span>{{ unit.status === 'validated' ? '已核验' : unit.status === 'unverifiable' ? '无法可靠判断' : unit.status === 'failed' ? '失败' : '审查中' }}</span>
            <p>{{ liveMessageByUnit[unit.unit_id] || `规则 ${unit.retrieved_rule_ids.length}/${unit.expected_rule_ids.length} 已装配` }}</p>
            <p v-if="unit.last_retry_reason" class="unverifiable-note">已使用重试：{{ unit.last_retry_reason }}</p>
          </article>
        </section>

        <section v-for="group in reviewGroups" :key="group.category" class="finding-group">
          <div class="finding-group-heading"><h3>{{ group.category }}</h3><span>{{ group.items.length }} 项</span></div>
          <article v-for="item in group.items" :key="item.rule_id" class="format-review-item" :class="item.result">
            <div class="finding-title"><ClipboardCheck :size="18" /><strong>{{ item.aspect || item.rule_title }}</strong><span>{{ resultLabel(item.result) }}</span></div>
            <p>{{ item.finding }}</p>
            <p v-if="item.suggestion" class="suggestion">{{ item.suggestion }}</p>
            <p v-if="item.result === 'unverifiable'" class="unverifiable-note">{{ item.evidence_status === 'incomplete' ? '规范或论文证据不足，未作合规推断。' : '无法从现有解析产物可靠判定。' }}</p>
            <footer><span v-if="item.page_numbers.length">涉及页码：{{ item.page_numbers.join('、') }}</span><span>论文证据 {{ item.paper_evidences.length }} 条 · 规范证据 {{ item.standard_evidences.length }} 条</span><button v-if="annotationFor(item)" class="text-button" @click="openEvidence(item)"><MapPin :size="15" />定位原文</button></footer>
            <details class="evidence-details"><summary>证据详情</summary><div class="evidence-columns"><div><h4>投稿场所格式要求</h4><p v-for="(evidence, index) in item.standard_evidences" :key="index">{{ evidenceQuote(evidence) }}</p><p v-if="!item.standard_evidences.length">未提供可引用的规范原文。</p></div><div><h4>论文实际情况</h4><p v-for="(evidence, index) in item.paper_evidences" :key="index">{{ evidencePage(evidence) ? `第 ${evidencePage(evidence)} 页：` : '' }}{{ evidenceQuote(evidence) }}</p><p v-if="!item.paper_evidences.length">未提供可定位的论文证据。</p></div></div></details>
          </article>
        </section>
      </section>
      <div v-else-if="!readyPapers.length || !profiles.length" class="empty-state"><AlertTriangle :size="34" /><h2>{{ !readyPapers.length ? '没有可审查论文' : '没有可用格式规范' }}</h2><p>{{ !readyPapers.length ? '请先完成论文解析与理解。' : '请由管理员创建格式规范并绑定对应的 RAGFlow 知识库。' }}</p></div>
        </div>
        <aside class="review-history" :class="{ open: historyOpen }" aria-label="格式审查历史记录">
          <button class="history-toggle" type="button" :aria-expanded="historyOpen" :aria-label="historyOpen ? '收起格式审查历史' : '展开格式审查历史'" @click="historyOpen = !historyOpen">
            <PanelRightClose v-if="historyOpen" :size="18" /><PanelRightOpen v-else :size="18" />
            <span v-if="historyOpen">格式审查历史</span>
          </button>
          <div v-if="historyOpen" class="history-list">
            <p v-if="!reviewHistory.length" class="history-empty">尚无格式审查记录。</p>
            <article v-for="item in reviewHistory" :key="item.format_review_id" class="history-item" :class="{ active: item.format_review_id === review?.format_review_id }">
              <button class="history-open" type="button" @click="openReviewHistory(item)">
                <span class="history-item-title">{{ profileLabel(item.format_profile) }}</span>
                <span>{{ modeLabel(item.submission_mode) }} · {{ item.status === 'succeeded' ? '已完成' : item.status === 'failed' ? '失败' : '进行中' }}</span>
                <time>{{ formatChinaDateTime(item.created_at) }}</time>
              </button>
              <button class="icon-button history-delete" type="button" :disabled="deletingHistoryId === item.format_review_id || item.status === 'pending' || item.status === 'running'" :title="item.status === 'pending' || item.status === 'running' ? '执行中的审查不能删除' : '删除审查记录'" aria-label="删除审查记录" @click="deleteReviewHistory(item)"><Trash2 :size="15" /></button>
            </article>
          </div>
        </aside>
      </div>
    </template>
    <FormatEvidencePreview v-if="previewTarget" v-bind="previewTarget" @close="previewTarget = null" />
  </section>
</template>

<style scoped>
.review-page, .review-config, .review-result, .finding-group, .review-units { display: grid; gap: 16px; }.review-layout { display: grid; grid-template-columns: minmax(0, 1fr) 286px; gap: 16px; align-items: start; }.review-layout.history-collapsed { grid-template-columns: minmax(0, 1fr) 46px; }.review-main { display: grid; min-width: 0; gap: 16px; }.review-config, .review-result { padding: 22px; }.config-step { display: flex; gap: 12px; align-items: flex-start; }.config-step > span { display: grid; width: 28px; height: 28px; place-items: center; color: white; background: #8b4b17; border-radius: 50%; font-size: 12px; }.config-step h2 { margin: 2px 0 4px; font-size: 16px; }.config-step p, .profile-description { margin: 0; color: var(--ink-soft); font-size: 13px; line-height: 1.5; }.review-select { width: 100%; min-height: 42px; padding: 0 12px; border: 1px solid var(--line); border-radius: 8px; background: white; color: var(--ink); }.result-title, .finding-title, .result-summary, .finding-group-heading, .format-review-item footer, .review-unit, .review-unit > div { display: flex; gap: 10px; align-items: center; }.result-title, .finding-group-heading { justify-content: space-between; }.result-title h2 { margin: 2px 0 0; }.result-summary { flex-wrap: wrap; padding: 12px 0; border-block: 1px solid var(--line); color: var(--ink-soft); font-size: 13px; }.result-summary span { display: inline-flex; gap: 5px; align-items: center; }.coverage-warning, .unverifiable-note { display: flex; gap: 7px; align-items: center; margin: 0; color: #88620b; font-size: 13px; line-height: 1.5; }.finding-group { padding-top: 4px; }.finding-group-heading h3 { margin: 0; font-size: 15px; text-transform: capitalize; }.finding-group-heading span, .review-unit span { color: var(--ink-faint); font-size: 12px; }.review-unit { flex-wrap: wrap; padding: 11px 13px; border-left: 3px solid #b8b1a6; background: #fafaf8; }.review-unit > div { flex: 1; min-width: 220px; }.review-unit > div span { margin-left: auto; }.review-unit > p { flex-basis: 100%; margin: 0; color: var(--ink-soft); font-size: 12px; }.review-unit.validated { border-color: #4a7a57; }.review-unit.unverifiable, .review-unit.failed { border-color: #9c7615; }.format-review-item { display: grid; gap: 9px; padding: 14px; border-left: 3px solid var(--line); background: #fafaf8; }.finding-title strong { flex: 1; }.finding-title span { color: var(--ink-soft); font-size: 13px; }.format-review-item p { margin: 0; line-height: 1.6; }.format-review-item footer { flex-wrap: wrap; color: var(--ink-faint); font-size: 12px; }.format-review-item.non_compliant { border-color: #b24031; }.format-review-item.compliant { border-color: #4a7a57; }.format-review-item.unverifiable { border-color: #9c7615; }.suggestion { color: var(--ink-soft); }.evidence-details { padding-top: 2px; color: var(--ink-soft); font-size: 13px; }.evidence-details summary { cursor: pointer; color: var(--ink); }.evidence-columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 10px; }.evidence-columns h4 { margin: 0 0 6px; color: var(--ink); font-size: 13px; }.evidence-columns p { padding: 8px 0; border-top: 1px solid var(--line); font-size: 12px; }.review-history { position: sticky; top: 82px; min-width: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: white; box-shadow: var(--shadow-sm); }.history-toggle { display: flex; width: 100%; min-height: 43px; gap: 8px; align-items: center; padding: 0 12px; border: 0; color: var(--ink); background: white; font-size: 13px; font-weight: 700; text-align: left; }.history-toggle:hover { color: var(--teal-800); background: var(--teal-100); }.history-list { display: grid; max-height: calc(100dvh - 140px); overflow-y: auto; border-top: 1px solid var(--line); }.history-empty { margin: 0; padding: 15px 12px; color: var(--ink-faint); font-size: 12px; }.history-item { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px; align-items: center; padding: 5px; border-bottom: 1px solid var(--line); color: var(--ink-soft); background: white; }.history-item:hover, .history-item.active { background: var(--teal-100); }.history-open { display: grid; min-width: 0; gap: 3px; padding: 7px; border: 0; color: var(--ink-soft); background: transparent; font-size: 11px; text-align: left; }.history-item-title { overflow: hidden; color: var(--ink); font-size: 12px; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }.history-item time { color: var(--ink-faint); }.history-delete { width: 30px; height: 30px; color: var(--ink-faint); }.history-delete:hover:not(:disabled) { color: var(--red); background: var(--red-bg); }.history-collapsed .review-history { width: 46px; }.history-collapsed .history-toggle { justify-content: center; padding: 0; }@media (min-width: 721px) and (max-width: 960px) { .review-layout { grid-template-columns: minmax(0, 1fr) 220px; gap: 12px; }.review-layout.history-collapsed { grid-template-columns: minmax(0, 1fr) 46px; } }.review-history { position: sticky; }@media (max-width: 720px) { .review-layout, .review-layout.history-collapsed { grid-template-columns: 1fr; }.review-history { position: static; }.history-collapsed .review-history { width: 100%; }.history-collapsed .history-toggle { justify-content: flex-start; padding: 0 12px; }.review-config, .review-result { padding: 16px; }.evidence-columns { grid-template-columns: 1fr; gap: 8px; } }
</style>
