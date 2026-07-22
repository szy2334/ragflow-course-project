<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { BookOpenCheck, CheckCircle2, FileDown, FileText, Layers3, LoaderCircle, RefreshCw, ScanSearch, Scale } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import IngestionProgress from '@/components/IngestionProgress.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { PaperSectionView, PaperView } from '@/api/contracts'
import { isUnderstandingUnavailable, paperOverview } from '@/utils/paperIngestion'

const props = defineProps<{ paperId: string }>()
const route = useRoute()
const router = useRouter()
const paper = ref<PaperView | null>(null)
const sections = ref<PaperSectionView[]>([])
const loading = ref(true)
const refreshing = ref(false)
const error = ref('')
const pdfUrl = ref('')
const pdfError = ref('')
const lastUpdatedAt = ref<Date | null>(null)
const completionNotice = ref('')
let refreshTimer: number | undefined
let completionNoticeTimer: number | undefined
const visibleSectionCounts = ref<Record<string, number>>({})
const sectionTypeLabels: Record<string, string> = {
  abstract: '摘要',
  text: '正文',
  figure: '图片',
  chart: '图表',
  figure_caption: '图注',
  table: '表格',
  formula: '公式',
  metadata: '元数据',
  reference: '参考文献',
}
const sectionTypeOrder = ['abstract', 'text', 'figure', 'chart', 'figure_caption', 'table', 'formula', 'metadata', 'reference']
const retryStage = computed(() => {
  if (paper.value?.status === 'ready' && paper.value.understanding?.status === 'unavailable') return 'understanding'
  return paper.value?.failure?.stage ?? (paper.value?.status === 'failed' ? 'mineru_parsing' : paper.value?.status)
})
const canRetry = computed(() => {
  const stageIsRetryable = ['mineru_parsing', 'ocr_processing', 'cleaning', 'quality_check', 'understanding'].includes(retryStage.value ?? '')
  if (paper.value?.status === 'failed') return paper.value.failure?.retryable === true && stageIsRetryable
  return paper.value?.status === 'ready' && paper.value.understanding?.reason === 'MODEL_ENDPOINT_UNAVAILABLE' && stageIsRetryable
})
const processing = computed(() => paper.value && !['ready', 'failed'].includes(paper.value.status))
const lastUpdatedLabel = computed(() => lastUpdatedAt.value?.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) ?? '尚未更新')
const sectionGroups = computed(() => {
  const groups = new Map<string, PaperSectionView[]>()
  for (const section of sections.value) {
    const key = section.content_type || 'text'
    groups.set(key, [...(groups.get(key) ?? []), section])
  }
  return [...groups.entries()]
    .map(([key, items]) => ({ key, label: sectionTypeLabels[key] ?? `${key} 类型`, items }))
    .sort((left, right) => {
      const leftIndex = sectionTypeOrder.indexOf(left.key)
      const rightIndex = sectionTypeOrder.indexOf(right.key)
      return (leftIndex === -1 ? sectionTypeOrder.length : leftIndex) - (rightIndex === -1 ? sectionTypeOrder.length : rightIndex)
    })
})
function visibleSections(group: { key: string; items: PaperSectionView[] }) {
  return group.items.slice(0, visibleSectionCounts.value[group.key] ?? 6)
}
function toggleSectionGroup(group: { key: string; items: PaperSectionView[] }) {
  const visible = visibleSectionCounts.value[group.key] ?? 6
  visibleSectionCounts.value[group.key] = visible >= group.items.length ? 6 : group.items.length
}

function scheduleRefresh() {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  if (!processing.value) return
  refreshTimer = window.setTimeout(() => { void load({ background: true }) }, 2_000)
}
function showCompletionNotice(message: string) {
  completionNotice.value = message
  if (completionNoticeTimer) window.clearTimeout(completionNoticeTimer)
  completionNoticeTimer = window.setTimeout(() => { completionNotice.value = '' }, 12_000)
}
async function load({ background = false }: { background?: boolean } = {}) {
  const previousStatus = paper.value?.status
  if (!paper.value && !background) loading.value = true
  else refreshing.value = true
  if (!background) error.value = ''
  try {
    const [paperResult, sectionResult] = await Promise.all([api.getPaper(props.paperId), api.listSections(props.paperId, { page: 1, page_size: 100 })])
    paper.value = paperResult; sections.value = sectionResult.items
    lastUpdatedAt.value = new Date()
    if (previousStatus && previousStatus !== 'ready' && paper.value.status === 'ready') {
      showCompletionNotice(paper.value.understanding?.status === 'unavailable'
        ? '论文已完成结构化入库，现在可以检索和阅读。'
        : '论文理解已完成，现在可以开始阅读。')
    }
    if (paper.value.status === 'ready' && !pdfUrl.value) await loadPdf()
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法读取这篇论文。' }
  finally { loading.value = false; refreshing.value = false; scheduleRefresh() }
}
async function loadPdf() {
  pdfError.value = ''
  try {
    const file = await api.getPaperFile(props.paperId)
    if (!file.size || !file.type.includes('pdf')) throw new Error('invalid PDF response')
    if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value)
    pdfUrl.value = URL.createObjectURL(file)
  } catch (cause) {
    pdfError.value = cause instanceof ApiError ? cause.message : 'PDF 原文加载失败，请检查文件是否仍在本地存储中。'
  }
}
async function createSession() {
  if (!paper.value || paper.value.status !== 'ready') return
  const session = await api.createSession({ title: paper.value.title, paper_ids: [paper.value.paper_id] })
  router.push(`/chat/${session.session_id}`)
}
function openFormatReview() {
  if (!paper.value || paper.value.status !== 'ready') return
  router.push({ path: '/review', query: { paperId: paper.value.paper_id } })
}
async function retryProcessing() {
  if (!paper.value || !canRetry.value || !retryStage.value) return
  try { await api.retryPaper(paper.value.paper_id, { stage: retryStage.value as 'mineru_parsing' | 'ocr_processing' | 'cleaning' | 'quality_check' | 'understanding', force: true }); await load() }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法从当前失败阶段重新处理。' }
}
watch(() => props.paperId, () => {
  if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value)
  paper.value = null
  sections.value = []
  pdfUrl.value = ''
  pdfError.value = ''
  visibleSectionCounts.value = {}
  loading.value = true
  void load()
})
onMounted(load)
onBeforeUnmount(() => {
  if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value)
  if (refreshTimer) window.clearTimeout(refreshTimer)
  if (completionNoticeTimer) window.clearTimeout(completionNoticeTimer)
})
</script>

<template>
  <section class="page paper-detail">
    <PageHeader eyebrow="论文阅读" :title="paper?.title || '正在读取论文'" :description="paper?.authors?.join(' · ') || '论文元数据与可追溯分析'">
      <button class="secondary-button" :disabled="paper?.status !== 'ready'" @click="createSession"><BookOpenCheck :size="18" />开始阅读</button>
      <button class="primary-button" :disabled="paper?.status !== 'ready'" @click="openFormatReview"><Scale :size="18" />格式审查</button>
      <button v-if="canRetry" class="ghost-button" @click="retryProcessing"><RefreshCw :size="17" />{{ paper?.status === 'ready' ? '重新尝试论文理解' : '从失败阶段重试' }}</button>
    </PageHeader>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="detail-skeleton"><div /><div /><div /></div>
    <template v-else-if="paper">
      <p v-if="processing" class="live-detail-status" role="status" aria-live="polite"><LoaderCircle :size="16" class="spin" /><span>正在持续更新处理状态，每 2 秒刷新；最近更新：{{ lastUpdatedLabel }}</span><button class="text-button" :disabled="refreshing" @click="load({ background: true })"><RefreshCw :size="14" :class="{ spin: refreshing }" />立即更新</button></p>
      <p v-if="completionNotice" class="completion-notice" role="status" aria-live="polite"><CheckCircle2 :size="17" />{{ completionNotice }}</p>
      <section class="paper-hero-card"><div class="paper-stamp"><FileText :size="28" /></div><div class="paper-hero-main"><div class="paper-badges"><StatusPill :status="paper.status" /><span v-if="paper.publication_year">{{ paper.publication_year }}</span><span v-if="paper.page_count">{{ paper.page_count }} 页</span></div><p>{{ paperOverview(paper) }}</p><IngestionProgress :status="paper.status" :progress="paper.parse_progress" :failure="paper.failure" :understanding="paper.understanding" /><div class="doi-row"><span>文件：{{ paper.file_name }}</span><a v-if="paper.doi" :href="`https://doi.org/${paper.doi}`" target="_blank" rel="noopener noreferrer">DOI: {{ paper.doi }}</a></div></div></section>
      <section v-if="isUnderstandingUnavailable(paper.understanding)" class="analysis-section availability-note"><div class="section-heading"><div><p class="eyebrow">论文理解</p><h2>模型理解暂不可用</h2></div></div><p>{{ paper.understanding?.message || '论文已可在本地检索和阅读；配置生成模型后可生成摘要与关键事实。' }}</p></section>
      <section v-else-if="paper.understanding?.facts?.length" class="analysis-section"><div class="section-heading"><div><p class="eyebrow">论文理解</p><h2>关键事实</h2></div></div><ul class="understanding-facts"><li v-for="fact in paper.understanding?.facts ?? []" :key="fact.claim"><strong>{{ fact.claim }}</strong><span>{{ Math.round(fact.confidence * 100) }}%</span></li></ul></section>
      <section class="reader-split"><div class="document-panel"><div class="panel-heading"><div><FileDown :size="17" /><strong>论文原文</strong></div><span v-if="route.query.page">定位至第 {{ route.query.page }} 页</span></div><iframe v-if="pdfUrl" :src="`${pdfUrl}#page=${route.query.page || 1}`" title="论文 PDF 预览" class="pdf-frame" /><div v-else class="pdf-unavailable"><ScanSearch :size="29" /><p>{{ pdfError || '正在加载 PDF 原文…' }}</p><button class="secondary-button" @click="loadPdf">重新加载 PDF</button></div></div><aside class="outline-panel"><div class="panel-heading"><div><Layers3 :size="17" /><strong>内容结构</strong></div><span>{{ sections.length }} 块</span></div><div class="section-groups"><section v-for="group in sectionGroups" :key="group.key" class="section-group"><div class="section-group-heading"><strong>{{ group.label }}</strong><span>{{ group.items.length }} 块</span></div><ol class="section-tree"><li v-for="section in visibleSections(group)" :key="section.section_id"><strong>{{ section.section_title || section.content_role || group.label }}</strong><span>第 {{ section.page_start }}{{ section.page_end !== section.page_start ? `–${section.page_end}` : '' }} 页</span></li></ol><button v-if="group.items.length > 6" class="text-button section-toggle" @click="toggleSectionGroup(group)">{{ (visibleSectionCounts[group.key] ?? 6) >= group.items.length ? '收起' : `查看其余 ${group.items.length - (visibleSectionCounts[group.key] ?? 6)} 块` }}</button></section></div></aside></section>
    </template>
  </section>
</template>

<style scoped>
.understanding-facts { display: grid; max-height: 480px; gap: 0; margin: 0; padding: 0 12px; overflow-y: auto; border: 1px solid var(--line); border-radius: var(--radius); background: white; box-shadow: var(--shadow-sm); list-style: none; scrollbar-gutter: stable; scrollbar-width: thin; scrollbar-color: #8fbbaa transparent; }.understanding-facts::-webkit-scrollbar { width: 8px; }.understanding-facts::-webkit-scrollbar-track { background: transparent; }.understanding-facts::-webkit-scrollbar-thumb { border: 2px solid transparent; border-radius: 999px; background: #8fbbaa; background-clip: padding-box; }.understanding-facts::-webkit-scrollbar-thumb:hover { background: #5d9c87; background-clip: padding-box; }.understanding-facts li { display: flex; gap: 12px; align-items: baseline; justify-content: space-between; padding: 11px 0; border-top: 1px solid var(--line); color: var(--ink); font-size: 13px; line-height: 1.55; }.understanding-facts li:first-child { border-top: 0; }.understanding-facts span { flex: 0 0 auto; color: var(--ink-faint); font-size: 11px; }
.availability-note > p { margin: 0; color: var(--ink-soft); line-height: 1.7; }
.live-detail-status, .completion-notice { display: flex; align-items: center; gap: 8px; margin: 0 0 14px; padding: 10px 12px; border-radius: 8px; font-size: 13px; }.live-detail-status { border: 1px solid #b8ddd3; background: #f3fbf8; color: #176d60; }.live-detail-status span { flex: 1; }.live-detail-status .text-button { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }.completion-notice { border: 1px solid #b6dccc; background: #f1faf4; color: #167046; font-weight: 600; }.spin { animation: spin .9s linear infinite; }@keyframes spin { to { transform: rotate(360deg); } }
.pdf-unavailable { display: grid; justify-items: center; gap: 12px; padding: 40px 20px; color: var(--ink-soft); text-align: center; }.pdf-unavailable p { margin: 0; }
.reader-split { --reader-panel-content-height: 650px; align-items: start; }.document-panel, .outline-panel { height: calc(var(--reader-panel-content-height) + 46px); }.panel-heading { height: 46px; box-sizing: border-box; flex: 0 0 auto; }.outline-panel { display: flex; flex-direction: column; }.section-groups { display: grid; min-height: 0; flex: 1; gap: 14px; padding: 0 12px 14px; overflow-y: auto; scrollbar-gutter: stable; scrollbar-width: thin; scrollbar-color: #8fbbaa transparent; }.section-groups::-webkit-scrollbar { width: 8px; }.section-groups::-webkit-scrollbar-track { background: transparent; }.section-groups::-webkit-scrollbar-thumb { border: 2px solid transparent; border-radius: 999px; background: #8fbbaa; background-clip: padding-box; }.section-groups::-webkit-scrollbar-thumb:hover { background: #5d9c87; background-clip: padding-box; }.section-group { padding-top: 12px; border-top: 1px solid var(--line); }.section-group:first-child { padding-top: 0; border-top: 0; }.section-group-heading { display: flex; justify-content: space-between; gap: 8px; color: var(--ink); font-size: 13px; }.section-group-heading span { color: var(--ink-faint); font-size: 12px; }.section-tree { display: grid; max-height: none; gap: 6px; margin: 8px 0 0; padding: 0; overflow: visible; list-style: none; }.section-tree li { display: grid; gap: 3px; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-muted); }.section-tree li strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.section-tree li span { color: var(--ink-faint); font-size: 11px; }.section-toggle { margin-top: 6px; font-size: 12px; }
</style>
