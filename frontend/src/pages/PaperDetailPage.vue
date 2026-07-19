<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { BarChart3, BookOpenCheck, Braces, FileDown, FileText, FlaskConical, Layers3, RefreshCw, ScanSearch, ShieldAlert, Sparkles } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import IngestionProgress from '@/components/IngestionProgress.vue'
import StatusPill from '@/components/StatusPill.vue'
import MarkdownContent from '@/components/MarkdownContent.vue'
import WorkflowTimeline from '@/components/WorkflowTimeline.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { PaperSectionView, PaperView, TaskAccepted } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'

const props = defineProps<{ paperId: string }>()
const route = useRoute()
const router = useRouter()
const workspace = useWorkspaceStore()
const paper = ref<PaperView | null>(null)
const sections = ref<PaperSectionView[]>([])
const loading = ref(true)
const error = ref('')
const pdfUrl = ref('')
const activeTaskId = ref('')
const activeTask = computed(() => activeTaskId.value ? workspace.workflows[activeTaskId.value] : null)
const retryStage = computed(() => paper.value?.failure?.stage ?? (paper.value?.status === 'failed' ? 'mineru_parsing' : paper.value?.status))
const canRetry = computed(() => ['mineru_parsing', 'ocr_processing', 'cleaning', 'quality_check', 'indexing'].includes(retryStage.value ?? ''))
const analysisActions = [
  { kind: 'summary' as const, title: '结构化摘要', text: '研究问题、方法、结果与局限', icon: FileText },
  { kind: 'method' as const, title: '方法拆解', text: '模块、公式与创新点', icon: Braces },
  { kind: 'experiment' as const, title: '实验分析', text: '数据集、指标、主实验与消融', icon: FlaskConical },
  { kind: 'critical-review' as const, title: '批判性审阅', text: '结论边界与证据充分性', icon: ShieldAlert },
]

async function load() {
  loading.value = true; error.value = ''
  try {
    const [paperResult, sectionResult] = await Promise.all([api.getPaper(props.paperId), api.listSections(props.paperId, { page: 1, page_size: 100 })])
    paper.value = paperResult; sections.value = sectionResult.items
    if (paper.value.status === 'ready') await loadPdf()
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法读取这篇论文。' }
  finally { loading.value = false }
}
async function loadPdf() {
  try { pdfUrl.value = URL.createObjectURL(await api.getPaperFile(props.paperId)) }
  catch { /* PDF read failure does not block the metadata and analysis views. */ }
}
async function createSession() {
  if (!paper.value || paper.value.status !== 'ready') return
  const session = await api.createSession({ title: paper.value.title, paper_ids: [paper.value.paper_id] })
  router.push(`/chat/${session.session_id}`)
}
async function runAnalysis(kind: 'summary' | 'method' | 'experiment' | 'critical-review') {
  if (!paper.value || paper.value.status !== 'ready') return
  try {
    const task = await api.createAnalysis(paper.value.paper_id, kind, { force_refresh: false })
    startTask(task)
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法启动分析。' }
}
function startTask(task: TaskAccepted) { activeTaskId.value = task.task_id; workspace.startWorkflow(task); void workspace.streamWorkflow(task.task_id) }
async function reindex() {
  try { startTask(await api.reindexPaper(props.paperId, { force: true })) }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法启动索引重建。' }
}
async function retryProcessing() {
  if (!paper.value || !canRetry.value || !retryStage.value) return
  try { startTask(await api.retryPaper(paper.value.paper_id, { stage: retryStage.value as 'mineru_parsing' | 'ocr_processing' | 'cleaning' | 'quality_check' | 'indexing', force: true })) }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法从当前失败阶段重新处理。' }
}
watch(() => props.paperId, load)
onMounted(load)
onBeforeUnmount(() => { if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value) })
</script>

<template>
  <section class="page paper-detail">
    <PageHeader eyebrow="论文阅读" :title="paper?.title || '正在读取论文'" :description="paper?.authors?.join(' · ') || '论文元数据与可追溯分析'">
      <button class="secondary-button" :disabled="paper?.status !== 'ready'" @click="createSession"><BookOpenCheck :size="18" />进入阅读工作台</button>
      <button v-if="canRetry" class="ghost-button" @click="retryProcessing"><RefreshCw :size="17" />从当前阶段重试</button>
      <button v-else class="ghost-button" :disabled="paper?.status !== 'ready'" @click="reindex"><RefreshCw :size="17" />重建索引</button>
    </PageHeader>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="detail-skeleton"><div /><div /><div /></div>
    <template v-else-if="paper">
      <section class="paper-hero-card"><div class="paper-stamp"><FileText :size="28" /></div><div class="paper-hero-main"><div class="paper-badges"><StatusPill :status="paper.status" /><StatusPill :status="paper.index_status" /><span v-if="paper.publication_year">{{ paper.publication_year }}</span><span v-if="paper.page_count">{{ paper.page_count }} 页</span></div><p>{{ paper.abstract || '尚未取得摘要。论文完成解析后，系统将呈现可阅读的结构化内容。' }}</p><IngestionProgress :status="paper.status" :progress="paper.parse_progress" :failure="paper.failure" /><div class="doi-row"><span>文件：{{ paper.file_name }}</span><a v-if="paper.doi" :href="`https://doi.org/${paper.doi}`" target="_blank" rel="noopener noreferrer">DOI: {{ paper.doi }}</a></div></div></section>
      <section class="analysis-section"><div class="section-heading"><div><p class="eyebrow">专项分析</p><h2>从不同的研究视角进入论文</h2></div><span class="section-note">所有分析均会启动独立工作流</span></div><div class="analysis-grid"><button v-for="item in analysisActions" :key="item.kind" class="analysis-card" :disabled="paper.status !== 'ready'" @click="runAnalysis(item.kind)"><component :is="item.icon" :size="21" /><strong>{{ item.title }}</strong><span>{{ item.text }}</span><Sparkles :size="16" class="analysis-arrow" /></button></div></section>
      <section class="reader-split"><div class="document-panel"><div class="panel-heading"><div><FileDown :size="17" /><strong>论文原文</strong></div><span v-if="route.query.page">定位至第 {{ route.query.page }} 页</span></div><iframe v-if="pdfUrl" :src="`${pdfUrl}#page=${route.query.page || 1}`" title="论文 PDF 预览" class="pdf-frame" /><div v-else class="pdf-unavailable"><ScanSearch :size="29" /><p>PDF 预览加载失败；你仍可查看章节与进行分析。</p></div></div><aside class="outline-panel"><div class="panel-heading"><div><Layers3 :size="17" /><strong>文章结构</strong></div><span>{{ sections.length }} 节</span></div><ol class="section-tree"><li v-for="section in sections" :key="section.section_id" :style="{ paddingLeft: `${Math.min(section.section_level, 4) * 12}px` }"><strong>{{ section.section_title }}</strong><span>第 {{ section.page_start }}{{ section.page_end !== section.page_start ? `–${section.page_end}` : '' }} 页</span></li></ol></aside></section>
      <section v-if="activeTask" class="analysis-live"><div class="analysis-live-head"><BarChart3 :size="18" /><strong>分析结果</strong><span>{{ activeTask.phase }}</span></div><div class="analysis-live-grid"><MarkdownContent :content="activeTask.text || '正在获取经证据核验的结果…'" /><WorkflowTimeline :events="activeTask.events" :phase="activeTask.phase" /></div><p v-if="activeTask.error" class="inline-error">{{ activeTask.error }}</p></section>
    </template>
  </section>
</template>
