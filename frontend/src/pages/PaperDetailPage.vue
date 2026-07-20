<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { BookOpenCheck, FileDown, FileText, Layers3, RefreshCw, ScanSearch, Scale } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import IngestionProgress from '@/components/IngestionProgress.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { PaperSectionView, PaperView } from '@/api/contracts'

const props = defineProps<{ paperId: string }>()
const route = useRoute()
const router = useRouter()
const paper = ref<PaperView | null>(null)
const sections = ref<PaperSectionView[]>([])
const loading = ref(true)
const error = ref('')
const pdfUrl = ref('')
const retryStage = computed(() => paper.value?.failure?.stage ?? (paper.value?.status === 'failed' ? 'mineru_parsing' : paper.value?.status))
const canRetry = computed(() => ['mineru_parsing', 'ocr_processing', 'cleaning', 'quality_check', 'understanding'].includes(retryStage.value ?? ''))

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
async function createReviewSession() {
  if (!paper.value || paper.value.status !== 'ready') return
  try {
    const session = await api.createSession({ title: `审阅：${paper.value.title}`, paper_ids: [paper.value.paper_id] })
    router.push({ path: `/chat/${session.session_id}`, query: { mode: 'review' } })
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法创建审阅会话。' }
}
async function retryProcessing() {
  if (!paper.value || !canRetry.value || !retryStage.value) return
  try { await api.retryPaper(paper.value.paper_id, { stage: retryStage.value as 'mineru_parsing' | 'ocr_processing' | 'cleaning' | 'quality_check' | 'understanding', force: true }); await load() }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法从当前失败阶段重新处理。' }
}
watch(() => props.paperId, load)
onMounted(load)
onBeforeUnmount(() => { if (pdfUrl.value) URL.revokeObjectURL(pdfUrl.value) })
</script>

<template>
  <section class="page paper-detail">
    <PageHeader eyebrow="论文阅读" :title="paper?.title || '正在读取论文'" :description="paper?.authors?.join(' · ') || '论文元数据与可追溯分析'">
      <button class="secondary-button" :disabled="paper?.status !== 'ready'" @click="createSession"><BookOpenCheck :size="18" />开始阅读</button>
      <button class="primary-button" :disabled="paper?.status !== 'ready'" @click="createReviewSession"><Scale :size="18" />开始审阅</button>
      <button v-if="canRetry" class="ghost-button" @click="retryProcessing"><RefreshCw :size="17" />从当前阶段重试</button>
    </PageHeader>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="detail-skeleton"><div /><div /><div /></div>
    <template v-else-if="paper">
      <section class="paper-hero-card"><div class="paper-stamp"><FileText :size="28" /></div><div class="paper-hero-main"><div class="paper-badges"><StatusPill :status="paper.status" /><span v-if="paper.publication_year">{{ paper.publication_year }}</span><span v-if="paper.page_count">{{ paper.page_count }} 页</span></div><p>{{ paper.understanding?.paper_summary || paper.abstract || '论文完成理解后将显示摘要。' }}</p><IngestionProgress :status="paper.status" :progress="paper.parse_progress" :failure="paper.failure" /><div class="doi-row"><span>文件：{{ paper.file_name }}</span><a v-if="paper.doi" :href="`https://doi.org/${paper.doi}`" target="_blank" rel="noopener noreferrer">DOI: {{ paper.doi }}</a></div></div></section>
      <section v-if="paper.understanding?.facts.length" class="analysis-section"><div class="section-heading"><div><p class="eyebrow">论文理解</p><h2>关键事实</h2></div></div><ul class="understanding-facts"><li v-for="fact in paper.understanding.facts" :key="fact.claim"><strong>{{ fact.claim }}</strong><span>{{ Math.round(fact.confidence * 100) }}%</span></li></ul></section>
      <section class="reader-split"><div class="document-panel"><div class="panel-heading"><div><FileDown :size="17" /><strong>论文原文</strong></div><span v-if="route.query.page">定位至第 {{ route.query.page }} 页</span></div><iframe v-if="pdfUrl" :src="`${pdfUrl}#page=${route.query.page || 1}`" title="论文 PDF 预览" class="pdf-frame" /><div v-else class="pdf-unavailable"><ScanSearch :size="29" /><p>PDF 预览加载失败；你仍可查看章节与进行分析。</p></div></div><aside class="outline-panel"><div class="panel-heading"><div><Layers3 :size="17" /><strong>文章结构</strong></div><span>{{ sections.length }} 节</span></div><ol class="section-tree"><li v-for="section in sections" :key="section.section_id" :style="{ paddingLeft: `${Math.min(section.section_level, 4) * 12}px` }"><strong>{{ section.section_title }}</strong><span>第 {{ section.page_start }}{{ section.page_end !== section.page_start ? `–${section.page_end}` : '' }} 页</span></li></ol></aside></section>
    </template>
  </section>
</template>

<style scoped>
.understanding-facts { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }.understanding-facts li { display: flex; gap: 12px; align-items: baseline; justify-content: space-between; padding: 12px 0; border-top: 1px solid var(--line); color: var(--ink); font-size: 14px; }.understanding-facts span { flex: 0 0 auto; color: var(--ink-faint); font-size: 12px; }
</style>
