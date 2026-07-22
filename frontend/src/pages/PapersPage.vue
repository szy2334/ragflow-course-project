<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { AlertCircle, ArrowUpRight, CheckCircle2, FileText, FolderOpen, LoaderCircle, Plus, RefreshCw, Search, UploadCloud, X } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import IngestionProgress from '@/components/IngestionProgress.vue'
import StatusPill from '@/components/StatusPill.vue'
import { ApiError } from '@/api/http'
import { api } from '@/api'
import type { PaperView } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'
import { ingestionStageLabel, paperOverview } from '@/utils/paperIngestion'

const workspace = useWorkspaceStore()
const router = useRouter()
const query = ref('')
const selectedStatus = ref('')
const loading = ref(true)
const showUpload = ref(false)
const files = ref<File[]>([])
const uploadError = ref('')
const uploading = ref(false)
const error = ref('')
const refreshing = ref(false)
const lastUpdatedAt = ref<Date | null>(null)
const completionNotice = ref('')
let refreshTimer: number | undefined
let completionNoticeTimer: number | undefined
const papers = computed(() => Object.values(workspace.papersById))
const readyPapers = computed(() => papers.value.filter((paper) => paper.status === 'ready').length)
const activePapers = computed(() => papers.value.filter((paper) => !['ready', 'failed'].includes(paper.status)))
const lastUpdatedLabel = computed(() => lastUpdatedAt.value?.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) ?? '尚未更新')
const activePaperSummary = computed(() => {
  const summaries = activePapers.value.slice(0, 2).map((paper) => `《${paper.title}》${ingestionStageLabel(paper.status, paper.failure)}`)
  return `${summaries.join('；')}${activePapers.value.length > 2 ? `；另有 ${activePapers.value.length - 2} 篇` : ''}`
})
const statusFilters = [
  { value: 'uploaded', label: '等待处理' },
  { value: 'mineru_parsing', label: '版面解析中' },
  { value: 'ocr_processing', label: '内容补全中' },
  { value: 'cleaning', label: '二次清洗中' },
  { value: 'quality_check', label: '入库校验中' },
  { value: 'understanding', label: '论文理解中' },
  { value: 'ready', label: '可阅读' },
  { value: 'failed', label: '失败' },
]

function scheduleRefresh() {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  if (!activePapers.value.length) return
  refreshTimer = window.setTimeout(() => { void load({ background: true }) }, 2_000)
}

function showCompletionNotice(message: string) {
  completionNotice.value = message
  if (completionNoticeTimer) window.clearTimeout(completionNoticeTimer)
  completionNoticeTimer = window.setTimeout(() => { completionNotice.value = '' }, 12_000)
}

async function load({ background = false }: { background?: boolean } = {}) {
  const previousPapers = new Map(papers.value.map((paper) => [paper.paper_id, paper]))
  if (loading.value && !background) loading.value = true
  else refreshing.value = true
  if (!background) error.value = ''
  try {
    await workspace.loadPapers({ query: query.value || undefined, status: selectedStatus.value || undefined })
    lastUpdatedAt.value = new Date()
    for (const current of papers.value) {
      const previous = previousPapers.get(current.paper_id)
      if (previous && previous.status !== 'ready' && current.status === 'ready') {
        showCompletionNotice(current.understanding?.status === 'unavailable'
          ? `《${current.title}》已完成结构化入库，现在可以检索和阅读。`
          : `《${current.title}》已完成论文理解，现在可以开始阅读。`)
      }
    }
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法更新论文处理状态。'
  } finally {
    loading.value = false
    refreshing.value = false
    scheduleRefresh()
  }
}
function chooseFiles(event: Event) {
  const input = event.target as HTMLInputElement
  const next = Array.from(input.files ?? [])
  const oversized = next.find((file) => file.size > 100 * 1024 * 1024)
  if (next.length > 20) uploadError.value = '单次最多上传 20 个文件。'
  else if (oversized) uploadError.value = `「${oversized.name}」超过 100 MB 限制。`
  else if (next.some((file) => !file.name.toLowerCase().endsWith('.pdf'))) uploadError.value = '只能上传 PDF 格式的论文。'
  else { files.value = next; uploadError.value = '' }
}
async function upload() {
  if (!files.value.length) { uploadError.value = '请先选择至少一篇 PDF。'; return }
  uploading.value = true; uploadError.value = ''
  try {
    const result = await api.uploadPapers(files.value)
    showUpload.value = false
    files.value = []
    const duplicates = result.items.filter((item) => item.duplicate)
    const accepted = result.items.filter((item) => !item.duplicate)
    if (duplicates.length && !accepted.length) {
      showCompletionNotice(`论文库中已有该论文：${duplicates.map((item) => `《${item.file_name}》`).join('、')}，未重复处理。`)
    } else if (duplicates.length) {
      showCompletionNotice(`已接收 ${accepted.length} 篇论文；${duplicates.map((item) => `《${item.file_name}》`).join('、')}已在论文库中，未重复处理。`)
    } else {
      showCompletionNotice(`已接收 ${accepted.length} 篇论文，正在持续更新处理进度。`)
    }
    await load()
  } catch (cause) { uploadError.value = cause instanceof ApiError ? cause.message : '上传未完成，请重试。' }
  finally { uploading.value = false }
}
async function startReading(paper: PaperView) {
  if (paper.status !== 'ready') return
  try {
    const session = await api.createSession({ title: paper.title, paper_ids: [paper.paper_id] })
    router.push(`/chat/${session.session_id}`)
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法创建阅读会话。' }
}
onMounted(load)
onBeforeUnmount(() => {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  if (completionNoticeTimer) window.clearTimeout(completionNoticeTimer)
})
</script>

<template>
  <section class="page page-papers">
    <PageHeader eyebrow="本地论文" title="读论文" description="基于本地结构化 chunks 检索、理解与总结，不评价论文质量。">
      <button class="primary-button" @click="showUpload = true"><Plus :size="18" />添加论文</button>
    </PageHeader>
    <div class="library-summary">
      <div><span class="summary-number">{{ workspace.paperTotal }}</span><span>篇已收录</span></div>
      <div><span class="summary-number">{{ readyPapers }}</span><span>篇可开始阅读</span></div>
      <p><AlertCircle :size="16" /> 完成结构化入库后即可阅读；摘要由模型理解阶段生成。</p>
    </div>
    <div v-if="activePapers.length" class="live-status" role="status" aria-live="polite"><LoaderCircle :size="18" class="spin" /><div><strong>正在持续更新 {{ activePapers.length }} 篇论文的处理状态</strong><span>{{ activePaperSummary }} · 每 2 秒刷新 · 最近更新：{{ lastUpdatedLabel }}</span></div><button class="text-button" :disabled="refreshing" @click="load({ background: true })"><RefreshCw :size="15" :class="{ spin: refreshing }" />立即更新</button></div>
    <div v-if="completionNotice" class="completion-notice" role="status" aria-live="polite"><CheckCircle2 :size="18" />{{ completionNotice }}</div>
    <div class="toolbar">
      <label class="search-input"><Search :size="18" /><input v-model="query" aria-label="搜索论文" placeholder="按标题、作者或 DOI 搜索" @keyup.enter="() => load()" /></label>
      <select v-model="selectedStatus" aria-label="按状态筛选" @change="() => load()"><option value="">全部状态</option><option v-for="item in statusFilters" :key="item.value" :value="item.value">{{ item.label }}</option></select>
      <button class="secondary-button" @click="() => load()">筛选</button>
    </div>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="skeleton-list"><div v-for="i in 4" :key="i" class="skeleton-row" /></div>
    <div v-else-if="papers.length" class="paper-grid">
      <article v-for="paper in papers" :key="paper.paper_id" class="paper-card">
        <div class="paper-card-icon"><FileText :size="22" /></div>
        <div class="paper-card-main"><div class="card-heading"><StatusPill :status="paper.status" /><span v-if="paper.publication_year">{{ paper.publication_year }}</span></div><h2>{{ paper.title }}</h2><p class="authors">{{ paper.authors?.join(' · ') || '作者信息待解析' }}</p><p class="abstract">{{ paperOverview(paper) }}</p><IngestionProgress :status="paper.status" :progress="paper.parse_progress" :failure="paper.failure" :understanding="paper.understanding" compact /><div class="paper-meta"><span>{{ paper.page_count ? `${paper.page_count} 页` : '页数待解析' }}</span><span>仅保存在本地工作区</span></div></div>
        <footer class="card-footer"><button class="text-button" @click="router.push(`/papers/${paper.paper_id}`)">查看详情 <ArrowUpRight :size="15" /></button><button class="action-button" :disabled="paper.status !== 'ready'" @click="startReading(paper)">开始阅读</button></footer>
      </article>
    </div>
    <div v-else class="empty-state"><FolderOpen :size="34" /><h2>还没有论文</h2><button class="primary-button" @click="showUpload = true"><UploadCloud :size="18" />添加第一篇论文</button></div>
  </section>

  <div v-if="showUpload" class="modal-scrim" @click.self="showUpload = false"><section class="modal upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title"><button class="icon-button modal-close" aria-label="关闭" @click="showUpload = false"><X :size="20" /></button><div class="modal-icon"><UploadCloud :size="25" /></div><h2 id="upload-title">添加论文</h2><p>支持一次选择 1–20 个 PDF，每个文件不超过 100 MB。系统将依次进行版面解析、内容补全、二次清洗、入库校验和论文理解。</p><label class="upload-drop"><input type="file" accept="application/pdf,.pdf" multiple @change="chooseFiles" /><UploadCloud :size="25" /><strong>选择 PDF 文件</strong><span>或将文件拖到这里</span></label><ul v-if="files.length" class="selected-files"><li v-for="file in files" :key="file.name"><FileText :size="15" />{{ file.name }} <span>{{ (file.size / 1024 / 1024).toFixed(1) }} MB</span></li></ul><p v-if="uploadError" class="inline-error" role="alert">{{ uploadError }}</p><div class="modal-actions"><button class="ghost-button" @click="showUpload = false">取消</button><button class="primary-button" :disabled="uploading || !files.length" @click="upload">{{ uploading ? '正在提交…' : '开始处理' }}</button></div></section></div>
</template>

<style scoped>
.live-status, .completion-notice { display: flex; align-items: center; gap: 10px; margin: 14px 0; padding: 12px 14px; border-radius: 9px; font-size: 13px; }
.live-status { border: 1px solid #b8ddd3; background: #f3fbf8; color: #176d60; }.live-status > div { display: grid; flex: 1; gap: 2px; }.live-status span { color: #53756d; font-size: 12px; }.live-status .text-button { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.completion-notice { border: 1px solid #b6dccc; background: #f1faf4; color: #167046; font-weight: 600; }
.spin { animation: spin .9s linear infinite; }@keyframes spin { to { transform: rotate(360deg); } }
@media (max-width: 680px) { .live-status { align-items: flex-start; flex-wrap: wrap; }.live-status .text-button { margin-left: 28px; } }
</style>
