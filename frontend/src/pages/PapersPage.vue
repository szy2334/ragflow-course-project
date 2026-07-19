<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { AlertCircle, ArrowUpRight, FileText, FolderOpen, Plus, Search, UploadCloud, X } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import IngestionProgress from '@/components/IngestionProgress.vue'
import StatusPill from '@/components/StatusPill.vue'
import { ApiError } from '@/api/http'
import { api } from '@/api'
import type { PaperView } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'

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
const papers = computed(() => Object.values(workspace.papersById))
const readyPapers = computed(() => papers.value.filter((paper) => paper.status === 'ready').length)
const statusFilters = [
  { value: 'mineru_parsing', label: 'MinerU 解析中' },
  { value: 'ocr_processing', label: '图表 OCR 中' },
  { value: 'cleaning', label: '结构化清洗中' },
  { value: 'quality_check', label: '质量检查中' },
  { value: 'indexing', label: '索引中' },
  { value: 'ready', label: '可问答' },
  { value: 'failed', label: '失败' },
]

async function load() {
  loading.value = true; error.value = ''
  try { await workspace.loadPapers({ query: query.value || undefined, status: selectedStatus.value || undefined }) }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法加载论文库。' }
  finally { loading.value = false }
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
    await Promise.all(result.items.map((item) => workspace.pollTask(item.task_id, 3)))
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
</script>

<template>
  <section class="page page-papers">
    <PageHeader eyebrow="我的研究空间" title="论文库" description="从上传到证据化阅读，所有论文状态一目了然。">
      <button class="primary-button" @click="showUpload = true"><Plus :size="18" />上传论文</button>
    </PageHeader>
    <div class="library-summary">
      <div><span class="summary-number">{{ workspace.paperTotal }}</span><span>篇已收录</span></div>
      <div><span class="summary-number">{{ readyPapers }}</span><span>篇可开始阅读</span></div>
      <p><AlertCircle :size="16" /> 仅 <strong>ready</strong> 状态的论文可进入问答，以确保回答可追溯。</p>
    </div>
    <div class="toolbar">
      <label class="search-input"><Search :size="18" /><input v-model="query" aria-label="搜索论文" placeholder="按标题、作者或 DOI 搜索" @keyup.enter="load" /></label>
      <select v-model="selectedStatus" aria-label="按状态筛选" @change="load"><option value="">全部状态</option><option v-for="item in statusFilters" :key="item.value" :value="item.value">{{ item.label }}</option></select>
      <button class="secondary-button" @click="load">筛选</button>
    </div>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="skeleton-list"><div v-for="i in 4" :key="i" class="skeleton-row" /></div>
    <div v-else-if="papers.length" class="paper-grid">
      <article v-for="paper in papers" :key="paper.paper_id" class="paper-card">
        <div class="paper-card-icon"><FileText :size="22" /></div>
        <div class="paper-card-main"><div class="card-heading"><StatusPill :status="paper.status" /><span v-if="paper.publication_year">{{ paper.publication_year }}</span></div><h2>{{ paper.title }}</h2><p class="authors">{{ paper.authors?.join(' · ') || '作者信息待解析' }}</p><p class="abstract">{{ paper.abstract || '论文正在解析中，摘要将在解析完成后显示。' }}</p><IngestionProgress :status="paper.status" :progress="paper.parse_progress" :failure="paper.failure" compact /><div class="paper-meta"><span>{{ paper.page_count ? `${paper.page_count} 页` : '页数待解析' }}</span><span>{{ paper.index_status === 'succeeded' ? '索引已就绪' : '索引：' + paper.index_status }}</span></div></div>
        <footer class="card-footer"><button class="text-button" @click="router.push(`/papers/${paper.paper_id}`)">查看详情 <ArrowUpRight :size="15" /></button><button class="action-button" :disabled="paper.status !== 'ready'" @click="startReading(paper)">开始阅读</button></footer>
      </article>
    </div>
    <div v-else class="empty-state"><FolderOpen :size="34" /><h2>还没有论文</h2><p>上传 PDF 后，系统会解析章节、表格与参考文献，并在索引完成后开放问答。</p><button class="primary-button" @click="showUpload = true"><UploadCloud :size="18" />上传第一篇论文</button></div>
  </section>

  <div v-if="showUpload" class="modal-scrim" @click.self="showUpload = false"><section class="modal upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title"><button class="icon-button modal-close" aria-label="关闭" @click="showUpload = false"><X :size="20" /></button><div class="modal-icon"><UploadCloud :size="25" /></div><h2 id="upload-title">上传论文</h2><p>支持一次选择 1–20 个 PDF，每个文件不超过 100 MB。服务端会依次校验版面解析、图表 OCR、结构化清洗、质量门禁与索引映射。</p><label class="upload-drop"><input type="file" accept="application/pdf,.pdf" multiple @change="chooseFiles" /><UploadCloud :size="25" /><strong>选择 PDF 文件</strong><span>或将文件拖到这里</span></label><ul v-if="files.length" class="selected-files"><li v-for="file in files" :key="file.name"><FileText :size="15" />{{ file.name }} <span>{{ (file.size / 1024 / 1024).toFixed(1) }} MB</span></li></ul><p v-if="uploadError" class="inline-error" role="alert">{{ uploadError }}</p><div class="modal-actions"><button class="ghost-button" @click="showUpload = false">取消</button><button class="primary-button" :disabled="uploading || !files.length" @click="upload">{{ uploading ? '正在提交…' : '开始解析与索引' }}</button></div></section></div>
</template>
