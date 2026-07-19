<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { FileOutput, FileText, Plus, X } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { ReadingReportView } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'
import { useRouter } from 'vue-router'

const workspace = useWorkspaceStore()
const router = useRouter()
const reports = ref<ReadingReportView[]>([])
const showCreate = ref(false)
const selectedPaperIds = ref<string[]>([])
const title = ref('')
const creating = ref(false)
const error = ref('')
const savedIds = ref<string[]>(JSON.parse(localStorage.getItem('known_report_ids') ?? '[]'))

function persist() { localStorage.setItem('known_report_ids', JSON.stringify(savedIds.value)) }
async function loadKnownReports() {
  reports.value = (await Promise.all(savedIds.value.map(async (id) => { try { return await api.getReport(id) } catch { return null } }))).filter((item): item is ReadingReportView => item !== null)
}
async function create() {
  if (!selectedPaperIds.value.length) { error.value = '请至少选择一篇可阅读论文。'; return }
  if (!title.value.trim()) { error.value = '请填写报告标题。'; return }
  creating.value = true; error.value = ''
  try {
    const task = await api.createReport({ paper_ids: selectedPaperIds.value, template_key: 'full_reading_report', title: title.value.trim() })
    const finished = await workspace.pollTask(task.task_id)
    const reportId = task.resource_id ?? (typeof finished?.resource_id === 'string' ? finished.resource_id : null)
    if (reportId) { savedIds.value = [...new Set([reportId, ...savedIds.value])]; persist(); router.push(`/reports/${reportId}`) }
    else { showCreate.value = false; error.value = '报告任务已创建，完成后请使用其报告 ID 打开详情。' }
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法创建阅读报告。' }
  finally { creating.value = false }
}
function toggle(id: string) { selectedPaperIds.value = selectedPaperIds.value.includes(id) ? selectedPaperIds.value.filter((paperId) => paperId !== id) : [...selectedPaperIds.value, id] }
onMounted(async () => { await workspace.loadPapers(); await loadKnownReports() })
</script>

<template>
  <section class="page reports-page"><PageHeader eyebrow="沉淀研究结果" title="阅读报告" description="将已核验的结论、证据和局限整理为可导出的阅读报告。"><button class="primary-button" @click="showCreate = true"><Plus :size="18" />新建报告</button></PageHeader><div v-if="reports.length" class="report-grid"><article v-for="report in reports" :key="report.report_id" class="report-card" @click="router.push(`/reports/${report.report_id}`)"><div class="report-icon"><FileText :size="23" /></div><div><StatusPill :status="report.status" /><h2>{{ report.title }}</h2><p>{{ report.paper_ids.length }} 篇论文 · {{ report.claims.length }} 条已核验结论</p><small>创建于 {{ new Date(report.created_at).toLocaleDateString('zh-CN') }}</small></div><FileOutput :size="18" /></article></div><div v-else class="empty-state"><FileOutput :size="35" /><h2>创建第一份阅读报告</h2><p>该页只读取当前浏览器已创建或打开过的报告详情，避免调用设计文档未定义的列表接口。</p><button class="primary-button" @click="showCreate = true"><Plus :size="18" />新建阅读报告</button></div><p v-if="error" class="inline-error">{{ error }}</p></section>
  <div v-if="showCreate" class="modal-scrim" @click.self="showCreate = false"><section class="modal report-modal" role="dialog" aria-modal="true"><button class="icon-button modal-close" aria-label="关闭" @click="showCreate = false"><X :size="20" /></button><div class="modal-icon"><FileOutput :size="24" /></div><h2>新建阅读报告</h2><label class="input-field plain"><span>报告标题</span><input v-model="title" placeholder="例如：证据感知阅读方法调研" /></label><fieldset class="paper-check-list"><legend>纳入论文</legend><label v-for="paper in Object.values(workspace.papersById).filter((item) => item.status === 'ready')" :key="paper.paper_id"><input type="checkbox" :checked="selectedPaperIds.includes(paper.paper_id)" @change="toggle(paper.paper_id)" /><span>{{ paper.title }}</span></label></fieldset><p v-if="error" class="inline-error">{{ error }}</p><div class="modal-actions"><button class="ghost-button" @click="showCreate = false">取消</button><button class="primary-button" :disabled="creating" @click="create">{{ creating ? '正在生成…' : '生成报告' }}</button></div></section></div>
</template>
