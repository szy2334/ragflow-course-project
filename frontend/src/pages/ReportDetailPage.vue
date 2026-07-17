<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Download, FileText, LoaderCircle, Quote, ShieldCheck } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import StatusPill from '@/components/StatusPill.vue'
import MarkdownContent from '@/components/MarkdownContent.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { ReadingReportView } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'

const props = defineProps<{ reportId: string }>()
const workspace = useWorkspaceStore()
const report = ref<ReadingReportView | null>(null)
const error = ref('')
const exporting = ref('')
const downloadUrl = ref('')
const completion = computed(() => report.value?.completed_at ? new Date(report.value.completed_at).toLocaleString('zh-CN') : '尚未完成')
async function load() { try { report.value = await api.getReport(props.reportId) } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法读取报告详情。' } }
async function exportReport(format: 'markdown' | 'pdf' | 'docx') {
  exporting.value = format; error.value = ''; downloadUrl.value = ''
  try {
    const task = await api.exportReport(props.reportId, format)
    const finished = await workspace.pollTask(task.task_id)
    const result = finished?.result
    const url = result && typeof result.download_url === 'string' ? result.download_url : ''
    if (url) downloadUrl.value = url
    else error.value = '导出任务已完成，但未返回下载地址。'
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '导出失败，请稍后重试。' }
  finally { exporting.value = '' }
}
onMounted(load)
</script>

<template>
  <section class="page report-detail"><PageHeader eyebrow="阅读报告" :title="report?.title || '正在加载报告'" :description="report ? `${report.paper_ids.length} 篇论文 · 完成于 ${completion}` : '报告中的结论均来自持久化证据' "><template v-if="report"><StatusPill :status="report.status" /><div class="export-actions"><button v-for="format in ['markdown', 'pdf', 'docx'] as const" :key="format" class="secondary-button" :disabled="Boolean(exporting)" @click="exportReport(format)"><LoaderCircle v-if="exporting === format" :size="16" class="spin" /><Download v-else :size="16" />{{ format.toUpperCase() }}</button></div></template></PageHeader><p v-if="error" class="inline-error">{{ error }}</p><a v-if="downloadUrl" class="download-ready" :href="downloadUrl" target="_blank" rel="noopener noreferrer"><Download :size="17" />导出文件已准备好，点击下载</a><div v-if="report" class="report-detail-grid"><article class="report-content card-surface"><MarkdownContent :content="report.content_markdown || '报告仍在生成中。'" /></article><aside class="report-aside"><section class="report-stat"><FileText :size="19" /><div><strong>{{ report.paper_ids.length }}</strong><span>纳入论文</span></div></section><section class="report-stat"><ShieldCheck :size="19" /><div><strong>{{ report.claims.length }}</strong><span>核验结论</span></div></section><section class="report-evidence-ids"><div class="evidence-heading"><div><Quote :size="17" /><strong>引用证据</strong></div><span>{{ report.evidence_ids.length }} 条</span></div><p>报告接口仅返回证据 ID；为避免越权读取，本页不会调用文档未定义的跨报告证据接口。</p><code v-for="id in report.evidence_ids" :key="id">{{ id }}</code></section></aside></div></section>
</template>
