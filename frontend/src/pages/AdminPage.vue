<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { Activity, BarChart3, Database, FileCog, Gauge, Layers3, Play, Settings2, SlidersHorizontal } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { AdminRecord, MetricsOverviewView } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'

const props = defineProps<{ section: 'models' | 'prompts' | 'indexes' | 'datasets' | 'evaluations' | 'monitoring' }>()
const workspace = useWorkspaceStore()
const records = ref<AdminRecord[]>([])
const metrics = ref<MetricsOverviewView | null>(null)
const datasets = ref<AdminRecord[]>([])
const selectedDataset = ref('')
const loading = ref(true)
const error = ref('')
const running = ref(false)
const titleMap = { models: ['模型配置', '配置生成、嵌入与重排模型；密钥仅展示引用而非明文。', Settings2], prompts: ['Prompt 模板', '管理智能体模板的版本、变量与发布状态。', FileCog], indexes: ['索引与知识库', '查看知识库版本，并根据任务状态重建索引。', Layers3], datasets: ['数据集管理', '维护 QASPER、SciFact 等数据集的版本、许可证与导入状态。', Database], evaluations: ['评测中心', '选择已就绪数据集，启动生产图或消融图评测。', SlidersHorizontal], monitoring: ['运行监控', '聚合调用量、Token、延迟、错误与工作流健康度。', Gauge] } as const
const pageInfo = computed(() => titleMap[props.section])
const resource = computed(() => (new Map<string, 'model-configs' | 'prompt-templates' | 'knowledge-bases' | 'datasets'>([
  ['models', 'model-configs'], ['prompts', 'prompt-templates'], ['indexes', 'knowledge-bases'], ['datasets', 'datasets'],
])).get(props.section))

function summary(record: AdminRecord, keys: string[]) { return keys.map((key) => record[key]).filter((value) => value !== undefined && value !== null).join(' · ') }
async function load() {
  loading.value = true; error.value = ''; records.value = []; metrics.value = null
  try {
    if (props.section === 'monitoring') { const end = new Date(); const start = new Date(end.getTime() - 24 * 60 * 60 * 1000); metrics.value = await api.getMetrics({ start_time: start.toISOString(), end_time: end.toISOString(), interval: 'hour' }) }
    else if (props.section === 'evaluations') { datasets.value = (await api.listAdmin('datasets', { page: 1, page_size: 100, status: 'ready' })).items; selectedDataset.value = String(datasets.value[0]?.dataset_id ?? '') }
    else if (resource.value) records.value = (await api.listAdmin(resource.value, { page: 1, page_size: 100 })).items
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法读取管理数据。' }
  finally { loading.value = false }
}
async function runEvaluation() {
  if (!selectedDataset.value) { error.value = '请先导入并选择一个状态为 ready 的数据集。'; return }
  running.value = true; error.value = ''
  try { const task = await api.createEvaluation({ dataset_id: selectedDataset.value, split: 'test', experiment_type: 'multi_agent_rag', sample_limit: 100, random_seed: 20260717 }); await workspace.pollTask(task.task_id) }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法启动评测。' }
  finally { running.value = false }
}
watch(() => props.section, load)
onMounted(load)
</script>

<template>
  <section class="page admin-page"><PageHeader eyebrow="管理员控制台" :title="pageInfo[0]" :description="pageInfo[1]"><component :is="pageInfo[2]" :size="22" class="admin-title-icon" /></PageHeader><p v-if="error" class="inline-error">{{ error }}</p><div v-if="loading" class="skeleton-list"><div v-for="i in 4" :key="i" class="skeleton-row" /></div><template v-else-if="section === 'monitoring' && metrics"><div class="metric-grid"><article v-for="item in [{ label: '请求量', value: metrics.request_count }, { label: '问答量', value: metrics.question_count }, { label: 'P50 延迟', value: `${(metrics.latency_p50_ms / 1000).toFixed(1)}s` }, { label: '错误率', value: `${(metrics.error_rate * 100).toFixed(1)}%` }]" :key="item.label" class="metric-card"><span>{{ item.label }}</span><strong>{{ item.value }}</strong></article></div><div class="monitor-grid"><section class="card-surface"><div class="evidence-heading"><div><Activity :size="17" /><strong>检索质量</strong></div></div><dl><template v-for="(value, key) in metrics.retrieval_metrics" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template></dl></section><section class="card-surface"><div class="evidence-heading"><div><BarChart3 :size="17" /><strong>工作流健康度</strong></div></div><dl><template v-for="(value, key) in metrics.workflow_metrics" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template></dl></section></div></template><section v-else-if="section === 'evaluations'" class="evaluation-launch card-surface"><div><p class="eyebrow">受控评测</p><h2>启动实验</h2><p>评测使用指定数据集和实验类型；测试集调参会由后端拒绝。</p></div><label class="input-field plain"><span>已就绪数据集</span><select v-model="selectedDataset"><option v-for="dataset in datasets" :key="String(dataset.dataset_id)" :value="String(dataset.dataset_id)">{{ dataset.dataset_name }} · {{ dataset.dataset_version }}</option></select></label><button class="primary-button" :disabled="running" @click="runEvaluation"><Play :size="17" />{{ running ? '评测运行中…' : '运行 multi_agent_rag' }}</button></section><div v-else class="admin-record-list"><article v-for="record in records" :key="String(record.model_config_id ?? record.prompt_template_id ?? record.knowledge_base_id ?? record.dataset_id)" class="admin-record"><div class="record-icon"><component :is="pageInfo[2]" :size="20" /></div><div class="record-copy"><div class="record-title"><h2>{{ String(record.name ?? record.template_key ?? record.dataset_name ?? '未命名资源') }}</h2><StatusPill v-if="typeof record.status === 'string'" :status="record.status" /></div><p>{{ section === 'models' ? summary(record, ['model_type', 'provider', 'model_name']) : section === 'prompts' ? summary(record, ['agent_name', 'version', 'is_published']) : section === 'indexes' ? summary(record, ['description', 'paper_count', 'active_index_version']) : summary(record, ['dataset_version', 'license_name', 'imported_at']) }}</p></div><code>v{{ String(record.version ?? record.active_index_version ?? '—') }}</code></article><div v-if="!records.length" class="empty-state compact"><FileCog :size="28" /><h2>暂时没有资源</h2><p>当前接口未返回可展示的配置或资源。</p></div></div></section>
</template>
