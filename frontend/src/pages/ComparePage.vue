<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Check, Columns2, FileText, GitCompareArrows, Sparkles } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import MarkdownContent from '@/components/MarkdownContent.vue'
import WorkflowTimeline from '@/components/WorkflowTimeline.vue'
import EvidenceCard from '@/components/EvidenceCard.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { EvidenceItem } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'
import { useRouter } from 'vue-router'

const workspace = useWorkspaceStore()
const router = useRouter()
const selected = ref<string[]>([])
const dimensions = ref(['research_problem', 'method', 'dataset', 'metrics', 'results', 'limitations'])
const customQuestion = ref('')
const taskId = ref('')
const error = ref('')
const submitting = ref(false)
const papers = computed(() => Object.values(workspace.papersById).filter((paper) => paper.status === 'ready'))
const workflow = computed(() => taskId.value ? workspace.workflows[taskId.value] : null)
const options = [{ value: 'research_problem', label: '研究问题' }, { value: 'method', label: '方法' }, { value: 'dataset', label: '数据集' }, { value: 'metrics', label: '评价指标' }, { value: 'results', label: '实验结果' }, { value: 'limitations', label: '局限' }]
function toggle(id: string) { selected.value = selected.value.includes(id) ? selected.value.filter((paperId) => paperId !== id) : selected.value.length < 10 ? [...selected.value, id] : selected.value }
function toggleDimension(value: string) { dimensions.value = dimensions.value.includes(value) ? dimensions.value.filter((item) => item !== value) : [...dimensions.value, value] }
async function compare() {
  if (selected.value.length < 2) { error.value = '请至少选择两篇已索引论文。'; return }
  if (!dimensions.value.length) { error.value = '请选择至少一个比较维度。'; return }
  submitting.value = true; error.value = ''
  try { const task = await api.comparePapers({ paper_ids: selected.value, dimensions: dimensions.value, question: customQuestion.value || undefined }); taskId.value = task.task_id; workspace.startWorkflow(task); void workspace.streamWorkflow(task.task_id) }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '对比任务没有提交成功。' }
  finally { submitting.value = false }
}
function locate(evidence: EvidenceItem) { router.push({ path: `/papers/${evidence.paper_id}`, query: { page: evidence.page_number } }) }
onMounted(() => workspace.loadPapers())
</script>

<template>
  <section class="page compare-page">
    <PageHeader eyebrow="跨论文研究" title="论文对比" description="仅对已选论文建立比较维度；不可比之处会在结果中被明确标注。" />
    <div class="compare-layout">
      <section class="compare-config card-surface"><div class="config-step"><span>01</span><div><h2>选择论文</h2><p>支持 2–10 篇状态为 ready 的论文。</p></div></div><div class="compare-paper-list"><button v-for="paper in papers" :key="paper.paper_id" class="compare-paper" :class="{ selected: selected.includes(paper.paper_id) }" @click="toggle(paper.paper_id)"><span class="check-box"><Check v-if="selected.includes(paper.paper_id)" :size="15" /></span><FileText :size="17" /><div><strong>{{ paper.title }}</strong><small>{{ paper.authors?.join(' · ') || '作者待解析' }}</small></div></button><p v-if="!papers.length" class="side-empty">还没有可用于对比的论文。请先完成解析和索引。</p></div><div class="config-step"><span>02</span><div><h2>确定比较维度</h2><p>将以同一框架展示相同与不同之处。</p></div></div><div class="dimension-list"><label v-for="option in options" :key="option.value"><input type="checkbox" :checked="dimensions.includes(option.value)" @change="toggleDimension(option.value)" />{{ option.label }}</label></div><label class="text-area-field"><span>补充问题（可选）</span><textarea v-model="customQuestion" rows="3" placeholder="例如：比较两篇论文的证据充分性和结果可复现性" /></label><p v-if="error" class="inline-error">{{ error }}</p><button class="primary-button full-width" :disabled="submitting" @click="compare"><GitCompareArrows :size="18" />{{ submitting ? '正在提交…' : `开始对比（${selected.length} 篇）` }}</button></section>
      <section class="compare-result"><div v-if="!workflow" class="comparison-empty"><div class="comparison-art"><Columns2 :size="35" /><span /><span /></div><p class="eyebrow">Evidence-informed comparison</p><h2>让差异落在同一把尺子上</h2><p>选定论文与维度后，系统将并行检索、核验并综合结果。所有结论都保留对应证据。</p><div class="compare-hints"><span><Sparkles :size="15" />统一维度</span><span><Sparkles :size="15" />冲突标记</span><span><Sparkles :size="15" />原文定位</span></div></div><template v-else><div class="result-heading"><div><p class="eyebrow">对比工作流</p><h2>{{ workflow.phase }}</h2></div><span class="task-indicator">{{ workflow.completedAnswer ? '已完成' : '进行中' }}</span></div><MarkdownContent :content="workflow.text || '正在汇集各篇论文的可核验证据…'" /><WorkflowTimeline :events="workflow.events" :phase="workflow.phase" /><div v-if="workflow.evidences.length" class="compare-evidence"><h3>使用的证据</h3><EvidenceCard v-for="evidence in workflow.evidences" :key="evidence.evidence_id" :evidence="evidence" @locate="locate" /></div></template></section>
    </div>
  </section>
</template>
