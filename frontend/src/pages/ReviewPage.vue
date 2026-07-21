<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { AlertTriangle, ClipboardCheck, FileSearch, Play } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import MarkdownContent from '@/components/MarkdownContent.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { FormatProfileView, FormatReviewView } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'

const route = useRoute()
const router = useRouter()
const workspace = useWorkspaceStore()
const profiles = ref<FormatProfileView[]>([])
const selectedPaperId = ref('')
const selectedProfileId = ref('')
const selectedRuleIds = ref<string[]>([])
const review = ref<FormatReviewView | null>(null)
const loading = ref(true)
const submitting = ref(false)
const error = ref('')

const readyPapers = computed(() => Object.values(workspace.papersById).filter((paper) => paper.status === 'ready'))
const selectedProfile = computed(() => profiles.value.find((profile) => profile.format_profile_id === selectedProfileId.value) ?? null)

function setSelectedRules() {
  selectedRuleIds.value = selectedProfile.value?.rules.map((rule) => rule.rule_id) ?? []
}

function toggleRule(ruleId: string) {
  selectedRuleIds.value = selectedRuleIds.value.includes(ruleId)
    ? selectedRuleIds.value.filter((id) => id !== ruleId)
    : [...selectedRuleIds.value, ruleId]
}

async function load() {
  loading.value = true; error.value = ''
  try {
    const [_, profileResult] = await Promise.all([workspace.loadPapers({ status: 'ready' }), api.listFormatProfiles()])
    profiles.value = profileResult.items
    selectedPaperId.value = String(route.query.paperId ?? readyPapers.value[0]?.paper_id ?? '')
    selectedProfileId.value = profiles.value[0]?.format_profile_id ?? ''
    setSelectedRules()
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法加载可用格式规范。'
  } finally { loading.value = false }
}

async function startReview() {
  if (!selectedPaperId.value || !selectedProfile.value || !selectedRuleIds.value.length) {
    error.value = '请选择论文、格式规范和至少一条审查规则。'
    return
  }
  submitting.value = true; error.value = ''; review.value = null
  try {
    const task = await api.createFormatReview({
      paper_id: selectedPaperId.value,
      format_profile_id: selectedProfile.value.format_profile_id,
      rule_ids: selectedRuleIds.value,
    })
    const finished = await workspace.pollTask(task.task_id)
    if (finished?.status !== 'succeeded') throw new Error(finished?.error?.message ?? '格式审查未完成。')
    if (!task.resource_id) throw new Error('服务端未返回格式审查标识。')
    review.value = await api.getFormatReview(task.resource_id)
  } catch (cause) {
    error.value = cause instanceof ApiError || cause instanceof Error ? cause.message : '无法完成格式审查。'
  } finally { submitting.value = false }
}

watch(selectedProfileId, setSelectedRules)
onMounted(load)
</script>

<template>
  <section class="page review-page">
    <PageHeader eyebrow="格式合规审查" title="审查论文格式" description="选择格式规范后，系统只核对版式与结构规则，不评价论文研究质量。">
      <button class="secondary-button" @click="router.push('/papers')"><FileSearch :size="18" />返回阅读</button>
    </PageHeader>

    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="skeleton-list"><div v-for="item in 3" :key="item" class="skeleton-row" /></div>
    <template v-else>
      <section class="review-config card-surface">
        <div class="config-step"><span>01</span><div><h2>选择论文</h2><p>仅可审查已完成解析与理解的本地论文。</p></div></div>
        <select v-model="selectedPaperId" class="review-select"><option v-for="paper in readyPapers" :key="paper.paper_id" :value="paper.paper_id">{{ paper.title }}</option></select>

        <div class="config-step"><span>02</span><div><h2>选择格式规范</h2><p>规范库由服务端绑定，客户端不会接触 RAGFlow 数据集标识。</p></div></div>
        <select v-model="selectedProfileId" class="review-select"><option v-for="profile in profiles" :key="profile.format_profile_id" :value="profile.format_profile_id">{{ profile.name }} · {{ profile.version }}</option></select>
        <p v-if="selectedProfile?.description" class="profile-description">{{ selectedProfile.description }}</p>

        <div class="config-step"><span>03</span><div><h2>选择核对规则</h2><p>没有足够结构化证据的规则会明确标为“需人工核对”。</p></div></div>
        <div class="rule-list"><label v-for="rule in selectedProfile?.rules ?? []" :key="rule.rule_id"><input type="checkbox" :checked="selectedRuleIds.includes(rule.rule_id)" @change="toggleRule(rule.rule_id)" /><span><strong>{{ rule.title }}</strong><small>{{ rule.description }}</small></span></label></div>
        <button class="primary-button full-width" :disabled="submitting || !readyPapers.length || !profiles.length" @click="startReview"><Play :size="18" />{{ submitting ? '正在核对格式规范…' : `开始审查（${selectedRuleIds.length} 条规则）` }}</button>
      </section>

      <section v-if="review" class="review-result card-surface">
        <div class="result-title"><div><p class="eyebrow">{{ review.format_profile.name }} · {{ review.format_profile.version }}</p><h2>格式审查结果</h2></div><StatusPill :status="review.status" /></div>
        <MarkdownContent :content="review.summary_markdown || '审查结果尚未生成。'" />
        <article v-for="item in review.items" :key="item.rule_id" class="format-review-item" :class="item.result"><div><ClipboardCheck :size="18" /><strong>{{ item.rule_title }}</strong><span>{{ item.result === 'compliant' ? '符合' : item.result === 'non_compliant' ? '不符合' : item.result === 'not_applicable' ? '不适用' : '需人工核对' }}</span></div><p>{{ item.finding }}</p><p v-if="item.suggestion" class="suggestion">建议：{{ item.suggestion }}</p><footer><span v-if="item.page_numbers.length">涉及页码：{{ item.page_numbers.join('、') }}</span><span>论文证据 {{ item.paper_evidences.length }} 条 · 规范证据 {{ item.standard_evidences.length }} 条</span></footer></article>
      </section>
      <div v-else-if="!readyPapers.length || !profiles.length" class="empty-state"><AlertTriangle :size="34" /><h2>{{ !readyPapers.length ? '没有可审查论文' : '没有可用格式规范' }}</h2><p>{{ !readyPapers.length ? '请先完成论文解析与理解。' : '请由管理员创建格式规范并绑定对应的 RAGFlow 知识库。' }}</p></div>
    </template>
  </section>
</template>

<style scoped>
.review-page { display: grid; gap: 18px; }.review-config, .review-result { display: grid; gap: 16px; padding: 22px; }.config-step { display: flex; gap: 12px; align-items: flex-start; }.config-step > span { display: grid; width: 28px; height: 28px; place-items: center; color: white; background: #8b4b17; border-radius: 50%; font-size: 12px; }.config-step h2 { margin: 2px 0 4px; font-size: 16px; }.config-step p, .profile-description { margin: 0; color: var(--ink-soft); font-size: 13px; line-height: 1.5; }.review-select { width: 100%; min-height: 42px; padding: 0 12px; border: 1px solid var(--line); border-radius: 8px; background: white; color: var(--ink); }.rule-list { display: grid; gap: 8px; }.rule-list label { display: flex; gap: 10px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; cursor: pointer; }.rule-list span { display: grid; gap: 4px; }.rule-list small { color: var(--ink-soft); line-height: 1.45; }.result-title { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }.result-title h2 { margin: 2px 0 0; }.format-review-item { display: grid; gap: 8px; padding: 14px; border-left: 3px solid var(--line); background: #fafaf8; }.format-review-item > div { display: flex; gap: 8px; align-items: center; }.format-review-item > div span { margin-left: auto; color: var(--ink-soft); font-size: 13px; }.format-review-item p { margin: 0; line-height: 1.6; }.format-review-item footer { display: flex; gap: 14px; color: var(--ink-faint); font-size: 12px; }.format-review-item.non_compliant { border-color: #b24031; }.format-review-item.compliant { border-color: #4a7a57; }.format-review-item.needs_manual_check { border-color: #9c7615; }.suggestion { color: var(--ink-soft); }
</style>
