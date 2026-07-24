<script setup lang="ts">
import { computed } from 'vue'
import { AlertTriangle, BookOpenCheck, MessageCircle } from 'lucide-vue-next'
import type { AnswerView, RouteType } from '@/api/contracts'

const props = defineProps<{ answer: AnswerView }>()
const routeLabels: Record<RouteType, string> = {
  fact: '论文事实问答',
  explain: '论文解释问答',
  follow_up: '连续追问',
  general_chat: '通用对话',
  out_of_scope: '范围外问题',
}
const routeLabel = computed(() => routeLabels[props.answer.route_type ?? 'fact'])
const isGeneralChat = computed(() => props.answer.route_type === 'general_chat')
</script>

<template>
  <section class="answer-context" aria-label="阅读回答说明">
    <div class="context-head"><MessageCircle v-if="isGeneralChat" :size="17" /><BookOpenCheck v-else :size="17" /><div><strong>{{ routeLabel }}</strong><span>{{ isGeneralChat ? '本次回答未查阅所选论文' : '仅基于当前论文原文证据' }}</span></div></div>
    <p v-if="answer.warnings.length" class="uncertainty"><AlertTriangle :size="15" />{{ answer.warnings.join('；') }}</p>
    <p v-if="answer.evidence_sufficient === false && !answer.is_refusal" class="insufficient"><AlertTriangle :size="15" />当前检索证据未完全覆盖问题：{{ answer.evidence_gap_reason || '缺少直接相关的论文片段。' }}</p>
  </section>
</template>

<style scoped>
.answer-context { display: grid; gap: 9px; margin-top: 12px; padding: 13px; border: 1px solid var(--line); border-radius: 12px; background: white; }.context-head { display: flex; gap: 7px; align-items: center; color: var(--teal-700); }.context-head > div { display: grid; gap: 2px; }.context-head strong { color: var(--ink); font-size: 13px; }.context-head span { color: var(--ink-faint); font-size: 11px; }.uncertainty { display: flex; gap: 6px; align-items: flex-start; margin: 0; padding: 9px; border-radius: 7px; color: #845e0b; background: var(--amber-bg); font-size: 11px; line-height: 1.5; }
.insufficient { display: flex; gap: 6px; align-items: flex-start; margin: 0; padding: 9px; border-radius: 7px; color: #845e0b; background: var(--amber-bg); font-size: 11px; line-height: 1.5; }
</style>
