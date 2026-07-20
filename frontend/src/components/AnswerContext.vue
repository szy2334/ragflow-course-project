<script setup lang="ts">
import { computed } from 'vue'
import { BadgeCheck, FileSearch, MessageSquareWarning, Scale, ShieldCheck } from 'lucide-vue-next'
import type { AnswerView, ReviewOpinion, RouteType } from '@/api/contracts'

const props = defineProps<{ answer: AnswerView }>()

const routeLabels: Record<RouteType, string> = {
  fact: '论文事实问答', explain: '论文解释问答', review: '证据化评审', score: '标准化评分', follow_up: '连续追问', out_of_scope: '范围外问题',
}
const routeLabel = computed(() => props.answer.route_type ? routeLabels[props.answer.route_type] : '已核验回答')
const hasReview = computed(() => Boolean(props.answer.review_opinions?.length))
const opinionLabel = (opinion: ReviewOpinion) => ({ critical: '严格视角', supportive: '平衡视角', mixed: '综合视角' }[opinion.position])
</script>

<template>
  <section class="answer-context" aria-label="回答依据与评审结论">
    <div class="context-head"><div><FileSearch :size="17" /><strong>{{ routeLabel }}</strong></div><span>{{ Math.round(answer.confidence * 100) }}% 置信度</span></div>
    <div v-if="answer.score" class="score-row"><Scale :size="18" /><div><span>评分维度</span><strong>{{ answer.score.dimension }}</strong></div><b>{{ answer.score.value }}<small>/ {{ answer.score.scale }}</small></b></div>
    <div v-if="hasReview" class="review-list"><article v-for="opinion in answer.review_opinions" :key="opinion.reviewer"><div><ShieldCheck :size="15" /><strong>{{ opinion.reviewer === 'review_a' ? '评审 A' : '评审 B' }}</strong><span>{{ opinionLabel(opinion) }}</span></div><p>{{ opinion.summary }}</p></article></div>
    <div v-if="answer.standards?.length" class="standards"><BadgeCheck :size="15" /><span>参考论文：</span><strong>{{ answer.standards.map((item) => `${item.name} ${item.version}`).join('；') }}</strong></div>
    <div v-if="answer.warnings.length || answer.is_refusal" class="uncertainty"><MessageSquareWarning :size="16" /><div><strong>不确定性说明</strong><p>{{ answer.refusal_reason || answer.warnings.join('；') }}</p></div></div>
  </section>
</template>

<style scoped>
.answer-context { display: grid; gap: 11px; margin-top: 12px; padding: 13px; border: 1px solid #d8e8e1; border-radius: 8px; background: #fafffc; }
.context-head, .context-head > div, .score-row, .review-list article > div, .standards, .uncertainty { display: flex; gap: 7px; align-items: center; }
.context-head { justify-content: space-between; color: #176d60; }.context-head strong { color: #1c302a; font-size: 13px; }.context-head > span { color: #6d837a; font-size: 10px; }
.score-row { padding: 10px; border: 1px solid #d9e7e2; border-radius: 7px; background: white; color: #176d60; }.score-row > div { display: grid; gap: 1px; flex: 1; }.score-row span { color: #789087; font-size: 10px; }.score-row strong { color: #284038; font-size: 12px; }.score-row b { color: #176d60; font-size: 20px; }.score-row small { color: #789087; font-size: 11px; font-weight: 600; }
.review-list { display: grid; gap: 7px; }.review-list article { padding: 9px 0; border-top: 1px solid #e1ece7; }.review-list article > div { color: #5c776d; font-size: 10px; }.review-list strong { color: #30483f; font-size: 11px; }.review-list span { margin-left: auto; color: #176d60; }.review-list p, .uncertainty p { margin: 5px 0 0; color: #526b61; font-size: 11px; line-height: 1.5; }
.standards { color: #506d62; font-size: 11px; line-height: 1.4; }.standards strong { color: #30483f; }
.uncertainty { align-items: flex-start; padding: 9px; border-radius: 7px; color: #89620d; background: #fff8e8; }.uncertainty strong { font-size: 11px; }.uncertainty p { color: #785b1c; }
</style>
