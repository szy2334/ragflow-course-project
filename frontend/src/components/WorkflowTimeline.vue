<script setup lang="ts">
import { computed } from 'vue'
import { CheckCircle2, LoaderCircle, SearchCheck, XCircle } from 'lucide-vue-next'
import type { StreamEvent } from '@/api/contracts'

const props = defineProps<{ events: StreamEvent[]; phase: string }>()
const items = computed(() => props.events.filter((event) => ['status', 'final', 'error'].includes(event.event_type)))
type TimelineState = 'done' | 'running' | 'error'
const itemState = (event: StreamEvent, index: number): TimelineState => {
  if (event.event_type === 'error') return 'error'
  if (event.event_type === 'final' || index < items.value.length - 1) return 'done'
  return 'running'
}
const itemIcon = (event: StreamEvent, index: number) => {
  const state = itemState(event, index)
  if (state === 'done') return CheckCircle2
  if (state === 'error') return XCircle
  return LoaderCircle
}
const phaseState = computed(() => {
  const latest = items.value[items.value.length - 1]
  if (!latest) return 'idle'
  if (latest.event_type === 'error') return 'error'
  if (latest.event_type === 'final') return 'done'
  return 'running'
})
const label = (event: StreamEvent) => {
  const data = event.data as { stage?: string; label?: string; message?: string }
  if (event.event_type === 'status') return data.label ?? data.stage ?? '正在处理问题'
  if (event.event_type === 'final') return '已生成并核验最终回答'
  return data.message ?? '工作流未能完成'
}
</script>

<template>
  <section class="timeline-card">
    <div class="timeline-heading"><SearchCheck :size="17" /><strong>协作进度</strong><span class="timeline-phase" :class="`phase-${phaseState}`"><i />{{ phase }}</span></div>
    <ol v-if="items.length" class="timeline-list">
      <li v-for="(event, index) in items" :key="event.event_id" :class="[`timeline-${itemState(event, index)}`, { 'timeline-last': index === items.length - 1 }]">
        <span class="timeline-node"><component :is="itemIcon(event, index)" :size="15" :class="{ spin: itemState(event, index) === 'running' }" /></span>
        <span>{{ label(event) }}</span>
      </li>
    </ol>
    <p v-else class="timeline-empty">提交问题后，这里会展示检索、核验与综合的状态。</p>
  </section>
</template>
