<script setup lang="ts">
import { computed } from 'vue'
import { Check, CircleDot, LoaderCircle, SearchCheck } from 'lucide-vue-next'
import type { StreamEvent } from '@/api/contracts'

const props = defineProps<{ events: StreamEvent[]; phase: string }>()
const items = computed(() => props.events.filter((event) => ['workflow_started', 'agent_started', 'agent_completed', 'retrieval_completed'].includes(event.event_type)))
const label = (event: StreamEvent) => {
  if (event.event_type === 'workflow_started') return '已规划阅读工作流'
  if (event.event_type === 'retrieval_completed') return '已检索并筛选证据'
  const data = event.data as { agent_name?: string; node_name?: string }
  const name = data.agent_name ?? event.agent_name ?? data.node_name ?? '智能体'
  return event.event_type === 'agent_completed' ? `${name} 已完成` : `${name} 正在工作`
}
</script>

<template>
  <section class="timeline-card">
    <div class="timeline-heading"><SearchCheck :size="17" /><strong>协作进度</strong><span>{{ phase }}</span></div>
    <ol v-if="items.length" class="timeline-list">
      <li v-for="event in items" :key="event.event_id" :class="event.event_type"><component :is="event.event_type === 'agent_completed' ? Check : event.event_type === 'agent_started' ? LoaderCircle : CircleDot" :size="15" :class="{ spin: event.event_type === 'agent_started' }" /><span>{{ label(event) }}</span></li>
    </ol>
    <p v-else class="timeline-empty">提交问题后，这里会展示检索、核验与综合的状态。</p>
  </section>
</template>
