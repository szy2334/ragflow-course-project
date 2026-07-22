<script setup lang="ts">
import { computed } from 'vue'
import { Check, CircleDot, LoaderCircle, SearchCheck } from 'lucide-vue-next'
import type { StreamEvent } from '@/api/contracts'

const props = defineProps<{ events: StreamEvent[]; phase: string }>()
const items = computed(() => props.events.filter((event) => ['status', 'final', 'error'].includes(event.event_type)))
const label = (event: StreamEvent) => {
  const data = event.data as { stage?: string; label?: string; message?: string }
  if (event.event_type === 'status') return data.label ?? data.stage ?? '正在处理问题'
  if (event.event_type === 'final') return '已生成并核验最终回答'
  return data.message ?? '工作流未能完成'
}
</script>

<template>
  <section class="timeline-card">
    <div class="timeline-heading"><SearchCheck :size="17" /><strong>协作进度</strong><span>{{ phase }}</span></div>
    <ol v-if="items.length" class="timeline-list">
      <li v-for="event in items" :key="event.event_id" :class="event.event_type"><component :is="event.event_type === 'final' ? Check : event.event_type === 'status' ? LoaderCircle : CircleDot" :size="15" :class="{ spin: event.event_type === 'status' }" /><span>{{ label(event) }}</span></li>
    </ol>
    <p v-else class="timeline-empty">提交问题后，这里会展示检索、核验与综合的状态。</p>
  </section>
</template>
