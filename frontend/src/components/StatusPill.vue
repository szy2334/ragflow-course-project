<script setup lang="ts">
import { computed } from 'vue'
import { AlertTriangle, Check, Clock3, LoaderCircle, PauseCircle, X } from 'lucide-vue-next'

const props = defineProps<{ status: string; label?: string }>()
const variant = computed(() => {
  if (['ready', 'succeeded', 'parsed'].includes(props.status)) return 'success'
  if (['failed', 'cancelled', 'stale'].includes(props.status)) return 'danger'
  if (['running', 'mineru_parsing', 'ocr_processing', 'cleaning', 'quality_check', 'understanding', 'indexing'].includes(props.status)) return 'progress'
  return 'muted'
})
const text = computed(() => props.label ?? ({
  uploaded: '已上传', mineru_parsing: 'MinerU 解析中', ocr_processing: '图表 OCR 中', cleaning: '结构化清洗中', quality_check: '质量检查中', understanding: '论文理解中', indexing: '历史索引中', ready: '可阅读',
  pending: '等待执行', running: '执行中', succeeded: '已完成', failed: '失败', cancelled: '已取消', stale: '需要重建', not_indexed: '未建索引',
} as Record<string, string>)[props.status] ?? props.status)
const icon = computed(() => ({ success: Check, danger: props.status === 'cancelled' ? PauseCircle : AlertTriangle, progress: LoaderCircle, muted: props.status === 'pending' ? Clock3 : X }[variant.value]))
</script>

<template><span class="status-pill" :class="`status-${variant}`"><component :is="icon" :size="13" :class="{ spin: variant === 'progress' }" />{{ text }}</span></template>
