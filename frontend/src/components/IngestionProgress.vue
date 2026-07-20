<script setup lang="ts">
import { computed } from 'vue'
import { BadgeCheck, BrainCircuit, CircleAlert, FileUp, ScanSearch, ScanText, Sparkles, TableProperties } from 'lucide-vue-next'
import type { PaperFailure, PaperStatus } from '@/api/contracts'

const props = withDefaults(defineProps<{
  status: PaperStatus
  progress: number
  failure?: PaperFailure | null
  compact?: boolean
}>(), { failure: null, compact: false })

const stages = [
  { key: 'uploaded', label: '接收', icon: FileUp },
  { key: 'mineru_parsing', label: '解析', icon: ScanText },
  { key: 'ocr_processing', label: '图表 OCR', icon: ScanSearch },
  { key: 'cleaning', label: '清洗', icon: Sparkles },
  { key: 'quality_check', label: '质检', icon: TableProperties },
  { key: 'understanding', label: '理解', icon: BrainCircuit },
  { key: 'ready', label: '可问答', icon: BadgeCheck },
] as const

const activeIndex = computed(() => {
  const key = props.status === 'failed' ? props.failure?.stage ?? 'uploaded' : props.status
  return Math.max(0, stages.findIndex((stage) => stage.key === key))
})
const normalizedProgress = computed(() => Math.round(Math.max(0, Math.min(1, props.progress)) * 100))
const statusLabel = computed(() => ({
  uploaded: '文件已接收，等待进入处理队列',
  mineru_parsing: '正在解析论文版面、章节与媒体对象',
  ocr_processing: '正在识别图片、流程图和表格内容',
  cleaning: '正在关联正文、图表与结构化内容',
  quality_check: '正在检查来源追踪和可索引性',
  understanding: '正在提取论文问题、方法、实验与结论',
  indexing: '正在整理历史索引任务',
  ready: '解析、质检和论文理解均已完成',
  failed: props.failure?.message ?? '处理未完成，可从失败阶段重新尝试。',
} as Record<PaperStatus, string>)[props.status])
const lineWidth = computed(() => `${(activeIndex.value / (stages.length - 1)) * 100}%`)
</script>

<template>
  <section class="ingestion-progress" :class="{ compact }" :aria-label="`论文入库状态：${statusLabel}`">
    <div class="ingestion-summary">
      <span :class="{ failed: status === 'failed' }"><CircleAlert v-if="status === 'failed'" :size="15" /><span>{{ statusLabel }}</span></span>
      <strong v-if="status !== 'ready' && status !== 'failed'">{{ normalizedProgress }}%</strong>
    </div>
    <div class="stage-track" role="list" aria-label="论文入库阶段">
      <i class="stage-line" :style="{ width: lineWidth }" />
      <div v-for="(stage, index) in stages" :key="stage.key" class="stage" :class="{ complete: index < activeIndex, active: index === activeIndex && status !== 'failed', failed: index === activeIndex && status === 'failed' }" role="listitem" :title="stage.label">
        <component :is="stage.icon" :size="13" />
        <span>{{ stage.label }}</span>
      </div>
    </div>
    <p v-if="failure" class="failure-detail"><CircleAlert :size="14" /><span>{{ failure.error_code }}：{{ failure.message }}</span></p>
  </section>
</template>

<style scoped>
.ingestion-progress { display: grid; gap: 10px; padding: 12px; border: 1px solid #d9e6e1; border-radius: 8px; background: #fbfefd; }
.ingestion-summary { display: flex; gap: 12px; align-items: center; justify-content: space-between; color: #48645b; font-size: 12px; line-height: 1.45; }
.ingestion-summary > span, .failure-detail { display: flex; gap: 6px; align-items: flex-start; }
.ingestion-summary .failed, .failure-detail { color: #ad3944; }
.ingestion-summary strong { color: #12665b; font-size: 12px; }
.stage-track { position: relative; display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 2px; }
.stage-track::before, .stage-line { position: absolute; top: 7px; left: 7%; width: 86%; height: 2px; background: #dbe7e2; content: ''; }
.stage-line { z-index: 1; background: #2c9583; transition: width .25s ease; }
.stage { position: relative; z-index: 2; display: grid; gap: 4px; justify-items: center; color: #99aaa4; font-size: 10px; line-height: 1.2; text-align: center; }
.stage svg { width: 24px; height: 24px; padding: 5px; border: 1px solid #dbe7e2; border-radius: 50%; background: white; }
.stage.complete, .stage.active { color: #176d60; }
.stage.complete svg, .stage.active svg { border-color: #62ad9d; background: #e9f7f2; }
.stage.active svg { box-shadow: 0 0 0 3px rgba(44, 149, 131, .13); }
.stage.failed { color: #ad3944; }
.stage.failed svg { border-color: #df9098; background: #fff1f2; }
.failure-detail { margin: 0; font-size: 11px; line-height: 1.45; }
.compact { gap: 8px; padding: 9px 0 0; border: 0; border-top: 1px solid #e2ebe7; border-radius: 0; background: transparent; }
.compact .ingestion-summary { font-size: 11px; }
.compact .stage { font-size: 9px; }
.compact .stage svg { width: 21px; height: 21px; padding: 4px; }
@media (max-width: 640px) { .stage-track { overflow-x: auto; grid-template-columns: repeat(7, 55px); padding-bottom: 2px; }.stage-track::before, .stage-line { left: 27px; width: 330px; } }
</style>
