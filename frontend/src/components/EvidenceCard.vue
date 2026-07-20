<script setup lang="ts">
import { BookMarked, ExternalLink, MapPin, Scale } from 'lucide-vue-next'
import { computed } from 'vue'
import type { EvidenceItem } from '@/api/contracts'

const { evidence, active } = defineProps<{ evidence: EvidenceItem; active?: boolean }>()
const emit = defineEmits<{ locate: [evidence: EvidenceItem] }>()
const isStandard = computed(() => evidence.source_type === 'standard')
const isReferencePaper = computed(() => evidence.metadata?.evidence_role === 'reference_paper')
const sourceName = computed(() => isStandard.value ? String(evidence.metadata?.name ?? evidence.standard_name ?? evidence.paper_title ?? '参考论文') : evidence.paper_title ?? '上传论文')
const sourceLocation = computed(() => isReferencePaper.value ? '固定参考论文库' : isStandard.value ? `标准 ${evidence.standard_version ?? '未标注版本'}` : `${evidence.section_title} · 第 ${evidence.page_number} 页`)
</script>

<template>
  <article class="evidence-card" :class="{ active }">
    <div class="evidence-top"><span><component :is="isStandard ? Scale : BookMarked" :size="15" />{{ sourceName }}</span><span class="score">{{ Math.round((evidence.rerank_score ?? evidence.retrieval_score) * 100) }}%</span></div>
    <p class="evidence-quote">“{{ evidence.quote }}”</p>
    <footer><span><MapPin :size="13" />{{ sourceLocation }}</span><button v-if="evidence.paper_id" class="text-button" @click="emit('locate', evidence)">定位 <ExternalLink :size="13" /></button></footer>
  </article>
</template>
