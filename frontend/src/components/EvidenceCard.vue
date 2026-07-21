<script setup lang="ts">
import { computed } from 'vue'
import { BookMarked, MapPin } from 'lucide-vue-next'
import type { EvidenceItem } from '@/api/contracts'

const props = defineProps<{ evidence: EvidenceItem }>()
defineEmits<{ locate: [evidence: EvidenceItem] }>()
const sourceLocation = computed(() => `${props.evidence.section_title || '论文正文'}${props.evidence.page_number ? ` · 第 ${props.evidence.page_number} 页` : ''}`)
</script>

<template>
  <article class="evidence-card">
    <div class="evidence-top"><span><BookMarked :size="15" />{{ sourceLocation }}</span><span class="score">{{ Math.round(evidence.retrieval_score * 100) }}%</span></div>
    <p class="evidence-quote">{{ evidence.quote }}</p>
    <footer><span><MapPin :size="12" />{{ evidence.chunk_id }}</span><button class="text-button" @click="$emit('locate', evidence)">查看原文</button></footer>
  </article>
</template>
