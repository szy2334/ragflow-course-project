<script setup lang="ts">
import { BookMarked, ExternalLink, MapPin } from 'lucide-vue-next'
import type { EvidenceItem } from '@/api/contracts'

defineProps<{ evidence: EvidenceItem; active?: boolean }>()
const emit = defineEmits<{ locate: [evidence: EvidenceItem] }>()
</script>

<template>
  <article class="evidence-card" :class="{ active }">
    <div class="evidence-top"><span><BookMarked :size="15" />{{ evidence.paper_title }}</span><span class="score">{{ Math.round((evidence.rerank_score ?? evidence.retrieval_score) * 100) }}%</span></div>
    <p class="evidence-quote">“{{ evidence.quote }}”</p>
    <footer><span><MapPin :size="13" />{{ evidence.section_title }} · 第 {{ evidence.page_number }} 页</span><button class="text-button" @click="emit('locate', evidence)">定位 <ExternalLink :size="13" /></button></footer>
  </article>
</template>
