<script setup lang="ts">
import { FileText } from 'lucide-vue-next'
import type { EvidenceItem } from '@/api/contracts'

defineProps<{ evidence: EvidenceItem }>()
defineEmits<{ locate: [evidence: EvidenceItem] }>()
</script>

<template>
  <article class="evidence-card">
    <div class="evidence-top"><span><FileText :size="14" />原文摘录</span><span v-if="evidence.page_number">第 {{ evidence.page_number }} 页</span></div>
    <p class="evidence-quote">{{ evidence.quote }}</p>
    <footer><button class="text-button" @click="$emit('locate', evidence)">查看原文</button></footer>
  </article>
</template>

<style scoped>
.evidence-card { padding: 11px; border: 1px solid #e0e8e4; border-radius: 9px; background: #fcfefd; }
.evidence-card:hover { border-color: #a6cabc; }
.evidence-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--ink-faint); font-size: 10px; }
.evidence-top span { display: inline-flex; align-items: center; gap: 4px; }
.evidence-quote { display: -webkit-box; overflow: hidden; margin: 8px 0 10px; color: var(--ink-soft); font-family: Georgia, "Noto Serif SC", serif; font-size: 11px; line-height: 1.6; -webkit-box-orient: vertical; -webkit-line-clamp: 4; }
.evidence-card footer { display: flex; justify-content: flex-end; }
.evidence-card footer .text-button { font-size: 11px; }
</style>
