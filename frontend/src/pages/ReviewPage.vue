<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight, FileSearch, Scale, ShieldCheck } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import { useWorkspaceStore } from '@/stores/workspace'

const router = useRouter()
const workspace = useWorkspaceStore()
const loading = ref(true)
const error = ref('')
const startingPaperId = ref('')
const readyPapers = computed(() => Object.values(workspace.papersById).filter((paper) => paper.status === 'ready'))

async function load() {
  loading.value = true
  error.value = ''
  try { await workspace.loadPapers({ status: 'ready' }) }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法读取可审阅论文。' }
  finally { loading.value = false }
}

async function startReview(paperId: string) {
  const paper = workspace.papersById[paperId]
  if (!paper) return
  startingPaperId.value = paperId
  error.value = ''
  try {
    const session = await api.createSession({ title: `审阅：${paper.title}`, paper_ids: [paper.paper_id] })
    router.push({ path: `/chat/${session.session_id}`, query: { mode: 'review' } })
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法创建审阅会话。' }
  finally { startingPaperId.value = '' }
}

onMounted(load)
</script>

<template>
  <section class="page review-page">
    <PageHeader eyebrow="论文审阅" title="审论文" description="选择一篇已理解论文开始证据化审阅。">
      <button class="secondary-button" @click="router.push('/papers')"><FileSearch :size="18" />返回阅读</button>
    </PageHeader>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <div v-if="loading" class="skeleton-list"><div v-for="item in 3" :key="item" class="skeleton-row" /></div>
    <section v-else-if="readyPapers.length" class="review-paper-list" aria-label="可审阅论文">
      <article v-for="paper in readyPapers" :key="paper.paper_id" class="review-paper-row">
        <div class="review-paper-icon"><Scale :size="21" /></div>
        <div class="review-paper-copy"><div><StatusPill :status="paper.status" /><span>本地原文已就绪</span></div><h2>{{ paper.title }}</h2><p>{{ paper.understanding?.paper_summary || '论文理解结果正在同步。' }}</p></div>
        <button class="primary-button" :disabled="Boolean(startingPaperId)" @click="startReview(paper.paper_id)"><ShieldCheck :size="17" />{{ startingPaperId === paper.paper_id ? '正在打开…' : '开始审阅' }}<ArrowRight :size="16" /></button>
      </article>
    </section>
    <div v-else class="empty-state"><Scale :size="34" /><h2>没有可审阅论文</h2><button class="primary-button" @click="router.push('/papers')">去读论文</button></div>
  </section>
</template>

<style scoped>
.review-paper-list { display: grid; gap: 10px; }
.review-paper-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 16px; align-items: center; padding: 18px; border: 1px solid var(--line); border-radius: var(--radius); background: white; }
.review-paper-icon { display: grid; width: 42px; height: 42px; place-items: center; color: #8b4b17; background: #fff1e5; border-radius: 7px; }.review-paper-copy { min-width: 0; }.review-paper-copy > div { display: flex; gap: 8px; align-items: center; color: var(--ink-faint); font-size: 12px; }.review-paper-copy h2 { margin: 8px 0 4px; color: var(--ink); font-size: 16px; }.review-paper-copy p { margin: 0; color: var(--ink-soft); font-size: 13px; line-height: 1.55; }
@media (max-width: 720px) { .review-paper-row { grid-template-columns: auto 1fr; }.review-paper-row .primary-button { grid-column: 1 / -1; width: 100%; justify-content: center; } }
</style>
