<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { AlertTriangle, BookOpen, CheckCircle2, ChevronRight, CircleStop, MessageCircleQuestion, Plus, SearchCheck, Send, Sparkles, X } from 'lucide-vue-next'
import EvidenceCard from '@/components/EvidenceCard.vue'
import MarkdownContent from '@/components/MarkdownContent.vue'
import StatusPill from '@/components/StatusPill.vue'
import WorkflowTimeline from '@/components/WorkflowTimeline.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { AnswerDetailView, ChatSessionView, EvidenceItem } from '@/api/contracts'
import { useWorkspaceStore } from '@/stores/workspace'

const props = defineProps<{ sessionId: string }>()
const router = useRouter()
const workspace = useWorkspaceStore()
const question = ref('')
const selectedPaperIds = ref<string[]>([])
const criticalReview = ref(false)
const sending = ref(false)
const error = ref('')
const activeTaskId = ref('')
const detail = ref<AnswerDetailView | null>(null)
const showPaperPicker = ref(false)
const chatScroll = ref<HTMLElement | null>(null)
const session = computed<ChatSessionView | undefined>(() => workspace.sessions.find((item) => item.session_id === props.sessionId))
const messages = computed(() => workspace.messagesBySession[props.sessionId] ?? [])
const papers = computed(() => Object.values(workspace.papersById))
const activeWorkflow = computed(() => activeTaskId.value ? workspace.workflows[activeTaskId.value] : null)
const activeEvidences = computed(() => activeWorkflow.value?.completedAnswer?.evidences ?? activeWorkflow.value?.evidences ?? [])
const activeClaims = computed(() => activeWorkflow.value?.completedAnswer?.claims ?? [])

async function load() {
  error.value = ''
  try {
    await Promise.all([workspace.loadPapers(), workspace.loadSessions(), workspace.loadMessages(props.sessionId)])
    selectedPaperIds.value = session.value?.paper_ids ?? []
    await scrollToEnd()
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法加载这个阅读会话。' }
}
async function scrollToEnd() { await nextTick(); chatScroll.value?.scrollTo({ top: chatScroll.value.scrollHeight, behavior: 'smooth' }) }
function togglePaper(paperId: string) {
  if (selectedPaperIds.value.includes(paperId)) selectedPaperIds.value = selectedPaperIds.value.filter((id) => id !== paperId)
  else if (selectedPaperIds.value.length < 10) selectedPaperIds.value.push(paperId)
  else error.value = '一次联合问答最多选择 10 篇论文。'
}
async function submit() {
  const normalized = question.value.trim()
  if (!normalized) return
  if (!selectedPaperIds.value.length) { error.value = '请至少选择一篇已就绪论文。'; showPaperPicker.value = true; return }
  const blocked = selectedPaperIds.value.some((id) => workspace.papersById[id]?.status !== 'ready')
  if (blocked) { error.value = '所选论文仍在解析或索引，完成后才能提问。'; return }
  sending.value = true; error.value = ''
  try {
    workspace.appendMessage({ message_id: `local-${crypto.randomUUID()}`, session_id: props.sessionId, role: 'user', content: normalized, task_id: null, status: null, confidence: null, created_at: new Date().toISOString() })
    const task = await api.askQuestion(props.sessionId, { question: normalized, paper_ids: selectedPaperIds.value, enable_critical_review: criticalReview.value })
    activeTaskId.value = task.task_id
    workspace.startWorkflow(task)
    workspace.appendMessage({ message_id: task.resource_id ?? `pending-${task.task_id}`, session_id: props.sessionId, role: 'assistant', content: '', task_id: task.task_id, status: task.status, confidence: null, created_at: new Date().toISOString() })
    question.value = ''
    await scrollToEnd()
    void workspace.streamWorkflow(task.task_id).finally(async () => {
      await workspace.loadMessages(props.sessionId)
      activeTaskId.value = ''
    })
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '问题没有提交成功，请重试。' }
  finally { sending.value = false }
}
async function stop() {
  if (!activeTaskId.value) return
  try { await api.cancelWorkflow(activeTaskId.value, '用户主动停止') }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法停止当前任务。' }
}
async function inspectMessage(messageId: string) {
  try { detail.value = await api.getAnswerDetail(messageId) }
  catch { /* Pending or non-answer messages do not have a detail view. */ }
}
function locate(evidence: EvidenceItem) { router.push({ path: `/papers/${evidence.paper_id}`, query: { page: evidence.page_number } }) }
watch(() => props.sessionId, load)
onMounted(load)
</script>

<template>
  <section class="reading-workspace">
    <aside class="reading-left">
      <div class="workspace-brand-row"><div><p class="eyebrow">阅读会话</p><h1>{{ session?.title || '正在加载会话' }}</h1></div><button class="icon-button" aria-label="回到论文库" title="回到论文库" @click="router.push('/papers')"><X :size="17" /></button></div>
      <button class="new-session" @click="router.push('/papers')"><Plus :size="16" />新建阅读会话</button>
      <nav class="session-list" aria-label="阅读会话列表"><button v-for="item in workspace.sessions" :key="item.session_id" :class="{ active: item.session_id === sessionId }" @click="router.push(`/chat/${item.session_id}`)"><MessageCircleQuestion :size="16" /><span>{{ item.title }}</span><ChevronRight :size="15" /></button></nav>
      <div class="workspace-papers"><button class="workspace-block-head" @click="showPaperPicker = !showPaperPicker"><span><BookOpen :size="16" />本次证据范围</span><span>{{ selectedPaperIds.length }} 篇</span></button><div v-if="showPaperPicker" class="paper-picker"><p>每次问答必须明确限定论文范围。</p><label v-for="paper in papers" :key="paper.paper_id" :class="{ disabled: paper.status !== 'ready' }"><input type="checkbox" :checked="selectedPaperIds.includes(paper.paper_id)" :disabled="paper.status !== 'ready'" @change="togglePaper(paper.paper_id)" /><span>{{ paper.title }}</span><StatusPill :status="paper.status" /></label></div></div>
    </aside>

    <main class="reading-center">
      <header class="reader-titlebar"><div><p class="eyebrow">智能阅读工作台</p><h2>问题与证据在同一处对齐</h2></div><div class="scope-summary"><BookOpen :size="16" /><span>{{ selectedPaperIds.length }} 篇已选论文</span></div></header>
      <p v-if="error" class="inline-error reader-error" role="alert">{{ error }}</p>
      <div ref="chatScroll" class="message-stream">
        <div v-if="!messages.length && !activeWorkflow" class="chat-empty"><Sparkles :size="29" /><h3>从论文中开始一个问题</h3><p>例如：这个方法的核心创新是什么？主要实验是否充分支持作者的结论？</p></div>
        <article v-for="message in messages" :key="message.message_id" class="message" :class="message.role"><div class="message-avatar">{{ message.role === 'user' ? '你' : '知' }}</div><div class="message-body"><div class="message-meta"><span>{{ message.role === 'user' ? '你的问题' : '知阅助手' }}</span><span>{{ new Date(message.created_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) }}</span></div><p v-if="message.role === 'user'">{{ message.content }}</p><MarkdownContent v-else-if="message.content" :content="message.content" /><div v-else-if="message.task_id === activeTaskId" class="answer-pending"><span class="dot-loader"><i /><i /><i /></span>正在等待已核验的答案…</div><button v-if="message.role === 'assistant' && message.content" class="inspect-link" @click="inspectMessage(message.message_id)">查看引用与执行记录</button></div></article>
        <article v-if="activeWorkflow" class="message assistant live-answer"><div class="message-avatar">知</div><div class="message-body"><div class="message-meta"><span>知阅助手 · 实时回答</span><StatusPill :status="activeWorkflow.completedAnswer ? 'succeeded' : activeWorkflow.error ? 'failed' : 'running'" /></div><MarkdownContent v-if="activeWorkflow.text" :content="activeWorkflow.text" /><div v-else class="answer-pending"><span class="dot-loader"><i /><i /><i /></span>{{ activeWorkflow.phase }}</div><p v-if="activeWorkflow.completedAnswer?.is_refusal" class="refusal-note"><AlertTriangle :size="16" />{{ activeWorkflow.completedAnswer.refusal_reason || '现有论文证据不足，系统没有使用外部常识补答。' }}</p><p v-if="activeWorkflow.error" class="inline-error">{{ activeWorkflow.error }}</p></div></article>
      </div>
      <form class="question-box" @submit.prevent="submit"><textarea v-model="question" rows="2" :disabled="sending" placeholder="针对已选论文提问；答案将只依据检索到的原文证据…" @keydown.ctrl.enter="submit" /><div class="question-actions"><label class="toggle-option"><input v-model="criticalReview" type="checkbox" /><span>启用批判性审阅</span></label><span class="shortcut-hint">Ctrl + Enter 发送</span><button v-if="activeWorkflow && !activeWorkflow.completedAnswer && !activeWorkflow.error" type="button" class="stop-button" @click="stop"><CircleStop :size="17" />停止</button><button type="submit" class="send-button" :disabled="sending || !question.trim()"><Send :size="18" /><span>{{ sending ? '提交中' : '发送' }}</span></button></div></form>
    </main>

    <aside class="reading-right">
      <WorkflowTimeline :events="activeWorkflow?.events || detail?.workflow_run ? (activeWorkflow?.events || []) : []" :phase="activeWorkflow?.phase || '尚未开始任务'" />
      <section class="evidence-panel"><div class="evidence-heading"><div><SearchCheck :size="17" /><strong>证据定位</strong></div><span>{{ activeEvidences.length }} 条</span></div><div v-if="activeEvidences.length" class="evidence-list"><EvidenceCard v-for="evidence in activeEvidences" :key="evidence.evidence_id" :evidence="evidence" @locate="locate" /></div><p v-else class="side-empty">回答完成后，实际使用的原文证据会显示在这里。</p></section>
      <section v-if="activeClaims.length" class="claims-panel"><div class="evidence-heading"><div><CheckCircle2 :size="17" /><strong>已核验结论</strong></div></div><article v-for="claim in activeClaims" :key="claim.claim_id" class="claim-row"><span :class="`verdict-${claim.verdict}`">{{ claim.verdict === 'supported' ? '支持' : claim.verdict === 'refuted' ? '反驳' : claim.verdict === 'conflicting_evidence' ? '冲突' : '不足' }}</span><p>{{ claim.text }}</p><small>{{ Math.round(claim.confidence * 100) }}% · {{ claim.reason }}</small></article></section>
    </aside>
  </section>
</template>
