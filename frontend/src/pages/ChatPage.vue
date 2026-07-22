<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { AlertTriangle, BookOpen, ChevronRight, CircleStop, MessageCircleQuestion, Plus, SearchCheck, Send, Sparkles, X } from 'lucide-vue-next'
import EvidenceCard from '@/components/EvidenceCard.vue'
import AnswerContext from '@/components/AnswerContext.vue'
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
const sending = ref(false)
const creatingSession = ref(false)
const error = ref('')
const activeTaskId = ref('')
const detail = ref<AnswerDetailView | null>(null)
const showPaperPicker = ref(false)
const chatScroll = ref<HTMLElement | null>(null)
const previewEvidence = ref<EvidenceItem | null>(null)
const previewPdfUrl = ref('')
const previewLoading = ref(false)
const previewError = ref('')
let previewRequestId = 0
const session = computed<ChatSessionView | undefined>(() => workspace.sessions.find((item) => item.session_id === props.sessionId))
const messages = computed(() => workspace.messagesBySession[props.sessionId] ?? [])
const papers = computed(() => Object.values(workspace.papersById))
const activeWorkflow = computed(() => activeTaskId.value ? workspace.workflows[activeTaskId.value] : null)
const latestStoredAnswer = computed(() => {
  const completedMessage = [...messages.value].reverse().find((message) => message.answer)
  return completedMessage?.answer ?? null
})
const activeEvidences = computed(() => activeWorkflow.value?.completedAnswer?.evidences ?? activeWorkflow.value?.evidences ?? detail.value?.answer.evidences ?? latestStoredAnswer.value?.evidences ?? [])
const readableEvidences = computed<EvidenceItem[]>(() => activeEvidences.value
  .map((evidence) => ({ ...evidence, quote: readableQuote(evidence.quote) }))
  .filter((evidence) => Boolean(evidence.quote)))
const displayedAnswer = computed(() => activeWorkflow.value?.completedAnswer ?? detail.value?.answer ?? latestStoredAnswer.value)
const workspaceLabel = '论文阅读工作台'
const questionPlaceholder = '例如：这个方法的核心创新是什么？'

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
function readableQuote(quote: string) {
  const normalized = quote.replace(/\s+/g, ' ').trim()
  const relatedText = normalized.lastIndexOf('相关正文:')
  if (relatedText >= 0) return normalized.slice(relatedText + '相关正文:'.length).trim()
  if (normalized.includes('公式:') || normalized.includes('图片标题:') || normalized.includes('图片内文字:') || normalized.includes('表格:')) return ''
  const bodyText = normalized.lastIndexOf('正文:')
  if (bodyText >= 0) return normalized.slice(bodyText + '正文:'.length).trim()
  return normalized.startsWith('章节:') ? '' : normalized
}
async function createReadingSession() {
  const paperIds = selectedPaperIds.value.length ? selectedPaperIds.value : session.value?.paper_ids ?? []
  if (!paperIds.length) {
    error.value = '请先选择至少一篇已就绪论文。'
    showPaperPicker.value = true
    return
  }
  creatingSession.value = true
  error.value = ''
  try {
    const created = await api.createSession({ paper_ids: paperIds, title: session.value?.title || undefined })
    await workspace.loadSessions()
    await router.push(`/chat/${created.session_id}`)
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法新建阅读会话。'
  } finally {
    creatingSession.value = false
  }
}
async function submit() {
  const normalized = question.value.trim()
  if (!normalized) return
  if (!selectedPaperIds.value.length) { error.value = '请至少选择一篇已就绪论文。'; showPaperPicker.value = true; return }
  const blocked = selectedPaperIds.value.some((id) => workspace.papersById[id]?.status !== 'ready')
  if (blocked) { error.value = '所选论文仍在解析或理解中，完成后才能提问。'; return }
  sending.value = true; error.value = ''
  try {
    workspace.appendMessage({ message_id: `local-${crypto.randomUUID()}`, session_id: props.sessionId, role: 'user', content: normalized, task_id: null, status: null, confidence: null, created_at: new Date().toISOString() })
    const task = await api.askQuestion(props.sessionId, { question: normalized, paper_ids: selectedPaperIds.value })
    activeTaskId.value = task.task_id
    workspace.startWorkflow(task)
    workspace.appendMessage({ message_id: task.resource_id ?? `pending-${task.task_id}`, session_id: props.sessionId, role: 'assistant', content: '', task_id: task.task_id, status: task.status, confidence: null, created_at: new Date().toISOString() })
    question.value = ''
    await scrollToEnd()
    void workspace.streamWorkflow(task.task_id).finally(async () => {
      const messageId = task.message_id ?? task.resource_id
      if (messageId) {
        try { detail.value = await api.getAnswerDetail(messageId) }
        catch { /* The message list still provides the completed answer if details are not ready yet. */ }
      }
      await workspace.loadMessages(props.sessionId)
      activeTaskId.value = ''
    })
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '问题没有提交成功，请重试。' }
  finally { sending.value = false }
}
async function stop() {
  if (!activeTaskId.value) return
  const messageId = activeWorkflow.value?.task.message_id ?? activeWorkflow.value?.task.resource_id
  if (!messageId) { error.value = '当前问题尚未获得可取消的消息标识。'; return }
  try { await api.cancelWorkflow(messageId, '用户主动停止') }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法停止当前任务。' }
}
async function inspectMessage(messageId: string) {
  try { detail.value = await api.getAnswerDetail(messageId) }
  catch { /* Pending or non-answer messages do not have a detail view. */ }
}
async function locate(evidence: EvidenceItem) {
  if (!evidence.paper_id) return
  const requestId = ++previewRequestId
  previewEvidence.value = evidence
  previewError.value = ''
  if (previewPdfUrl.value) URL.revokeObjectURL(previewPdfUrl.value)
  previewPdfUrl.value = ''
  previewLoading.value = true
  try {
    const file = await api.getPaperFile(evidence.paper_id)
    if (requestId !== previewRequestId) return
    if (!file.size) throw new Error('empty PDF response')
    previewPdfUrl.value = URL.createObjectURL(file)
  } catch (cause) {
    if (requestId === previewRequestId) previewError.value = cause instanceof ApiError ? cause.message : '无法加载论文原文。'
  } finally {
    if (requestId === previewRequestId) previewLoading.value = false
  }
}
function closePreview() {
  previewRequestId += 1
  previewEvidence.value = null
  previewLoading.value = false
  previewError.value = ''
  if (previewPdfUrl.value) URL.revokeObjectURL(previewPdfUrl.value)
  previewPdfUrl.value = ''
}
watch(() => props.sessionId, load)
onMounted(load)
onBeforeUnmount(closePreview)
</script>

<template>
  <section class="reading-workspace">
    <aside class="reading-left">
      <div class="workspace-brand-row"><div><p class="eyebrow">阅读会话</p><h1>{{ session?.title || '正在加载会话' }}</h1></div><button class="icon-button" aria-label="回到论文库" title="回到论文库" @click="router.push('/papers')"><X :size="17" /></button></div>
      <button class="new-session" :disabled="creatingSession" @click="createReadingSession"><Plus :size="16" />{{ creatingSession ? '正在新建…' : '新建阅读会话' }}</button>
      <nav class="session-list" aria-label="阅读会话列表"><button v-for="item in workspace.sessions" :key="item.session_id" :class="{ active: item.session_id === sessionId }" @click="router.push(`/chat/${item.session_id}`)"><MessageCircleQuestion :size="16" /><span>{{ item.title }}</span><ChevronRight :size="15" /></button></nav>
      <div class="workspace-papers"><button class="workspace-block-head" @click="showPaperPicker = !showPaperPicker"><span><BookOpen :size="16" />本次证据范围</span><span>{{ selectedPaperIds.length }} 篇</span></button><div v-if="showPaperPicker" class="paper-picker"><p>每次问答必须明确限定论文范围。</p><label v-for="paper in papers" :key="paper.paper_id" :class="{ disabled: paper.status !== 'ready' }"><input type="checkbox" :checked="selectedPaperIds.includes(paper.paper_id)" :disabled="paper.status !== 'ready'" @change="togglePaper(paper.paper_id)" /><span>{{ paper.title }}</span><StatusPill :status="paper.status" /></label></div></div>
    </aside>

    <main class="reading-center">
      <header class="reader-titlebar"><div><p class="eyebrow">{{ workspaceLabel }}</p><h2>基于本地论文证据的阅读与问答</h2></div><div class="scope-summary"><BookOpen :size="16" /><span>{{ selectedPaperIds.length }} 篇已选论文</span></div></header>
      <p v-if="error" class="inline-error reader-error" role="alert">{{ error }}</p>
      <div ref="chatScroll" class="message-stream">
        <div v-if="!messages.length && !activeWorkflow" class="chat-empty"><Sparkles :size="29" /><h3>从论文中开始一个问题</h3><p>聚焦研究问题、方法、实验与结论，并查看对应原文证据。</p></div>
        <article v-for="message in messages" :key="message.message_id" class="message" :class="message.role"><div class="message-avatar">{{ message.role === 'user' ? '你' : '知' }}</div><div class="message-body"><div class="message-meta"><span>{{ message.role === 'user' ? '你的问题' : '知阅助手' }}</span><span>{{ new Date(message.created_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) }}</span></div><p v-if="message.role === 'user'">{{ message.content }}</p><MarkdownContent v-else-if="message.content" :content="message.content" /><div v-else-if="message.task_id === activeTaskId" class="answer-pending"><span class="dot-loader"><i /><i /><i /></span>正在等待已核验的答案…</div><button v-if="message.role === 'assistant' && message.content" class="inspect-link" @click="inspectMessage(message.message_id)">查看引用与执行记录</button></div></article>
        <article v-if="activeWorkflow" class="message assistant live-answer"><div class="message-avatar">知</div><div class="message-body"><div class="message-meta"><span>知阅助手 · 实时回答</span><StatusPill :status="activeWorkflow.completedAnswer ? 'succeeded' : activeWorkflow.error ? 'failed' : 'running'" /></div><MarkdownContent v-if="activeWorkflow.text" :content="activeWorkflow.text" /><div v-else class="answer-pending"><span class="dot-loader"><i /><i /><i /></span>{{ activeWorkflow.phase }}</div><p v-if="activeWorkflow.completedAnswer?.is_refusal" class="refusal-note"><AlertTriangle :size="16" />{{ activeWorkflow.completedAnswer.refusal_reason || '现有论文证据不足，系统没有使用外部常识补答。' }}</p><p v-if="activeWorkflow.error" class="inline-error">{{ activeWorkflow.error }}</p></div></article>
      </div>
      <form class="question-box" @submit.prevent="submit"><textarea v-model="question" rows="2" :disabled="sending" :placeholder="questionPlaceholder" @keydown.ctrl.enter="submit" /><div class="question-actions"><span class="shortcut-hint">只检索当前选择的本地论文</span><span class="shortcut-hint">Ctrl + Enter 发送</span><button v-if="activeWorkflow && !activeWorkflow.completedAnswer && !activeWorkflow.error" type="button" class="stop-button" @click="stop"><CircleStop :size="17" />停止</button><button type="submit" class="send-button" :disabled="sending || !question.trim()"><Send :size="18" /><span>{{ sending ? '提交中' : '发送' }}</span></button></div></form>
      <article v-if="!activeWorkflow && displayedAnswer" class="message assistant persisted-answer"><div class="message-avatar">知</div><div class="message-body"><div class="message-meta"><span>知阅助手</span><span>{{ new Date(displayedAnswer.completed_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) }}</span></div><MarkdownContent :content="displayedAnswer.answer" /><button class="inspect-link" @click="inspectMessage(displayedAnswer.message_id)">查看引用与执行记录</button></div></article>
    </main>

    <aside class="reading-right">
      <WorkflowTimeline :events="activeWorkflow?.events || detail?.workflow_run ? (activeWorkflow?.events || []) : []" :phase="activeWorkflow?.phase || '尚未开始任务'" />
      <AnswerContext v-if="displayedAnswer" :answer="displayedAnswer" />
      <section class="evidence-panel"><div class="evidence-heading"><div><SearchCheck :size="17" /><strong>证据定位</strong></div><span>{{ readableEvidences.length }} 条</span></div><div v-if="readableEvidences.length" class="evidence-list"><EvidenceCard v-for="evidence in readableEvidences" :key="evidence.evidence_id" :evidence="evidence" @locate="locate" /></div><p v-else class="side-empty">{{ activeEvidences.length ? '本次回答没有可直接阅读的正文证据。' : '回答完成后，实际使用的原文证据会显示在这里。' }}</p></section>
    </aside>

    <div v-if="previewEvidence" class="evidence-preview-scrim" @click.self="closePreview">
      <section class="evidence-preview-modal" role="dialog" aria-modal="true" aria-label="论文原文预览">
        <header><div><p class="eyebrow">原文预览</p><strong>第 {{ previewEvidence.page_number || 1 }} 页</strong></div><button class="icon-button" aria-label="关闭原文预览" title="关闭" @click="closePreview"><X :size="18" /></button></header>
        <iframe v-if="previewPdfUrl" :src="`${previewPdfUrl}#page=${previewEvidence.page_number || 1}`" title="论文原文预览" />
        <div v-else class="evidence-preview-status"><span v-if="previewLoading">正在加载论文原文…</span><span v-else>{{ previewError || '论文原文暂不可用。' }}</span></div>
      </section>
    </div>
  </section>
</template>

<style scoped>
.evidence-preview-scrim { position: fixed; z-index: 80; inset: 0; display: grid; place-items: center; padding: 4vh 3vw; background: rgba(5, 27, 23, .58); backdrop-filter: blur(3px); }
.evidence-preview-modal { display: flex; width: min(1100px, 94vw); height: min(860px, 88dvh); flex-direction: column; overflow: hidden; border: 1px solid #bdd5cc; border-radius: 14px; background: white; box-shadow: 0 24px 70px rgba(3, 29, 25, .35); }
.evidence-preview-modal header { display: flex; min-height: 64px; align-items: center; justify-content: space-between; padding: 11px 14px 11px 18px; border-bottom: 1px solid var(--line); }
.evidence-preview-modal header p { margin: 0 0 2px; }
.evidence-preview-modal header strong { color: var(--ink); font-size: 14px; }
.evidence-preview-modal iframe { width: 100%; height: 100%; flex: 1; border: 0; background: #eef1f0; }
.evidence-preview-status { display: grid; min-height: 220px; flex: 1; place-items: center; padding: 24px; color: var(--ink-soft); text-align: center; }
@media (max-width: 720px) { .evidence-preview-scrim { padding: 2vh 2vw; }.evidence-preview-modal { width: 96vw; height: 94dvh; border-radius: 11px; } }
</style>
