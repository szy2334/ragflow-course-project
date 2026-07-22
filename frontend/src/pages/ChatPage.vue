<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { AlertTriangle, BookOpen, ChevronRight, CircleStop, MessageCircleQuestion, Plus, SearchCheck, Send, Sparkles, Trash2, X } from 'lucide-vue-next'
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
const deletingSessionId = ref('')
const stopping = ref(false)
const error = ref('')
const activeTaskId = ref('')
const detail = ref<AnswerDetailView | null>(null)
const showDetail = ref(false)
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
const workflowRunning = computed(() => Boolean(activeTaskId.value && activeWorkflow.value && !activeWorkflow.value.completedAnswer && !activeWorkflow.value.error))
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
    const created = await api.createSession({ paper_ids: paperIds })
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
  if (!normalized || workflowRunning.value) return
  if (!selectedPaperIds.value.length) { error.value = '请至少选择一篇已就绪论文。'; showPaperPicker.value = true; return }
  const blocked = selectedPaperIds.value.some((id) => workspace.papersById[id]?.status !== 'ready')
  if (blocked) { error.value = '所选论文仍在解析或理解中，完成后才能提问。'; return }
  sending.value = true; error.value = ''
  try {
    const localMessageId = `local-${crypto.randomUUID()}`
    workspace.appendMessage({ message_id: localMessageId, session_id: props.sessionId, role: 'user', content: normalized, task_id: null, status: null, confidence: null, created_at: new Date().toISOString() })
    const task = await api.askQuestion(props.sessionId, { question: normalized, paper_ids: selectedPaperIds.value })
    activeTaskId.value = task.task_id
    workspace.startWorkflow(task)
    workspace.updateMessage(props.sessionId, localMessageId, { task_id: task.task_id, status: task.status })
    question.value = ''
    await scrollToEnd()
    void workspace.streamWorkflow(task.task_id).finally(async () => {
      await workspace.loadMessages(props.sessionId)
      activeTaskId.value = ''
      await scrollToEnd()
    })
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '问题没有提交成功，请重试。' }
  finally { sending.value = false }
}
async function stop() {
  if (!activeTaskId.value || stopping.value) return
  const taskId = activeTaskId.value
  const messageId = activeWorkflow.value?.task.message_id ?? activeWorkflow.value?.task.resource_id
  if (!messageId) { error.value = '当前问题尚未获得可取消的消息标识。'; return }
  stopping.value = true
  try {
    await workspace.cancelWorkflow(taskId, messageId)
    workspace.removeMessageByTaskId(props.sessionId, taskId)
    await workspace.pollTask(taskId, 15)
    await workspace.loadMessages(props.sessionId)
    error.value = ''
  }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法停止当前任务。' }
  finally { stopping.value = false }
}
async function inspectMessage(messageId: string) {
  try { detail.value = await api.getAnswerDetail(messageId); showDetail.value = true }
  catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法读取这条回答的引用与执行记录。' }
}
async function deleteSession(sessionId: string) {
  if (deletingSessionId.value || !window.confirm('删除此会话及其问答记录？此操作无法撤销。')) return
  deletingSessionId.value = sessionId
  error.value = ''
  try {
    await api.deleteSession(sessionId)
    workspace.removeSession(sessionId)
    if (sessionId === props.sessionId) await router.replace('/papers')
  } catch (cause) { error.value = cause instanceof ApiError ? cause.message : '无法删除会话。' }
  finally { deletingSessionId.value = '' }
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
      <nav class="session-list" aria-label="阅读会话列表"><div v-for="item in workspace.sessions" :key="item.session_id" class="session-row" :class="{ active: item.session_id === sessionId }"><button class="session-link" @click="router.push(`/chat/${item.session_id}`)"><MessageCircleQuestion :size="16" /><span>{{ item.title }}</span><ChevronRight :size="15" /></button><button class="session-delete" :aria-label="`删除会话：${item.title}`" title="删除会话" :disabled="deletingSessionId === item.session_id" @click="deleteSession(item.session_id)"><Trash2 :size="14" /></button></div></nav>
      <div class="workspace-papers"><button class="workspace-block-head" @click="showPaperPicker = !showPaperPicker"><span><BookOpen :size="16" />本次证据范围</span><span>{{ selectedPaperIds.length }} 篇</span></button><div v-if="showPaperPicker" class="paper-picker"><p>每次问答必须明确限定论文范围。</p><label v-for="paper in papers" :key="paper.paper_id" :class="{ disabled: paper.status !== 'ready' }"><input type="checkbox" :checked="selectedPaperIds.includes(paper.paper_id)" :disabled="paper.status !== 'ready'" @change="togglePaper(paper.paper_id)" /><span>{{ paper.title }}</span><StatusPill :status="paper.status" /></label></div></div>
    </aside>

    <main class="reading-center">
      <header class="reader-titlebar"><div><p class="eyebrow">{{ workspaceLabel }}</p><h2>基于本地论文证据的阅读与问答</h2></div><div class="scope-summary"><BookOpen :size="16" /><span>{{ selectedPaperIds.length }} 篇已选论文</span></div></header>
      <p v-if="error" class="inline-error reader-error" role="alert">{{ error }}</p>
      <div ref="chatScroll" class="message-stream">
        <div v-if="!messages.length && !activeWorkflow" class="chat-empty"><Sparkles :size="29" /><h3>从论文中开始一个问题</h3><p>聚焦研究问题、方法、实验与结论，并查看对应原文证据。</p></div>
        <template v-for="message in messages" :key="message.message_id">
          <article class="message user"><div class="message-avatar">你</div><div class="message-body"><div class="message-meta"><span>你的问题</span><span>{{ new Date(message.created_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) }}</span></div><p>{{ message.content }}</p></div></article>
          <article v-if="message.task_id === activeTaskId && activeWorkflow" class="message assistant live-answer"><div class="message-avatar">知</div><div class="message-body"><div class="message-meta"><span>知阅助手 · 实时回答</span><StatusPill :status="activeWorkflow.completedAnswer ? 'succeeded' : activeWorkflow.error ? 'failed' : 'running'" /></div><MarkdownContent v-if="activeWorkflow.text" :content="activeWorkflow.text" /><div v-else class="answer-pending"><span class="dot-loader"><i /><i /><i /></span>{{ activeWorkflow.phase }}</div><p v-if="activeWorkflow.completedAnswer?.is_refusal" class="refusal-note"><AlertTriangle :size="16" />{{ activeWorkflow.completedAnswer.refusal_reason || '现有论文证据不足，系统没有使用外部常识补答。' }}</p><p v-if="activeWorkflow.error" class="inline-error">{{ activeWorkflow.error }}</p></div></article>
          <article v-else-if="message.answer" class="message assistant"><div class="message-avatar">知</div><div class="message-body"><div class="message-meta"><span>知阅助手</span><span>{{ new Date(message.answer.completed_at).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }) }}</span></div><MarkdownContent :content="message.answer.answer" /><button class="inspect-link" @click="inspectMessage(message.answer.message_id)">查看引用与执行记录</button></div></article>
        </template>
      </div>
      <form class="question-box" @submit.prevent="submit"><textarea v-model="question" rows="2" :disabled="sending || workflowRunning" :placeholder="questionPlaceholder" @keydown.ctrl.enter="submit" /><div class="question-actions"><span class="shortcut-hint">只检索当前选择的本地论文</span><span class="shortcut-hint">Ctrl + Enter 发送</span><button v-if="workflowRunning" type="button" class="stop-button" :disabled="stopping" @click="stop"><CircleStop :size="17" />{{ stopping ? '正在停止' : '停止' }}</button><button type="submit" class="send-button" :disabled="sending || workflowRunning || !question.trim()"><Send :size="18" /><span>{{ sending ? '提交中' : '发送' }}</span></button></div></form>
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

    <div v-if="showDetail && detail" class="answer-detail-scrim" @click.self="showDetail = false">
      <section class="answer-detail-modal" role="dialog" aria-modal="true" aria-label="引用与执行记录">
        <header><div><p class="eyebrow">回答溯源</p><strong>引用与执行记录</strong></div><button class="icon-button" aria-label="关闭引用与执行记录" title="关闭" @click="showDetail = false"><X :size="18" /></button></header>
        <div class="answer-detail-content">
          <section><h3>引用</h3><div v-if="detail.answer.evidences.length" class="detail-evidence-list"><EvidenceCard v-for="evidence in detail.answer.evidences" :key="evidence.evidence_id" :evidence="evidence" @locate="locate" /></div><p v-else class="side-empty">这条回答没有可展示的论文引用。</p></section>
          <section><h3>执行记录</h3><div v-if="detail.agent_results.length" class="agent-result-list"><article v-for="result in detail.agent_results" :key="result.agent_name" class="agent-result"><div><strong>{{ result.agent_name }}</strong><StatusPill :status="result.status" /></div><p>{{ result.summary }}</p><small>耗时 {{ result.metrics.latency_ms }} ms · 输出 {{ result.metrics.output_tokens }} tokens</small></article></div><p v-else class="side-empty">此次执行未返回可展示的智能体节点记录。</p></section>
        </div>
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
.answer-detail-scrim { position: fixed; z-index: 81; inset: 0; display: grid; place-items: center; padding: 4vh 3vw; background: rgba(5, 27, 23, .58); backdrop-filter: blur(3px); }.answer-detail-modal { display: flex; width: min(820px, 94vw); max-height: min(760px, 88dvh); flex-direction: column; overflow: hidden; border: 1px solid #bdd5cc; border-radius: 14px; background: white; box-shadow: 0 24px 70px rgba(3, 29, 25, .35); }.answer-detail-modal header { display: flex; min-height: 64px; align-items: center; justify-content: space-between; padding: 11px 14px 11px 18px; border-bottom: 1px solid var(--line); }.answer-detail-modal header p { margin: 0 0 2px; }.answer-detail-modal header strong { color: var(--ink); font-size: 14px; }.answer-detail-content { display: grid; gap: 24px; padding: 20px; overflow-y: auto; }.answer-detail-content h3 { margin: 0 0 10px; color: var(--ink); font-size: 14px; }.detail-evidence-list, .agent-result-list { display: grid; gap: 9px; }.agent-result { padding: 12px; border: 1px solid var(--line); border-radius: 8px; }.agent-result > div { display: flex; align-items: center; justify-content: space-between; }.agent-result strong { color: var(--ink); font-size: 13px; }.agent-result p { margin: 8px 0 5px; color: var(--ink-soft); font-size: 12px; line-height: 1.55; }.agent-result small { color: var(--ink-faint); font-size: 11px; }
@media (max-width: 720px) { .evidence-preview-scrim { padding: 2vh 2vw; }.evidence-preview-modal { width: 96vw; height: 94dvh; border-radius: 11px; } }
</style>
