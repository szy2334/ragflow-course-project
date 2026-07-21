import { defineStore } from 'pinia'
import { api } from '@/api'
import { demo } from '@/api/demo'
import { getAccessToken } from '@/api/http'
import type { AnswerView, ChatMessageView, ChatSessionView, EvidenceItem, PaperView, StreamEvent, TaskAccepted, TaskView } from '@/api/contracts'

interface LiveWorkflow {
  task: TaskAccepted
  lastSequence: number
  eventIds: Set<string>
  text: string
  evidences: EvidenceItem[]
  warnings: string[]
  phase: string
  error: string | null
  completedAnswer: AnswerView | null
  events: StreamEvent[]
}

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

export const useWorkspaceStore = defineStore('workspace', {
  state: () => ({
    papersById: {} as Record<string, PaperView>,
    paperTotal: 0,
    sessions: [] as ChatSessionView[],
    messagesBySession: {} as Record<string, ChatMessageView[]>,
    tasksById: {} as Record<string, TaskView>,
    workflows: {} as Record<string, LiveWorkflow>,
  }),
  actions: {
    async loadPapers(params: Record<string, unknown> = {}) {
      const page = await api.listPapers({ page: 1, page_size: 50, sort_by: 'updated_at', sort_order: 'desc', ...params })
      this.papersById = Object.fromEntries(page.items.map((paper) => [paper.paper_id, paper]))
      this.paperTotal = page.total
      return page
    },
    async refreshPaper(paperId: string) {
      const paper = await api.getPaper(paperId)
      this.papersById[paper.paper_id] = paper
      return paper
    },
    async pollTask(taskId: string, attempts = 60) {
      for (let count = 0; count < attempts; count += 1) {
        const task = await api.getTask(taskId)
        this.tasksById[task.task_id] = task
        if (['succeeded', 'failed', 'cancelled'].includes(task.status)) return task
        await wait(2_000)
      }
      return this.tasksById[taskId]
    },
    async loadSessions() {
      const page = await api.listSessions({ page: 1, page_size: 50, sort_order: 'desc' })
      this.sessions = page.items
      return page.items
    },
    async loadMessages(sessionId: string) {
      const page = await api.listMessages(sessionId, { page: 1, page_size: 100 })
      this.messagesBySession[sessionId] = page.items.slice().reverse()
      return this.messagesBySession[sessionId]
    },
    appendMessage(message: ChatMessageView) {
      const messages = this.messagesBySession[message.session_id] ?? []
      if (!messages.some((item) => item.message_id === message.message_id)) messages.push(message)
      this.messagesBySession[message.session_id] = messages
    },
    startWorkflow(task: TaskAccepted) {
      this.workflows[task.task_id] = {
        task, lastSequence: 0, eventIds: new Set(), text: '', evidences: [], warnings: [], phase: '正在连接工作流', error: null, completedAnswer: null, events: [],
      }
    },
    applyEvent(taskId: string, event: StreamEvent) {
      const workflow = this.workflows[taskId]
      if (!workflow || workflow.eventIds.has(event.event_id) || event.sequence <= workflow.lastSequence) return
      workflow.eventIds.add(event.event_id)
      workflow.lastSequence = event.sequence
      workflow.events.push(event)
      const eventData = event.data as {
        delta?: string; evidence?: EvidenceItem; answer?: AnswerView; error?: { message?: string }; message?: string; label?: string; stage?: string; warnings?: string[]
      }
      if (event.event_type === 'status') workflow.phase = eventData.label ?? eventData.stage ?? '正在处理问题'
      if (event.event_type === 'citation' && eventData.evidence) workflow.evidences.push(eventData.evidence)
      if (event.event_type === 'delta') {
        workflow.phase = '正在生成回答，待最终校验'
        workflow.text += eventData.delta ?? ''
      }
      if (event.event_type === 'final' && eventData.answer) {
        workflow.completedAnswer = eventData.answer
        workflow.text = eventData.answer.answer
        workflow.evidences = eventData.answer.evidences
        workflow.warnings = eventData.answer.warnings
        workflow.phase = eventData.answer.is_refusal ? '已说明证据不足' : '回答已完成'
      }
      if (event.event_type === 'error') {
        workflow.error = eventData.error?.message ?? eventData.message ?? '工作流未能完成。'
        workflow.phase = '工作流已结束'
      }
    },
    async streamWorkflow(taskId: string) {
      const workflow = this.workflows[taskId]
      if (!workflow) return
      if (demo.active()) {
        const emit = (sequence: number, event_type: StreamEvent['event_type'], data: Record<string, unknown>) => this.applyEvent(taskId, {
          event_id: `demo-${taskId}-${sequence}`, event_type, task_id: taskId, message_id: demo.answer.message_id, session_id: demo.answer.session_id, agent_name: null, sequence, timestamp: new Date().toISOString(), data,
        })
        emit(1, 'status', { stage: 'routing', label: '正在规划阅读任务' })
        await wait(280)
        emit(2, 'status', { stage: 'retrieving', label: '正在检索论文证据' })
        await wait(280)
        emit(3, 'citation', { evidence: demo.evidence })
        emit(4, 'status', { stage: 'synthesizing', label: '正在汇总已核验证据' })
        await wait(280)
        emit(5, 'delta', { message_id: demo.answer.message_id, delta: '正在依据实验章节的原文证据组织回答…\n\n' })
        await wait(280)
        emit(6, 'final', { answer: demo.answer })
        return
      }
      if (!workflow.task.stream_url) {
        workflow.phase = '任务已提交，正在等待处理结果'
        const task = await this.pollTask(taskId, 60)
        if (task?.status === 'failed' || task?.status === 'cancelled') workflow.error = task.error?.message ?? '任务已停止。'
        else if (task?.status === 'succeeded') workflow.phase = '任务已完成，请查看结果'
        return
      }
      let reconnects = 0
      while (reconnects < 3 && !workflow.completedAnswer && !workflow.error) {
        try {
          const after = workflow.lastSequence ? `?after_sequence=${workflow.lastSequence}` : ''
          const eventIds = [...workflow.eventIds]
          const response = await fetch(`${workflow.task.stream_url}${after}`, {
            headers: { Authorization: `Bearer ${getAccessToken() ?? ''}`, 'Last-Event-ID': eventIds[eventIds.length - 1] ?? '' },
            credentials: 'include',
          })
          if (!response.ok || !response.body) throw new Error('SSE 连接不可用')
          const reader = response.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ''
          while (true) {
            const { done, value } = await reader.read()
            buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
            const chunks = buffer.split(/\r?\n\r?\n/)
            buffer = chunks.pop() ?? ''
            for (const chunk of chunks) {
              const payload = chunk.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('')
              if (!payload) continue
              this.applyEvent(taskId, JSON.parse(payload) as StreamEvent)
            }
            if (done) break
          }
          if (workflow.completedAnswer || workflow.error) return
        } catch {
          reconnects += 1
          workflow.phase = `连接中断，正在恢复（${reconnects}/3）`
          await wait(1_000 * reconnects)
        }
      }
      if (!workflow.completedAnswer && !workflow.error) {
        workflow.phase = '实时连接不可用，正在查询任务状态'
        const task = await this.pollTask(taskId, 30)
        if (task?.status === 'failed' || task?.status === 'cancelled') workflow.error = task.error?.message ?? '工作流已停止。'
        else workflow.phase = '任务完成，请刷新会话查看最终答案'
      }
    },
  },
})
