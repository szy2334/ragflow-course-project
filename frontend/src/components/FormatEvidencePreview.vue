<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { X } from 'lucide-vue-next'
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy } from 'pdfjs-dist'
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { api } from '@/api'

GlobalWorkerOptions.workerSrc = workerSrc

const props = defineProps<{
  paperId: string
  page: number
  bbox: [number, number, number, number]
  rotation?: number
  aspect: string
}>()
const emit = defineEmits<{ close: [] }>()
const canvas = ref<HTMLCanvasElement | null>(null)
const loading = ref(true)
const error = ref('')
const box = ref({ left: 0, top: 0, width: 0, height: 0 })
let documentProxy: PDFDocumentProxy | null = null

const boxStyle = computed(() => ({
  left: `${box.value.left * 100}%`,
  top: `${box.value.top * 100}%`,
  width: `${box.value.width * 100}%`,
  height: `${box.value.height * 100}%`,
}))

function rotateBox(
  bbox: [number, number, number, number],
  width: number,
  height: number,
  rotation: number,
) {
  const [x0, y0, x1, y1] = bbox
  const normalized = ((rotation % 360) + 360) % 360
  if (normalized === 90) return { left: (height - y1) / height, top: x0 / width, width: (y1 - y0) / height, height: (x1 - x0) / width }
  if (normalized === 180) return { left: (width - x1) / width, top: (height - y1) / height, width: (x1 - x0) / width, height: (y1 - y0) / height }
  if (normalized === 270) return { left: y0 / height, top: (width - x1) / width, width: (y1 - y0) / height, height: (x1 - x0) / width }
  return { left: x0 / width, top: y0 / height, width: (x1 - x0) / width, height: (y1 - y0) / height }
}

async function renderPage() {
  loading.value = true
  error.value = ''
  try {
    documentProxy?.destroy()
    const file = await api.getPaperFile(props.paperId)
    const bytes = new Uint8Array(await file.arrayBuffer())
    documentProxy = await getDocument({ data: bytes }).promise
    const page = await documentProxy.getPage(props.page)
    const rotation = props.rotation ?? page.rotate
    const viewport = page.getViewport({ scale: 1.35, rotation })
    if (!canvas.value) return
    const context = canvas.value.getContext('2d')
    if (!context) throw new Error('canvas unavailable')
    canvas.value.width = Math.ceil(viewport.width)
    canvas.value.height = Math.ceil(viewport.height)
    await page.render({ canvasContext: context, viewport }).promise
    const [x0, y0, x1, y1] = page.view as [number, number, number, number]
    box.value = rotateBox(props.bbox, x1 - x0, y1 - y0, rotation)
  } catch {
    error.value = '无法渲染这条发现对应的论文原文。'
  } finally { loading.value = false }
}

watch(() => [props.paperId, props.page, props.bbox, props.rotation], () => { void renderPage() }, { deep: true })
onMounted(() => { void renderPage() })
onBeforeUnmount(() => { documentProxy?.destroy() })
</script>

<template>
  <div class="preview-backdrop" role="dialog" aria-modal="true" :aria-label="`${aspect} 原文定位`" @click.self="emit('close')">
    <section class="preview-tool">
      <header><div><p class="eyebrow">原文定位</p><h2>{{ aspect }} · 第 {{ page }} 页</h2></div><button class="icon-button" aria-label="关闭预览" @click="emit('close')"><X :size="19" /></button></header>
      <p v-if="loading" class="preview-status">正在渲染论文原文…</p>
      <p v-else-if="error" class="inline-error">{{ error }}</p>
      <div v-else class="page-canvas"><canvas ref="canvas" /><span class="evidence-box" :style="boxStyle" /></div>
    </section>
  </div>
</template>

<style scoped>
.preview-backdrop { position: fixed; z-index: 30; inset: 0; display: grid; place-items: center; padding: 20px; background: rgb(11 29 27 / 58%); }.preview-tool { display: grid; width: min(920px, 100%); max-height: calc(100vh - 40px); gap: 14px; padding: 18px; overflow: auto; border-radius: 8px; background: white; box-shadow: 0 18px 48px rgb(0 0 0 / 25%); }.preview-tool header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }.preview-tool h2 { margin: 2px 0 0; font-size: 17px; }.preview-status { margin: 0; color: var(--ink-soft); }.page-canvas { position: relative; justify-self: center; line-height: 0; }.page-canvas canvas { display: block; max-width: 100%; height: auto; box-shadow: var(--shadow-sm); }.evidence-box { position: absolute; box-sizing: border-box; border: 2px solid #b24031; background: rgb(237 145 78 / 20%); pointer-events: none; }.icon-button { display: grid; width: 36px; height: 36px; place-items: center; border: 1px solid var(--line); border-radius: 8px; background: white; color: var(--ink); cursor: pointer; }@media (max-width: 620px) { .preview-backdrop { padding: 10px; }.preview-tool { max-height: calc(100vh - 20px); padding: 14px; } }
</style>
