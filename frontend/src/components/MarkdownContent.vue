<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import katex from 'katex'

const props = defineProps<{ content: string }>()
function renderMarkdown(markdown: string) {
  const formulas: string[] = []
  const protect = (expression: string, displayMode: boolean) => {
    const token = `PAPER_READER_MATH_${formulas.length}_TOKEN`
    formulas.push(katex.renderToString(expression.trim(), { throwOnError: false, displayMode }))
    return token
  }
  const textWithMathTokens = markdown
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, expression: string) => protect(expression, true))
    .replace(/(^|[^\\$])\$([^\n$]+)\$(?!\$)/g, (_, prefix: string, expression: string) => `${prefix}${protect(expression, false)}`)
  const renderer = new marked.Renderer()
  renderer.html = () => ''
  const rendered = marked.parse(textWithMathTokens, { async: false, breaks: true, renderer })
  const withMath = formulas.reduce((content, formula, index) => content.replace(`PAPER_READER_MATH_${index}_TOKEN`, formula), rendered)
  return DOMPurify.sanitize(withMath, { USE_PROFILES: { html: true, mathMl: true } })
}

const html = computed(() => renderMarkdown(props.content))
</script>

<template><div class="markdown-content" v-html="html" /></template>
