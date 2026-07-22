<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Check, ClipboardList, Plus, Trash2 } from 'lucide-vue-next'
import PageHeader from '@/components/PageHeader.vue'
import StatusPill from '@/components/StatusPill.vue'
import { api } from '@/api'
import { ApiError } from '@/api/http'
import type { AdminFormatProfileView, FormatProfileCreateInput } from '@/api/contracts'

type ModeMapping = { mode: string; documentId: string }

const profiles = ref<AdminFormatProfileView[]>([])
const loading = ref(true)
const saving = ref(false)
const error = ref('')
const success = ref('')
const profileKey = ref('')
const name = ref('')
const venueId = ref('')
const version = ref('')
const description = ref('')
const datasetId = ref('')
const retrievalQuery = ref('')
const sharedDocumentId = ref('')
const modeMappings = ref<ModeMapping[]>([{ mode: 'initial_submission', documentId: '' }])
const ruleManifest = ref('[]')
const isActive = ref(true)

const totalRules = computed(() => profiles.value.reduce((total, item) => total + item.rule_manifest.length, 0))

function resetForm() {
  profileKey.value = ''
  name.value = ''
  venueId.value = ''
  version.value = ''
  description.value = ''
  datasetId.value = ''
  retrievalQuery.value = ''
  sharedDocumentId.value = ''
  modeMappings.value = [{ mode: 'initial_submission', documentId: '' }]
  ruleManifest.value = '[]'
  isActive.value = true
  success.value = ''
}

function useAsTemplate(profile: AdminFormatProfileView) {
  profileKey.value = profile.profile_key
  name.value = profile.name
  venueId.value = profile.venue_id
  version.value = ''
  description.value = profile.description ?? ''
  datasetId.value = profile.ragflow_dataset_id
  retrievalQuery.value = profile.retrieval_query
  sharedDocumentId.value = profile.shared_document_id
  modeMappings.value = Object.entries(profile.mode_document_mapping).map(([mode, documentId]) => ({ mode, documentId }))
  ruleManifest.value = JSON.stringify(profile.rule_manifest, null, 2)
  isActive.value = profile.is_active
  success.value = ''
  error.value = ''
}

function addMode() { modeMappings.value.push({ mode: '', documentId: '' }) }
function removeMode(index: number) { modeMappings.value.splice(index, 1) }

async function load() {
  loading.value = true
  error.value = ''
  try {
    profiles.value = (await api.listAdminFormatProfiles()).items
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法读取格式规范档案。'
  } finally {
    loading.value = false
  }
}

async function createProfile() {
  error.value = ''
  success.value = ''
  let rules: Array<Record<string, unknown>>
  try {
    const parsed = JSON.parse(ruleManifest.value)
    if (!Array.isArray(parsed)) throw new Error('规则清单必须是 JSON 数组。')
    rules = parsed as Array<Record<string, unknown>>
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '规则清单不是有效 JSON。'
    return
  }
  const enabledMappings = modeMappings.value.filter((item) => item.mode.trim() || item.documentId.trim())
  if (!enabledMappings.length || enabledMappings.some((item) => !item.mode.trim() || !item.documentId.trim())) {
    error.value = '每个投稿模式都需要模式名称和规则文档 ID。'
    return
  }
  const allowedModes = enabledMappings.map((item) => item.mode.trim())
  if (new Set(allowedModes).size !== allowedModes.length) {
    error.value = '投稿模式不能重复。'
    return
  }
  const data: FormatProfileCreateInput = {
    profile_key: profileKey.value.trim(),
    name: name.value.trim(),
    venue_id: venueId.value.trim() || undefined,
    version: version.value.trim(),
    description: description.value.trim() || null,
    ragflow_dataset_id: datasetId.value.trim(),
    retrieval_query: retrievalQuery.value.trim(),
    shared_document_id: sharedDocumentId.value.trim(),
    allowed_submission_modes: allowedModes,
    mode_document_mapping: Object.fromEntries(enabledMappings.map((item) => [item.mode.trim(), item.documentId.trim()])),
    rules,
    is_active: isActive.value,
  }
  saving.value = true
  try {
    await api.createAdminFormatProfile(data)
    success.value = '格式规范版本已创建。'
    await load()
    resetForm()
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '无法创建格式规范版本。'
  } finally {
    saving.value = false
  }
}

onMounted(load)
</script>

<template>
  <section class="page format-profile-admin-page">
    <PageHeader eyebrow="管理员控制台" title="格式规范档案" description="格式版本、受控规则文档与规则清单" />
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    <p v-if="success" class="inline-success"><Check :size="16" />{{ success }}</p>

    <section class="profile-overview" aria-label="格式规范档案概览">
      <span>{{ profiles.length }} 个档案</span>
      <span>{{ totalRules }} 条规则</span>
    </section>

    <div v-if="loading" class="skeleton-list"><div v-for="item in 3" :key="item" class="skeleton-row" /></div>
    <section v-else class="profile-list" aria-label="已有格式规范档案">
      <article v-for="profile in profiles" :key="profile.format_profile_id" class="profile-row">
        <div>
          <div class="profile-row-title"><strong>{{ profile.name }}</strong><StatusPill :status="profile.is_active && !profile.configuration_issues.length ? 'ready' : 'failed'" /><span v-if="profile.configuration_issues.length" class="configuration-warning">配置不完整</span></div>
          <p>{{ profile.venue_id }} · {{ profile.version }} · {{ profile.allowed_submission_modes.join('、') }} · {{ profile.rule_manifest.length }} 条规则</p>
        </div>
        <button class="secondary-button" type="button" @click="useAsTemplate(profile)"><ClipboardList :size="16" />基于此版本新建</button>
      </article>
      <p v-if="!profiles.length" class="empty-copy">尚未创建格式规范档案。</p>
    </section>

    <section class="profile-editor" aria-labelledby="profile-editor-title">
      <div class="editor-heading"><div><p class="eyebrow">新建版本</p><h2 id="profile-editor-title">受控格式规范</h2></div><button class="text-button" type="button" @click="resetForm">清空</button></div>
      <form class="profile-form" @submit.prevent="createProfile">
        <div class="field-grid">
          <label class="input-field plain"><span>档案键</span><input v-model="profileKey" required maxlength="128" /></label>
          <label class="input-field plain"><span>显示名称</span><input v-model="name" required maxlength="300" /></label>
          <label class="input-field plain"><span>投稿场所 ID</span><input v-model="venueId" maxlength="128" /></label>
          <label class="input-field plain"><span>规范版本</span><input v-model="version" required maxlength="128" /></label>
        </div>
        <label class="input-field plain"><span>说明</span><textarea v-model="description" rows="2" maxlength="4000" /></label>
        <div class="field-grid">
          <label class="input-field plain"><span>RAGFlow 数据集 ID</span><input v-model="datasetId" required maxlength="128" /></label>
          <label class="input-field plain"><span>共享规则文档 ID</span><input v-model="sharedDocumentId" required maxlength="128" /></label>
        </div>
        <label class="input-field plain"><span>检索主题</span><input v-model="retrievalQuery" required maxlength="4000" /></label>

        <div class="mode-editor"><div class="mode-heading"><span>投稿模式规则文档</span><button class="icon-button" type="button" title="添加投稿模式" aria-label="添加投稿模式" @click="addMode"><Plus :size="17" /></button></div>
          <div v-for="(mapping, index) in modeMappings" :key="index" class="mode-row">
            <input v-model="mapping.mode" aria-label="投稿模式" placeholder="initial_submission" maxlength="64" />
            <input v-model="mapping.documentId" aria-label="投稿模式规则文档 ID" placeholder="RAGFlow 文档 ID" maxlength="128" />
            <button class="icon-button" type="button" title="删除投稿模式" aria-label="删除投稿模式" :disabled="modeMappings.length === 1" @click="removeMode(index)"><Trash2 :size="16" /></button>
          </div>
        </div>

        <div class="manifest-heading"><span>规则清单 JSON</span><p>每条启用规则必须含 <code>applicable_unit_kinds</code>、<code>is_global</code>、<code>requires_cross_unit</code>、<code>cross_unit_kinds</code>、<code>applicability_conditions</code> 和 <code>evidence_selector</code>；提交前由服务端校验。</p></div>
        <label class="input-field plain"><textarea v-model="ruleManifest" class="manifest-input" rows="14" spellcheck="false" required /></label>
        <label class="toggle-field"><input v-model="isActive" type="checkbox" /><span>启用此格式规范版本</span></label>
        <button class="primary-button" type="submit" :disabled="saving">{{ saving ? '正在创建…' : '创建格式规范版本' }}</button>
      </form>
    </section>
  </section>
</template>

<style scoped>
.format-profile-admin-page, .profile-list, .profile-editor, .profile-form, .manifest-heading { display: grid; gap: 16px; }
.profile-overview { display: flex; gap: 18px; color: var(--ink-soft); font-size: 13px; }
.profile-row { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 16px 0; border-bottom: 1px solid var(--line); }
.profile-row-title, .editor-heading, .mode-heading, .mode-row, .inline-success { display: flex; align-items: center; gap: 10px; }
.profile-row-title strong { font-size: 15px; }.configuration-warning { color: #88620b; font-size: 12px; }.profile-row p { margin: 5px 0 0; color: var(--ink-soft); font-size: 13px; }.empty-copy { margin: 0; color: var(--ink-soft); }
.profile-editor { padding-top: 20px; border-top: 1px solid var(--line); }.editor-heading { justify-content: space-between; }.editor-heading h2 { margin: 3px 0 0; font-size: 18px; }
.field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }.mode-editor { display: grid; gap: 9px; }.mode-heading { justify-content: space-between; color: var(--ink-soft); font-size: 13px; }.mode-row { display: grid; grid-template-columns: minmax(160px, 0.55fr) minmax(0, 1fr) 36px; }.mode-row input { min-width: 0; min-height: 40px; padding: 0 10px; border: 1px solid var(--line); border-radius: 6px; background: white; color: var(--ink); }.manifest-heading { gap: 5px; }.manifest-heading p { margin: 0; color: var(--ink-soft); font-size: 12px; line-height: 1.5; }.manifest-heading code { color: var(--ink); }.manifest-input { min-height: 240px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; line-height: 1.45; }.toggle-field { display: inline-flex; align-items: center; gap: 8px; color: var(--ink-soft); font-size: 13px; }.inline-success { margin: 0; color: #2f6a45; font-size: 13px; }
@media (max-width: 720px) { .profile-row { align-items: flex-start; flex-direction: column; }.field-grid { grid-template-columns: 1fr; }.mode-row { grid-template-columns: 1fr 1fr 36px; }.profile-row .secondary-button { width: 100%; justify-content: center; } }
</style>
