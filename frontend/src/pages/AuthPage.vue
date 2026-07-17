<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowRight, Eye, EyeOff, LockKeyhole, Mail, Sparkles, UserRound } from 'lucide-vue-next'
import { ApiError } from '@/api/http'
import { DEMO_ACCOUNT } from '@/api/demo'
import { useAuthStore } from '@/stores/auth'

const props = defineProps<{ mode: 'login' | 'register' }>()
const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const email = ref('')
const password = ref('')
const displayName = ref('')
const showPassword = ref(false)
const busy = ref(false)
const error = ref('')
const isRegister = computed(() => props.mode === 'register')
const demoEnabled = import.meta.env.DEV

function fillDemoAccount() {
  email.value = DEMO_ACCOUNT.email
  password.value = DEMO_ACCOUNT.password
}

async function submit() {
  error.value = ''
  if (isRegister.value && displayName.value.trim().length < 2) { error.value = '请输入至少 2 个字符的显示名称。'; return }
  if (password.value.length < 8) { error.value = '密码至少需要 8 个字符。'; return }
  busy.value = true
  try {
    if (isRegister.value) await auth.register(email.value.trim(), password.value, displayName.value.trim())
    else await auth.login(email.value.trim(), password.value)
    router.push((route.query.redirect as string) || '/papers')
  } catch (cause) {
    error.value = cause instanceof ApiError ? cause.message : '暂时无法连接服务，请稍后重试。'
  } finally { busy.value = false }
}
</script>

<template>
  <main class="auth-layout">
    <section class="auth-intro">
      <div class="auth-brand"><span class="brand-mark"><Sparkles :size="19" /></span><strong>知阅</strong></div>
      <div class="auth-copy"><p class="eyebrow">Evidence-grounded reading</p><h1>把每个结论<br />落回论文原文。</h1><p>上传、比较、追问和导出报告。所有回答均附带可定位的证据，而不是无出处的概述。</p></div>
      <div class="auth-signal"><span /><p>基于多智能体协作与证据溯源</p></div>
    </section>

    <section class="auth-form-wrap">
      <form class="auth-form" @submit.prevent="submit">
        <p class="eyebrow">{{ isRegister ? '创建研究空间' : '欢迎回来' }}</p>
        <h2>{{ isRegister ? '开始有证据的阅读' : '登录你的工作台' }}</h2>
        <p class="form-lede">{{ isRegister ? '用一个账户管理论文、会话和阅读报告。' : '使用你的账户继续上次的研究。' }}</p>
        <div v-if="error" class="form-error" role="alert">{{ error }}</div>
        <button v-if="demoEnabled && !isRegister" type="button" class="demo-account" @click="fillDemoAccount">填入演示账号</button>
        <label v-if="isRegister" class="input-field"><span>显示名称</span><div><UserRound :size="17" /><input v-model="displayName" autocomplete="name" placeholder="例如：李同学" required /></div></label>
        <label class="input-field"><span>邮箱</span><div><Mail :size="17" /><input v-model="email" type="email" autocomplete="email" placeholder="name@university.edu" required /></div></label>
        <label class="input-field"><span>密码</span><div><LockKeyhole :size="17" /><input v-model="password" :type="showPassword ? 'text' : 'password'" :autocomplete="isRegister ? 'new-password' : 'current-password'" placeholder="至少 8 个字符" minlength="8" required /><button type="button" class="input-icon" :aria-label="showPassword ? '隐藏密码' : '显示密码'" @click="showPassword = !showPassword"><EyeOff v-if="showPassword" :size="17" /><Eye v-else :size="17" /></button></div></label>
        <button class="primary-button full-width" :disabled="busy"><span>{{ busy ? '正在验证…' : (isRegister ? '创建账户' : '登录') }}</span><ArrowRight :size="18" /></button>
        <p class="form-switch">{{ isRegister ? '已经有账户？' : '还没有账户？' }} <RouterLink :to="isRegister ? '/login' : '/register'">{{ isRegister ? '去登录' : '免费注册' }}</RouterLink></p>
      </form>
    </section>
  </main>
</template>
