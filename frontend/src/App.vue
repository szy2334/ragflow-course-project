<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { BookOpenText, BrainCircuit, ChevronLeft, ClipboardList, FileSearch, FolderCog, LayoutDashboard, LogOut, Menu, MessageSquareText, Scale, Settings2, Sparkles, X } from 'lucide-vue-next'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const mobileOpen = ref(false)
const isPublic = computed(() => route.path === '/login' || route.path === '/register')
watch(() => route.fullPath, () => { mobileOpen.value = false })

const primaryNav = [
  { to: '/papers', label: '读论文', icon: FileSearch },
  { to: '/review', label: '格式审查', icon: Scale },
]
const adminNav = [
  { to: '/admin/format-profiles', label: '格式规范档案', icon: Scale },
  { to: '/admin/models', label: '模型配置', icon: Settings2 },
  { to: '/admin/prompts', label: 'Prompt 模板', icon: ClipboardList },
  { to: '/admin/indexes', label: '索引与知识库', icon: FolderCog },
  { to: '/admin/datasets', label: '数据集', icon: BookOpenText },
  { to: '/admin/evaluations', label: '评测中心', icon: BrainCircuit },
  { to: '/admin/monitoring', label: '运行监控', icon: LayoutDashboard },
]

function go(to: string) { mobileOpen.value = false; router.push(to) }
function goBack() { window.history.state?.back ? router.back() : go('/papers') }
function logout() { auth.logout(); router.push('/login') }
</script>

<template>
  <RouterView v-if="isPublic" />

  <div v-else class="app-shell">
    <a class="skip-link" href="#main-content">跳到主要内容</a>
    <aside id="primary-navigation" class="sidebar" :class="{ 'is-open': mobileOpen }" aria-label="主导航">
      <button class="brand" type="button" @click="go('/papers')">
        <div class="brand-mark"><Sparkles :size="18" aria-hidden="true" /></div>
        <div><strong>知阅</strong><span>Evidence reader</span></div>
      </button>

      <nav class="nav-stack">
        <p class="nav-caption">研究工作台</p>
        <button v-for="item in primaryNav" :key="item.to" class="nav-link" :class="{ active: route.path.startsWith(item.to) }" @click="go(item.to)">
          <component :is="item.icon" :size="18" aria-hidden="true" />{{ item.label }}
        </button>
      </nav>

      <nav v-if="auth.isAdmin" class="nav-stack nav-admin">
        <p class="nav-caption">系统管理</p>
        <button v-for="item in adminNav" :key="item.to" class="nav-link" :class="{ active: route.path === item.to }" @click="go(item.to)">
          <component :is="item.icon" :size="18" aria-hidden="true" />{{ item.label }}
        </button>
      </nav>

      <div class="profile-card">
        <div class="avatar" aria-hidden="true">{{ auth.user?.display_name?.slice(0, 1) || '研' }}</div>
        <div class="profile-info"><strong>{{ auth.user?.display_name || '正在恢复会话' }}</strong><span>{{ auth.user?.role === 'admin' ? '管理员' : '研究者' }}</span></div>
        <button class="icon-button subtle" aria-label="退出登录" title="退出登录" @click="logout"><LogOut :size="17" /></button>
      </div>
    </aside>

    <div v-if="mobileOpen" class="sidebar-scrim" @click="mobileOpen = false" />
    <main id="main-content" class="main-area" tabindex="-1">
      <header class="topbar">
        <button class="icon-button mobile-menu" :aria-label="mobileOpen ? '收起导航' : '打开导航'" aria-controls="primary-navigation" :aria-expanded="mobileOpen" @click="mobileOpen = !mobileOpen">
          <X v-if="mobileOpen" :size="20" /><Menu v-else :size="20" />
        </button>
        <button v-if="route.path !== '/papers'" class="back-link" @click="goBack"><ChevronLeft :size="18" /> 返回</button>
        <div class="topbar-spacer" />
        <div class="privacy-note"><MessageSquareText :size="15" /> 阅读仅检索本地论文</div>
      </header>
      <RouterView />
    </main>
  </div>
</template>
