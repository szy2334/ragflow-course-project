import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { pinia } from '@/stores/pinia'

declare module 'vue-router' { interface RouteMeta { requiresAuth?: boolean; admin?: boolean } }

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/papers' },
  { path: '/login', component: () => import('@/pages/AuthPage.vue'), props: { mode: 'login' } },
  { path: '/register', component: () => import('@/pages/AuthPage.vue'), props: { mode: 'register' } },
  { path: '/papers', component: () => import('@/pages/PapersPage.vue'), meta: { requiresAuth: true } },
  { path: '/papers/:paperId', component: () => import('@/pages/PaperDetailPage.vue'), props: true, meta: { requiresAuth: true } },
  { path: '/review', component: () => import('@/pages/ReviewPage.vue'), meta: { requiresAuth: true } },
  { path: '/chat/:sessionId', component: () => import('@/pages/ChatPage.vue'), props: true, meta: { requiresAuth: true } },
  { path: '/compare', component: () => import('@/pages/ComparePage.vue'), meta: { requiresAuth: true } },
  { path: '/reports', component: () => import('@/pages/ReportsPage.vue'), meta: { requiresAuth: true } },
  { path: '/reports/:reportId', component: () => import('@/pages/ReportDetailPage.vue'), props: true, meta: { requiresAuth: true } },
  { path: '/admin/format-profiles', component: () => import('@/pages/FormatProfileAdminPage.vue'), meta: { requiresAuth: true, admin: true } },
  { path: '/admin/:section(models|prompts|indexes|datasets|evaluations|monitoring)', component: () => import('@/pages/AdminPage.vue'), props: true, meta: { requiresAuth: true, admin: true } },
  { path: '/:pathMatch(.*)*', redirect: '/papers' },
]

export const router = createRouter({ history: createWebHistory(), routes, scrollBehavior: () => ({ top: 0 }) })

router.beforeEach(async (to) => {
  const auth = useAuthStore(pinia)
  if (to.meta.requiresAuth || to.meta.admin) await auth.initialize()
  if (to.meta.requiresAuth && auth.status !== 'authenticated') return { path: '/login', query: { redirect: to.fullPath } }
  if (to.meta.admin && !auth.isAdmin) return '/papers'
  if ((to.path === '/login' || to.path === '/register') && auth.status === 'authenticated') return '/papers'
  return true
})
