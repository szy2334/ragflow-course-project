import { defineStore } from 'pinia'
import { api } from '@/api'
import { getAccessTokenExpiresAt, refreshAccessToken, setAccessToken } from '@/api/http'
import type { TokenView, UserView } from '@/api/contracts'

type AuthStatus = 'unknown' | 'authenticated' | 'anonymous'

export const useAuthStore = defineStore('auth', {
  state: () => ({ user: null as UserView | null, status: 'unknown' as AuthStatus, error: '' }),
  getters: { isAdmin: (state) => state.user?.role === 'admin' },
  actions: {
    setSession(user: UserView, token: TokenView) {
      this.user = user
      this.status = 'authenticated'
      this.error = ''
      setAccessToken(token.access_token, token.access_expires_at)
      this.scheduleRefresh()
    },
    scheduleRefresh() {
      const expiresAt = getAccessTokenExpiresAt()
      const timestamp = expiresAt ? Date.parse(expiresAt) : Number.NaN
      const delay = Number.isFinite(timestamp)
        ? Math.max(15_000, timestamp - Date.now() - 60_000)
        : 12 * 60_000
      window.setTimeout(() => { void this.refreshSession() }, delay)
    },
    async refreshSession() {
      if (this.status !== 'authenticated') return
      try {
        await refreshAccessToken()
        this.scheduleRefresh()
      } catch {
        this.logout()
      }
    },
    async initialize() {
      if (this.status !== 'unknown') return
      try {
        this.user = await api.me()
        this.status = 'authenticated'
        this.scheduleRefresh()
      } catch {
        this.user = null
        this.status = 'anonymous'
        setAccessToken(null)
      }
    },
    async login(email: string, password: string) {
      const auth = await api.login({ email, password })
      this.setSession(auth.user, auth.token)
    },
    async register(email: string, password: string, display_name: string) {
      const auth = await api.register({ email, password, display_name })
      this.setSession(auth.user, auth.token)
    },
    logout() {
      this.user = null
      this.status = 'anonymous'
      setAccessToken(null)
    },
  },
})
