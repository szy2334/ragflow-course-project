import { defineStore } from 'pinia'
import { api } from '@/api'
import { setAccessToken } from '@/api/http'
import type { UserView } from '@/api/contracts'

type AuthStatus = 'unknown' | 'authenticated' | 'anonymous'

export const useAuthStore = defineStore('auth', {
  state: () => ({ user: null as UserView | null, status: 'unknown' as AuthStatus, error: '' }),
  getters: { isAdmin: (state) => state.user?.role === 'admin' },
  actions: {
    setSession(user: UserView, accessToken: string) {
      this.user = user
      this.status = 'authenticated'
      this.error = ''
      setAccessToken(accessToken)
    },
    async initialize() {
      if (this.status !== 'unknown') return
      try {
        this.user = await api.me()
        this.status = 'authenticated'
      } catch {
        this.user = null
        this.status = 'anonymous'
        setAccessToken(null)
      }
    },
    async login(email: string, password: string) {
      const auth = await api.login({ email, password })
      this.setSession(auth.user, auth.token.access_token)
    },
    async register(email: string, password: string, display_name: string) {
      const auth = await api.register({ email, password, display_name })
      this.setSession(auth.user, auth.token.access_token)
    },
    logout() {
      this.user = null
      this.status = 'anonymous'
      setAccessToken(null)
    },
  },
})
