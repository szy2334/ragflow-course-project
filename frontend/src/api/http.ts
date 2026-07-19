import axios, { type AxiosError, type AxiosRequestConfig, type InternalAxiosRequestConfig } from 'axios'
import type { ApiErrorBody, ApiResponse, TokenView } from './contracts'

type RetryableConfig = InternalAxiosRequestConfig & { _refresh_attempted?: boolean }

export class ApiError extends Error {
  code: string
  details?: Record<string, unknown>
  requestId?: string
  status?: number

  constructor(body: ApiErrorBody, status?: number) {
    super(body.message ?? '请求未能完成，请稍后重试。')
    this.name = 'ApiError'
    this.code = body.code ?? 'network_error'
    this.details = body.details
    this.requestId = body.request_id
    this.status = status
  }
}

const baseURL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'
const http = axios.create({ baseURL, withCredentials: true, timeout: 45_000 })
let accessToken = sessionStorage.getItem('access_token')
let refreshPromise: Promise<TokenView> | null = null

export function setAccessToken(token: string | null) {
  accessToken = token
  if (token) sessionStorage.setItem('access_token', token)
  else sessionStorage.removeItem('access_token')
}

export const getAccessToken = () => accessToken

function newRequestId() {
  return typeof crypto?.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

http.interceptors.request.use((config) => {
  config.headers.set('X-Request-Id', newRequestId())
  if (accessToken) config.headers.set('Authorization', `Bearer ${accessToken}`)
  return config
})

async function refreshAccessToken() {
  if (!refreshPromise) {
    refreshPromise = axios.post<ApiResponse<TokenView>>(`${baseURL}/auth/refresh`, undefined, { withCredentials: true })
      .then(({ data }) => {
        if (data.code !== 'ok' || !data.data) throw new ApiError(data)
        setAccessToken(data.data.access_token)
        return data.data
      })
      .finally(() => { refreshPromise = null })
  }
  return refreshPromise
}

http.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ApiErrorBody>) => {
    const config = error.config as RetryableConfig | undefined
    const isAuthRequest = config?.url === '/auth/login' || config?.url === '/auth/register' || config?.url === '/auth/refresh'
    if (error.response?.status === 401 && config && !config._refresh_attempted && !isAuthRequest) {
      config._refresh_attempted = true
      try {
        await refreshAccessToken()
        return http(config)
      } catch {
        setAccessToken(null)
        window.dispatchEvent(new CustomEvent('auth-expired'))
      }
    }
    const body = error.response?.data ?? { code: 'network_error', message: '网络连接异常，请检查服务是否可用。' }
    return Promise.reject(new ApiError(body, error.response?.status))
  },
)

/** Orval mutator: returns the documented `data` field while preserving errors. */
export async function request<T>(config: AxiosRequestConfig): Promise<T> {
  const response = await http.request<ApiResponse<T>>(config)
  if (response.data.code !== 'ok' || response.data.data === undefined) throw new ApiError(response.data, response.status)
  return response.data.data
}

export { http, newRequestId }
