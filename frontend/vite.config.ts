import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [vue()],
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: {
      port: 5173,
      proxy: {
        '/api': {
          // On Windows, localhost can resolve to an unrelated IPv6 listener.
          // The local FastAPI process listens on IPv4, so make the development
          // proxy deterministic for API calls and PDF blobs alike.
          target: env.VITE_API_ORIGIN ?? 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
