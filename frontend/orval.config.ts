import { defineConfig } from 'orval'

/** The backend exposes the document's sole API contract at this address. */
export default defineConfig({
  paperReader: {
    input: {
      target: process.env.OPENAPI_URL ?? 'http://localhost:8000/api/v1/openapi.json',
    },
    output: {
      target: './src/api/generated/client.ts',
      schemas: './src/api/generated/models',
      client: 'axios',
      mode: 'split',
      clean: true,
      override: {
        mutator: {
          path: './src/api/http.ts',
          name: 'request',
        },
      },
    },
  },
})
