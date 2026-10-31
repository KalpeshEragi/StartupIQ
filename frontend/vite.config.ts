import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the API runs separately; proxying /api keeps the browser on one origin, so the refresh cookie works.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: process.env.VITE_API_TARGET ?? 'http://localhost:8000' } },
  },
})
