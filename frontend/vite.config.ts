import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/upload': 'http://127.0.0.1:8000',
      '/analyses': 'http://127.0.0.1:8000',
      '/dashboard': 'http://127.0.0.1:8000',
      '/model': 'http://127.0.0.1:8000',
      '/evaluation': 'http://127.0.0.1:8000',
    }
  }
})
