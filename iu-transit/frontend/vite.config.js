import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // In production, VITE_API_URL points to Railway.
  // In dev, we proxy /api to localhost:8000 so you don't need CORS.
  const backendUrl = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        // REST API calls
        '/api': {
          target: backendUrl,
          changeOrigin: true,
          secure: false,
        },
        // WebSocket — note: in production the frontend connects
        // directly to Railway (set via VITE_API_URL in the store)
        '/api/buses/live': {
          target: backendUrl.replace('https://', 'wss://').replace('http://', 'ws://'),
          ws: true,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    },
  }
})
