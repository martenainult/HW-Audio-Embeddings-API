import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,      // Essential for Docker: allows mapping to 0.0.0.0
    port: 5173,
    watch: {
      usePolling: true // Helps with file syncing on some Windows/WSL setups
    }
  }
})