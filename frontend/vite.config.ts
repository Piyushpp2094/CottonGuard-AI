import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const PROXY_TARGET = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: PROXY_TARGET,
        changeOrigin: true,
      },
      '/artifacts': {
        target: PROXY_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react_vendor: ['react', 'react-dom', 'react-router-dom'],
          charts_vendor: ['recharts'],
          icons_vendor: ['lucide-react'],
        },
      },
    },
  },
});
