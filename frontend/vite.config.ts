import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 백엔드가 5000에서 동작한다고 가정. 다르면 수정하세요.
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
      '/temp': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
});
