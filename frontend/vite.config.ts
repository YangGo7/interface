import path from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [react()],
  worker: {
    format: 'es',
  },
  resolve: {
    alias: {
      '@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs': path.resolve(__dirname, 'node_modules/@cornerstonejs/codec-libjpeg-turbo-8bit/dist/libjpegturbowasm_decode.js'),
      '@cornerstonejs/codec-charls/decodewasmjs': path.resolve(__dirname, 'node_modules/@cornerstonejs/codec-charls/dist/charlswasm_decode.js'),
      '@cornerstonejs/codec-openjpeg/decodewasmjs': path.resolve(__dirname, 'node_modules/@cornerstonejs/codec-openjpeg/dist/openjpegwasm_decode.js'),
      '@cornerstonejs/codec-openjph/wasmjs': path.resolve(__dirname, 'node_modules/@cornerstonejs/codec-openjph/dist/openjphjs.js'),
    },
  },
  server: {
    port: 5173,
    allowedHosts: ['.ngrok-free.app', '.ngrok.app','oversized-ravage-spotter.ngrok-free.dev'],
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
  optimizeDeps: {
    exclude: [
      '@cornerstonejs/dicom-image-loader',
      '@cornerstonejs/codec-libjpeg-turbo-8bit',
      '@cornerstonejs/codec-charls',
      '@cornerstonejs/codec-openjpeg',
      '@cornerstonejs/codec-openjph',
    ],
    include: ['dicom-parser'],
  },
});
