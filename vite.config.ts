import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

// Pure static site, fully offline: both models are served from public/models.
export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: {
    alias: {
      // Only the tfjs runtime of the face detector is used (see src/ml/mediapipeStub.ts).
      '@mediapipe/face_detection': fileURLToPath(new URL('./src/ml/mediapipeStub.ts', import.meta.url)),
    },
  },
  server: {
    port: 5174,
    host: 'localhost', // getUserMedia needs a secure context; localhost is one
  },
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 3000,
  },
});
