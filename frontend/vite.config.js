import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: [
      '6641072add9a.ngrok-free.app',
      '.ngrok-free.app', // Allow all ngrok-free.app subdomains
      '.ngrok.app',      // Allow all ngrok.app subdomains
      '.ngrok.io'        // Allow all ngrok.io subdomains (legacy)
    ]
  }
});
