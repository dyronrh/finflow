import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs on :8000 (make api); Vite proxies /v1 to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/v1": "http://localhost:8000" },
  },
  build: {
    chunkSizeWarningLimit: 800,
  },
});
