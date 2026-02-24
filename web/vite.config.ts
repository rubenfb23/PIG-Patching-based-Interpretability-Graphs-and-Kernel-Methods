import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/ws-base": {
        target: "ws://127.0.0.1:8765",
        ws: true,
        changeOrigin: true,
      },
      "/ws-diff": {
        target: "ws://127.0.0.1:8766",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
