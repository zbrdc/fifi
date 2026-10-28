import { defineConfig } from "vite";

const engineTarget = "http://127.0.0.1:8100";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/health": engineTarget,
      "/ready": engineTarget,
      "/v1": engineTarget,
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    proxy: {
      "/health": engineTarget,
      "/ready": engineTarget,
      "/v1": engineTarget,
    },
  },
});
