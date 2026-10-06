/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./", // static hosting from any path
  plugins: [react(), tailwindcss()],
  test: { include: ["src/**/*.test.ts"], exclude: ["**/._*", "**/node_modules/**"] }, // ._*: exFAT AppleDouble files
});
