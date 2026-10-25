/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0b1026",
        surface: "#10182b",
        panel: "#151d33",
        border: "#243049",
        foreground: "#e6edf7",
        muted: "#8fa1bd",
        accent: "#38bdf8",
        violet: "#a78bfa",
        success: "#22c55e",
        warning: "#f59e0b",
        danger: "#ef4444",
      },
      borderRadius: {
        xl: "14px",
      },
    },
  },
  plugins: [],
}
