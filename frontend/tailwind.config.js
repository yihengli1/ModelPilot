/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', '"Inter"', "system-ui", "sans-serif"],
        body: ['"Inter"', "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#0f172a",
        accent: "#6dffe3",
        panel: "#0b1021",
      },
      boxShadow: {
        glow: "0 20px 80px rgba(109, 255, 227, 0.15)",
      },
    },
  },
  plugins: [],
};
