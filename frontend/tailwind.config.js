/** @type {import('tailwindcss').Config} */
export default {
	content: ["./index.html", "./src/**/*.{js,jsx}"],
	theme: {
		extend: {
			colors: {
				"main-black": "var(--primary-black)",
				"main-black-hover": "var(--primary-black-hover)",
				"main-white": "var(--primary-white)",
				"main-white-hover": "var(--primary-white-hover)",
			},
		},
	},
	plugins: [],
};
