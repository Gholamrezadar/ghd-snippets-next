/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'ghd-dark-primary': '#3B96EB',
        'ghd-dark-bg': '#1D2026',
        'ghd-dark-dark': '#181A1F',
        'ghd-dark-checkbox-disabled': '#787878',
        'ghd-dark-code-bg': 'rgb(40, 44, 52)',
      },
      textColor: {
        'ghd-dark-text': '#FFFFFF',
        'ghd-dark-muted-text': '#4E5054',
      },
    },
  },
  plugins: [require('tailwind-scrollbar')],
  darkMode: 'class',
  variants: {
    scrollbar: ['dark'],
  },
};
