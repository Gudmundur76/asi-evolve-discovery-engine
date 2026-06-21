/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0a192f',
          800: '#112240',
          700: '#233554',
          600: '#1d3461',
        },
        accent: {
          DEFAULT: '#64ffda',
          dark: '#4ad3b3',
        },
        'text-primary': '#e6f1ff',
        'text-secondary': '#8892b0',
        success: '#64ffda',
        warning: '#ffd700',
        danger: '#ff6b6b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        card: '4px',
        badge: '2px',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(100, 255, 218, 0.15)',
        'glow-lg': '0 0 30px rgba(100, 255, 218, 0.25)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};
