module.exports = {
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}'
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#0f172a',
          50: '#f5f7fb',
          100: '#e6eef9',
          200: '#c7def3',
          300: '#96bff0',
          400: '#5f92ea',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#172554'
        },
        accent: {
          DEFAULT: '#06b6d4'
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial']
      },
      boxShadow: {
        card: '0 6px 18px rgba(12, 20, 34, 0.08)'
      }
    },
  },
  plugins: [],
}
