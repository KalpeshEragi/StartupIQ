import { useEffect, useState } from 'react'
import { Icon } from './Icons'

type Theme = 'light' | 'dark'
const KEY = 'siq-theme'

function systemTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function savedTheme(): Theme | null {
  try {
    const t = localStorage.getItem(KEY)
    return t === 'light' || t === 'dark' ? t : null
  } catch { return null }
}

/** Sun / moon button. Follows the system until the user picks; the pick is remembered on this device. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => savedTheme() ?? systemTheme())

  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])

  function flip() {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try { localStorage.setItem(KEY, next) } catch { /* private mode: still switches for this visit */ }
  }

  const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
  return (
    <button className="icon-btn theme-toggle" onClick={flip} aria-label={label} title={label}>
      <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
    </button>
  )
}
