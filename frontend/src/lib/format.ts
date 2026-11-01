// Number formatting the Indian way: ₹12,34,567 in tables, ₹12.3 L / ₹1.2 Cr in tiles.

const SYMBOL: Record<string, string> = { INR: '₹', USD: '$', EUR: '€', GBP: '£', SGD: 'S$', AED: 'AED ' }

export function money(n: number | null | undefined, currency = 'INR'): string {
  if (n === null || n === undefined) return '—'
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency, maximumFractionDigits: 0 }).format(n)
}

export function compact(n: number | null | undefined, currency = 'INR'): string {
  if (n === null || n === undefined) return '—'
  const s = SYMBOL[currency] ?? ''
  const sign = n < 0 ? '−' : ''
  const a = Math.abs(n)
  if (currency === 'INR') {
    if (a >= 1e7) return `${sign}${s}${(a / 1e7).toFixed(a >= 1e8 ? 0 : 1)} Cr`
    if (a >= 1e5) return `${sign}${s}${(a / 1e5).toFixed(a >= 1e6 ? 0 : 1)} L`
  } else {
    if (a >= 1e6) return `${sign}${s}${(a / 1e6).toFixed(1)}M`
  }
  if (a >= 1e3) return `${sign}${s}${(a / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`
  return `${sign}${s}${Math.round(a)}`
}

export function pct(n: number | null | undefined, digits = 0): string {
  return n === null || n === undefined ? '—' : `${(n * 100).toFixed(digits)}%`
}

export function count(n: number | null | undefined): string {
  return n === null || n === undefined ? '—' : n.toLocaleString('en-IN')
}

export function monthLabel(ym: string, long = false): string {
  const [y, m] = ym.split('-').map(Number)
  return new Date(y, m - 1, 1).toLocaleDateString('en-IN', { month: long ? 'long' : 'short', year: long ? 'numeric' : '2-digit' })
}

export const today = () => new Date().toISOString().slice(0, 10)
