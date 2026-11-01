import { compact, monthLabel } from '../lib/format'

interface Point { month: string; revenue: number; expenses: number; closing_cash: number | null }

/** Revenue and expense bars per month, with closing cash as a line on its own scale. */
export function MonthlyChart({ data, currency }: { data: Point[]; currency: string }) {
  if (data.length === 0) return null
  const W = 640, H = 220, pad = { l: 56, r: 56, t: 12, b: 28 }
  const inner = { w: W - pad.l - pad.r, h: H - pad.t - pad.b }
  const maxBar = Math.max(1, ...data.flatMap(d => [d.revenue, d.expenses]))
  const cash = data.map(d => d.closing_cash).filter((c): c is number => c !== null)
  const maxCash = Math.max(1, ...cash)
  const slot = inner.w / data.length
  const bw = Math.min(18, slot / 3)
  const y = (v: number) => pad.t + inner.h - (v / maxBar) * inner.h
  const yc = (v: number) => pad.t + inner.h - (v / maxCash) * inner.h
  const line = data.map((d, i) => (d.closing_cash === null ? null : `${pad.l + slot * i + slot / 2},${yc(d.closing_cash)}`))
    .filter(Boolean).join(' ')
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Revenue, expenses and closing cash by month">
        {[0, 0.5, 1].map(f => (
          <g key={f}>
            <line x1={pad.l} x2={W - pad.r} y1={y(maxBar * f)} y2={y(maxBar * f)} className="grid" />
            <text x={pad.l - 6} y={y(maxBar * f) + 4} textAnchor="end">{compact(maxBar * f, currency)}</text>
            {cash.length > 0 && <text x={W - pad.r + 6} y={yc(maxCash * f) + 4} className="cash-axis">{compact(maxCash * f, currency)}</text>}
          </g>
        ))}
        {data.map((d, i) => {
          const x = pad.l + slot * i + slot / 2
          return (
            <g key={d.month}>
              <rect x={x - bw - 1} y={y(d.revenue)} width={bw} height={pad.t + inner.h - y(d.revenue)} className="bar-rev" rx={2}>
                <title>{`${monthLabel(d.month)} revenue ${compact(d.revenue, currency)}`}</title></rect>
              <rect x={x + 1} y={y(d.expenses)} width={bw} height={pad.t + inner.h - y(d.expenses)} className="bar-exp" rx={2}>
                <title>{`${monthLabel(d.month)} expenses ${compact(d.expenses, currency)}`}</title></rect>
              <text x={x} y={H - 8} textAnchor="middle">{monthLabel(d.month)}</text>
            </g>
          )
        })}
        {line && <polyline points={line} className="cash-line" />}
      </svg>
      <figcaption className="legend">
        <span><i className="key-rev" /> Revenue</span><span><i className="key-exp" /> Expenses</span>
        {cash.length > 0 && <span><i className="key-cash" /> Cash in bank (right scale)</span>}
      </figcaption>
    </figure>
  )
}

/** One horizontal bar per item, scaled to the largest. */
export function Bars({ items, currency }: { items: { label: string; amount: number }[]; currency: string }) {
  const max = Math.max(1, ...items.map(i => i.amount))
  return (
    <ul className="hbars">
      {items.map(i => (
        <li key={i.label}>
          <span className="hbar-label">{i.label}</span>
          <span className="hbar-track"><span className="hbar-fill" style={{ width: `${(i.amount / max) * 100}%` }} /></span>
          <span className="hbar-value">{compact(i.amount, currency)}</span>
        </li>
      ))}
    </ul>
  )
}
