import { compact, monthLabel } from '../lib/format'

interface Point { month: string; revenue: number; expenses: number; closing_cash: number | null }

/** Revenue and expense bars per month, with closing cash as a line on its own scale. */
export function MonthlyChart({ data, currency }: { data: Point[]; currency: string }) {
  if (data.length === 0) return null
  const W = 640, H = 240, pad = { l: 56, r: 56, t: 16, b: 30 }
  const inner = { w: W - pad.l - pad.r, h: H - pad.t - pad.b }
  const maxBar = Math.max(1, ...data.flatMap(d => [d.revenue, d.expenses]))
  const cash = data.map(d => d.closing_cash).filter((c): c is number => c !== null)
  const maxCash = Math.max(1, ...cash)
  const slot = inner.w / data.length
  const bw = Math.min(16, slot / 3.4)
  const y = (v: number) => pad.t + inner.h - (v / maxBar) * inner.h
  const yc = (v: number) => pad.t + inner.h - (v / maxCash) * inner.h
  const pts = data.map((d, i) => (d.closing_cash === null ? null : [pad.l + slot * i + slot / 2, yc(d.closing_cash)] as const))
    .filter((p): p is readonly [number, number] => p !== null)
  const line = pts.map(p => p.join(',')).join(' ')
  const area = pts.length > 1 ? `M${pts[0][0]},${pad.t + inner.h} L${line.replace(/ /g, ' L')} L${pts[pts.length - 1][0]},${pad.t + inner.h} Z` : ''
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Revenue, expenses and closing cash by month">
        {area && <path d={area} className="cash-area" />}
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
              <rect x={x - bw - 1} y={y(d.revenue)} width={bw} height={pad.t + inner.h - y(d.revenue)} className="bar-rev" rx={4}>
                <title>{`${monthLabel(d.month)} revenue ${compact(d.revenue, currency)}`}</title></rect>
              <rect x={x + 1} y={y(d.expenses)} width={bw} height={pad.t + inner.h - y(d.expenses)} className="bar-exp" rx={4}>
                <title>{`${monthLabel(d.month)} expenses ${compact(d.expenses, currency)}`}</title></rect>
              <text x={x} y={H - 8} textAnchor="middle">{monthLabel(d.month)}</text>
            </g>
          )
        })}
        {line && <polyline points={line} className="cash-line" />}
        {pts.map(([x, yy]) => <circle key={x} cx={x} cy={yy} r={3.5} className="cash-dot" />)}
      </svg>
      <table className="sr-only">
        <caption>Revenue, expenses and cash in bank by month</caption>
        <thead><tr><th>Month</th><th>Revenue</th><th>Expenses</th><th>Cash in bank</th></tr></thead>
        <tbody>{data.map(d => <tr key={d.month}><td>{monthLabel(d.month)}</td><td>{compact(d.revenue, currency)}</td><td>{compact(d.expenses, currency)}</td><td>{d.closing_cash === null ? '—' : compact(d.closing_cash, currency)}</td></tr>)}</tbody>
      </table>
      <figcaption className="legend" aria-hidden="true">
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

interface ForecastPoint { week: string; p10: number; p50: number; p90: number }
const shortMonth = (iso: string) => new Intl.DateTimeFormat('en-IN', { month: 'short' }).format(new Date(iso))

/** Market interest: the recent past as a solid line, the forecast as a dashed median inside its P10-P90 band. */
export function ForecastChart({ history, forecast, label }: { history: { week: string; index: number }[]; forecast: ForecastPoint[]; label: string }) {
  const past = history.slice(-26)
  const W = 680, H = 260, pad = { l: 44, r: 16, t: 16, b: 30 }
  const n = past.length + forecast.length
  const vals = [...past.map(p => p.index), ...forecast.flatMap(f => [f.p10, f.p90])]
  const lo = Math.floor(Math.min(...vals) * 0.95), hi = Math.ceil(Math.max(...vals) * 1.05)
  const x = (i: number) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r)
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo || 1)) * (H - pad.t - pad.b)
  const o = past.length - 1                                           // index of the last real week
  const pastLine = past.map((p, i) => `${x(i)},${y(p.index)}`).join(' ')
  const mid = [`${x(o)},${y(past[o].index)}`, ...forecast.map((f, i) => `${x(o + 1 + i)},${y(f.p50)}`)].join(' ')
  const band = [...forecast.map((f, i) => `${x(o + 1 + i)},${y(f.p90)}`), ...[...forecast].reverse().map((f, i) => `${x(o + forecast.length - i)},${y(f.p10)}`)]
  const bandPath = `M${x(o)},${y(past[o].index)} L${band.join(' L')} Z`
  const ticks = [lo, Math.round((lo + hi) / 2), hi]
  const all = [...past.map(p => p.week), ...forecast.map(f => f.week)]
  const monthTicks = all.map((w, i) => ({ w, i })).filter(({ w }, i) => i === 0 || shortMonth(w) !== shortMonth(all[i - 1]))
  return (
    <figure className="chart forecast-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: last ${past.length} weeks and a 12-week forecast with its likely range`}>
        {ticks.map(t => (
          <g key={t}><line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="grid" /><text x={pad.l - 8} y={y(t) + 4} textAnchor="end">{t}</text></g>
        ))}
        {monthTicks.map(({ w, i }) => <text key={w} x={x(i)} y={H - 8} textAnchor="middle">{shortMonth(w)}</text>)}
        <path d={bandPath} className="fc-band" />
        <line x1={x(o)} x2={x(o)} y1={pad.t} y2={H - pad.b} className="fc-today" />
        <text x={x(o) + 6} y={pad.t + 12} className="fc-today-label">Forecast</text>
        <polyline points={pastLine} className="fc-past" />
        <polyline points={mid} className="fc-mid" />
      </svg>
      <table className="sr-only">
        <caption>{label}: 12-week forecast</caption>
        <thead><tr><th>Week</th><th>Low (P10)</th><th>Likely (P50)</th><th>High (P90)</th></tr></thead>
        <tbody>{forecast.map(f => <tr key={f.week}><td>{f.week}</td><td>{f.p10}</td><td>{f.p50}</td><td>{f.p90}</td></tr>)}</tbody>
      </table>
      <figcaption className="legend" aria-hidden="true">
        <span><i className="key-past" /> Market interest (100 = a typical market)</span><span><i className="key-mid" /> Likely path</span>
        <span><i className="key-band" /> Likely range (P10 to P90)</span>
      </figcaption>
    </figure>
  )
}
