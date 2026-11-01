import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ForecastChart } from '../components/Charts'
import { ErrorBox } from '../components/Form'
import { Icon } from '../components/Icons'
import { api, errorText, type Market as MarketData } from '../lib/api'
import { compact, pct } from '../lib/format'

const day = (iso: string) => new Intl.DateTimeFormat('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }).format(new Date(iso))
const weekOf = (iso: string) => new Intl.DateTimeFormat('en-IN', { day: 'numeric', month: 'long' }).format(new Date(iso))
const signed = (x: number) => (Math.abs(x) < 0.005 ? '0%' : `${x > 0 ? '+' : '−'}${pct(Math.abs(x))}`)
const tone = (x: number) => (Math.abs(x) < 0.03 ? undefined : x > 0 ? 'good' as const : 'bad' as const)

/** "Food and beverage brands (D2C)" -> "food and beverage brands"; keeps acronyms such as SaaS and AI. */
function sentenceCase(label: string) {
  const s = label.replace(/\s*\(.*\)$/, '')
  return /^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s
}

function headline(label: string, change: number) {
  const market = sentenceCase(label)
  if (Math.abs(change) < 0.03) return `Interest in ${market} is expected to hold steady over the next 12 weeks.`
  return `Interest in ${market} is expected to ${change > 0 ? 'rise' : 'ease'} ${pct(Math.abs(change))} over the next 12 weeks.`
}

function Kpi({ label, value, note, tone }: { label: string; value: string; note: string; tone?: 'good' | 'bad' }) {
  return (
    <div className="card stat">
      <span className="stat-label">{label}</span>
      <strong className={tone ?? ''}>{value}</strong>
      <span className="muted small">{note}</span>
    </div>
  )
}

export function Market() {
  const { id = '' } = useParams()
  const [data, setData] = useState<MarketData | null>(null)
  const [sectors, setSectors] = useState<{ key: string; label: string }[]>([])
  const [currency, setCurrency] = useState('INR')
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const load = useCallback(() => {
    Promise.all([api.market(id), api.marketSectors(), api.startup(id)])
      .then(([m, s, st]) => { setData(m); setSectors(s); setCurrency(st.currency) })
      .catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  async function pick(key: string) {
    setSaving(true)
    try { await api.setMarketSector(id, key); load() } catch (e) { setError(errorText(e)) } finally { setSaving(false) }
  }

  if (error) return (
    <div className="narrow"><ErrorBox error={error} />
      <div className="form-actions"><button className="btn primary" onClick={() => { setError(null); load() }}>Try again</button></div>
    </div>
  )
  if (!data) return <p className="muted" role="status" aria-busy="true">Loading the market outlook…</p>

  const picker = (
    <label className="market-picker">
      <span>Your market</span>
      <select value={data.sector ?? ''} disabled={saving} onChange={e => pick(e.target.value)}>
        {!data.sector && <option value="" disabled>Choose a market…</option>}
        {sectors.map(s => <option key={s.key} value={s.key}>{s.label}{s.key === data.suggested && !data.chosen ? ' (suggested)' : ''}</option>)}
      </select>
      {!data.chosen && data.suggested && <span className="small">Suggested from your startup’s sector. Pick another if it fits better.</span>}
    </label>
  )

  if (!data.available || !data.outlook) return (
    <div>
      <header className="editorial">
        <div className="kicker"><span>Market outlook</span></div>
        <h1>Where is your market heading?</h1>
        <p className="lede">{data.message}</p>
        {picker}
      </header>
    </div>
  )

  const o = data.outlook
  const v = data.you_vs_market
  const better = o.model === 'network' ? o.backtest.network_mase : o.backtest.smoothing_mase
  const other = o.model === 'network' ? o.backtest.smoothing_mase : o.backtest.network_mase
  return (
    <div>
      <header className="editorial">
        <div className="kicker">
          <span>Market outlook</span><span className="sep" /><span>{o.label}</span>
          <span className="sep kx" /><span className="kx">Data through {day(data.data_through!)}</span>
        </div>
        <h1>{headline(o.label, o.change_12w)}</h1>
        <p className="lede">
          Likely between {signed(o.change_12w_low)} and {signed(o.change_12w_high)}. Interest is expected to peak in the week of {weekOf(o.peak_week)}.
          {' '}{v?.summary ?? ''}
        </p>
        {picker}
      </header>

      <section className="card">
        <h2>Interest in your market, and the next 12 weeks</h2>
        <ForecastChart history={o.history} forecast={o.forecast} label={o.label} />
      </section>

      <div className="market-tiles">
        <Kpi label="Next 4 weeks" value={signed(o.change_4w)} note="likely change against the last 4 weeks" tone={tone(o.change_4w)} />
        <Kpi label="Next 12 weeks" value={signed(o.change_12w)} note={`range ${signed(o.change_12w_low)} to ${signed(o.change_12w_high)}`} tone={tone(o.change_12w)} />
        <Kpi label="Peak week" value={weekOf(o.peak_week)} note="highest likely interest in the next 12 weeks" />
        <Kpi label="Track record" value={`${Math.round((1 - better / other) * 100)}% better`}
          note={`${o.model_label} vs ${o.model === 'network' ? 'exponential smoothing' : 'the deep learning ensemble'} on 12 held-out weeks`} />
      </div>

      <div className="two-col">
        <section className="card">
          <h2>You vs your market</h2>
          <p className="muted" style={{ margin: 0 }}>{v?.summary}</p>
          {v && v.months.length > 0 && (
            <table className="vs-table">
              <thead><tr><th>Month</th><th>Your revenue</th><th>Market interest</th></tr></thead>
              <tbody>{v.months.slice(-6).map(m => <tr key={m.month}><td>{m.month}</td><td>{compact(m.revenue, currency)}</td><td>{m.market.toFixed(0)}</td></tr>)}</tbody>
            </table>
          )}
          {v?.correlation != null && <p className="small muted">Month-to-month, your revenue and your market move together with a correlation of {v.correlation}.</p>}
          {v && v.months.length < 2 && <Link className="btn small" to={`/startups/${id}/transactions`}><Icon name="upload" size={14} /> Upload more months</Link>}
        </section>
        <section className="card">
          <h2>How this forecast is made</h2>
          <ul className="about-list">
            <li>Market interest is read from daily Wikipedia page views of about 170 topic and brand articles across 19 sectors, 2023 to today (public domain data).</li>
            <li>It is measured against all of Wikipedia and against the other sectors, so readers moving to AI chatbots does not look like every market shrinking.</li>
            <li>Three neural networks (MLP, 1D CNN and LSTM with attention) forecast it; for each sector the app uses whichever was more accurate on the last 12 held-out weeks: here, {o.model_label.toLowerCase()}.</li>
            <li>It shows attention, not sales. Use it for timing (stock, campaigns, hiring), not as a revenue forecast.</li>
          </ul>
        </section>
      </div>
    </div>
  )
}
