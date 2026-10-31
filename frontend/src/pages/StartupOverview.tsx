import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ErrorBox } from '../components/Form'
import { api, errorText, PRODUCT_TYPES, STAGES, type Activity, type Startup } from '../lib/api'
import { StartupForm } from './StartupForm'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const ACTIONS: Record<string, string> = { 'startup.create': 'Startup created', 'startup.update': 'Details changed' }
// What later phases add to this page, so the founder can see where the product is going.
const COMING = [
  ['Transactions', 'Upload a bank statement; every line is categorised automatically.'],
  ['Financials', 'Monthly P&L: revenue, gross profit, EBITDA, burn and runway.'],
  ['Cap table', 'Shareholders, rounds, SAFEs, ESOP and a dilution simulator.'],
  ['Forecast', 'Revenue and cash for the next 12 months, with a range.'],
]

function when(iso: string) {
  return new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export function StartupOverview() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [activity, setActivity] = useState<Activity[]>([])
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    Promise.all([api.startup(id), api.activity(id)])
      .then(([s, a]) => { setStartup(s); setActivity(a) })
      .catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  if (error) return <div className="narrow"><ErrorBox error={error} /><Link to="/">Back to your startups</Link></div>
  if (!startup) return <p className="muted">Loading…</p>

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>{startup.name}</h1>
          <p className="muted">{startup.sector} · {STAGES[startup.stage]} · {PRODUCT_TYPES[startup.product_type]}</p>
        </div>
        {!editing && <button className="btn" onClick={() => setEditing(true)}>Edit details</button>}
      </div>

      {editing ? (
        <div className="card">
          <StartupForm initial={startup} submitLabel="Save changes" onSubmit={async data => {
            setStartup(await api.updateStartup(startup.id, data)); setEditing(false); load()
          }} />
          <button className="btn ghost" onClick={() => setEditing(false)}>Cancel</button>
        </div>
      ) : (
        <div className="two-col">
          <section className="card">
            <h2>Company</h2>
            <dl className="facts">
              <dt>Founded</dt><dd>{startup.founded_on ? new Date(startup.founded_on).toLocaleDateString('en-IN', { year: 'numeric', month: 'long' }) : '—'}</dd>
              <dt>Currency</dt><dd>{startup.currency}</dd>
              <dt>Financial year</dt><dd>{MONTHS[startup.fy_start_month - 1]} to {MONTHS[(startup.fy_start_month + 10) % 12]}</dd>
              <dt>Your role</dt><dd className="cap">{startup.role}</dd>
            </dl>
          </section>
          <section className="card">
            <h2>Activity</h2>
            <ul className="activity">
              {activity.map((a, i) => (
                <li key={i}>
                  <span>{ACTIONS[a.action] ?? a.action}
                    {a.action === 'startup.update' && <span className="muted"> ({Object.keys(a.details).map(k => k.replaceAll('_', ' ')).join(', ')})</span>}
                  </span>
                  <time className="muted">{when(a.created_at)}</time>
                </li>
              ))}
            </ul>
          </section>
        </div>
      )}

      <h2 className="section-title">Coming next</h2>
      <div className="grid">
        {COMING.map(([t, d]) => <div key={t} className="card tile soon"><strong>{t}</strong><span className="muted">{d}</span></div>)}
      </div>
    </div>
  )
}
