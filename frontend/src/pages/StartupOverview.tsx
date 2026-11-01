import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Bars, MonthlyChart } from '../components/Charts'
import { ErrorBox } from '../components/Form'
import { StartupNav } from '../components/StartupNav'
import { api, errorText, PRODUCT_TYPES, STAGES, type Activity, type Dashboard, type Runway, type Startup } from '../lib/api'
import { compact, count, monthLabel, pct } from '../lib/format'
import { StartupForm } from './StartupForm'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const ACTIONS: Record<string, string> = {
  'startup.create': 'Startup created', 'startup.update': 'Details changed', 'bank_account.create': 'Bank account added',
  'statement.import': 'Statement uploaded', 'transaction.categorise': 'Transaction categorised',
  'unit_metrics.save': 'Monthly numbers saved', 'cap_table.setup': 'Cap table set up', 'round.record': 'Funding round recorded',
  'safe.create': 'SAFE added', 'options.grant': 'Options granted', 'deal.create': 'Investor added to pipeline',
  'deal.update': 'Pipeline updated',
}

function when(iso: string) {
  return new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export function runwayText(r: Runway | undefined): { value: string; note: string } {
  if (!r) return { value: '—', note: '' }
  if (r.status === 'burning') return { value: `${r.months} months`, note: 'at the last 3 months’ average burn' }
  if (r.status === 'not_burning') return { value: 'Not burning', note: 'cash coming in covers cash going out' }
  if (r.status === 'no_balance') return { value: 'Needs cash balance', note: 'add an opening balance in Financials' }
  return { value: '—', note: 'upload a statement' }
}

function Kpi({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: 'good' | 'bad' }) {
  return (
    <div className="card stat">
      <span className="muted">{label}</span>
      <strong className={tone ?? ''}>{value}</strong>
      {note && <span className="muted small">{note}</span>}
    </div>
  )
}

export function StartupOverview() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [dash, setDash] = useState<Dashboard | null>(null)
  const [activity, setActivity] = useState<Activity[]>([])
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    Promise.all([api.startup(id), api.dashboard(id), api.activity(id)])
      .then(([s, d, a]) => { setStartup(s); setDash(d); setActivity(a) })
      .catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  if (error) return <div className="narrow"><ErrorBox error={error} /><Link to="/">Back to your startups</Link></div>
  if (!startup || !dash) return <p className="muted">Loading…</p>
  const c = dash.currency
  const k = dash.kpis
  const run = runwayText(dash.runway)

  return (
    <div>
      <StartupNav id={id} name={startup.name} />
      <div className="page-head">
        <div>
          <h1>{dash.has_data && dash.month ? monthLabel(dash.month, true) : 'Overview'}</h1>
          <p className="muted">{startup.sector} · {STAGES[startup.stage]} · {PRODUCT_TYPES[startup.product_type]}</p>
        </div>
        {!editing && <button className="btn" onClick={() => setEditing(true)}>Edit details</button>}
      </div>

      {editing && (
        <div className="card" style={{ marginBottom: 16 }}>
          <StartupForm initial={startup} submitLabel="Save changes" onSubmit={async data => {
            setStartup(await api.updateStartup(startup.id, data)); setEditing(false); load()
          }} />
          <button className="btn ghost" onClick={() => setEditing(false)}>Cancel</button>
        </div>
      )}

      {!dash.has_data ? (
        <div className="card empty">
          <h2>Start with your bank statement</h2>
          <p className="muted">Upload a statement and this page fills in: revenue, expenses, EBITDA, burn and how many months of cash you have.</p>
          <Link className="btn primary" to={`/startups/${id}/transactions`}>Go to Transactions</Link>
        </div>
      ) : (
        <>
          {(dash.needs_review ?? 0) > 0 && (
            <div className="banner">
              <span><strong>{dash.needs_review} transaction{dash.needs_review === 1 ? '' : 's'} need{dash.needs_review === 1 ? 's' : ''} your review.</strong> Until then they count as “not categorised yet”, so the numbers below are provisional.</span>
              <Link className="btn small" to={`/startups/${id}/transactions`}>Review</Link>
            </div>
          )}
          <div className="stats stats-5">
            <Kpi label="Revenue" value={compact(k?.revenue, c)}
              note={k?.revenue_growth != null ? `${k.revenue_growth >= 0 ? '+' : ''}${pct(k.revenue_growth)} vs last month` : undefined}
              tone={k?.revenue_growth != null ? (k.revenue_growth >= 0 ? 'good' : 'bad') : undefined} />
            <Kpi label="Gross margin" value={pct(k?.gross_margin)} note={k?.gross_margin == null ? 'no revenue this month' : undefined} />
            <Kpi label="EBITDA" value={compact(k?.ebitda, c)} tone={(k?.ebitda ?? 0) >= 0 ? 'good' : 'bad'} />
            <Kpi label="Net burn" value={(k?.burn ?? 0) > 0 ? compact(k?.burn, c) : 'None'}
              note={(k?.burn ?? 0) > 0 ? 'cash out minus cash in, excluding money raised' : 'more cash came in than went out'} />
            <Kpi label="Cash runway" value={run.value} note={run.note}
              tone={dash.runway?.status === 'burning' && (dash.runway.months ?? 99) < 6 ? 'bad' : undefined} />
          </div>
          <div className="two-col wide-left">
            <section className="card">
              <h2>Last {dash.series?.length} months</h2>
              <MonthlyChart data={dash.series ?? []} currency={c} />
            </section>
            <section className="card">
              <h2>Biggest expenses this month</h2>
              {dash.top_expenses?.length ? <Bars items={dash.top_expenses} currency={c} /> : <p className="muted">No expenses this month.</p>}
              <Link className="small" to={`/startups/${id}/financials`}>Full P&amp;L →</Link>
            </section>
          </div>
          <section className="card" style={{ marginTop: 16 }}>
            <div className="page-head" style={{ marginBottom: 8 }}>
              <h2>Unit economics</h2>
              <Link className="btn small" to={`/startups/${id}/financials#units`}>{dash.units_entered ? 'Edit numbers' : 'Add this month’s numbers'}</Link>
            </div>
            {startup.product_type === 'physical' ? (
              <div className="metric-row">
                <div><span className="muted">Orders</span><strong>{count(dash.units?.orders)}</strong></div>
                <div><span className="muted">Average order value</span><strong>{compact(dash.units?.average_order_value, c)}</strong></div>
                <div><span className="muted">Cost of goods per unit</span><strong>{compact(dash.units?.cogs_per_unit, c)}</strong></div>
                <div><span className="muted">Months of stock</span><strong>{dash.units?.inventory_months?.toFixed(1) ?? '—'}</strong></div>
              </div>
            ) : (
              <div className="metric-row">
                <div><span className="muted">Customers</span><strong>{count(dash.units?.customers)}</strong></div>
                <div><span className="muted">Monthly churn</span><strong>{pct(dash.units?.churn_rate, 1)}</strong></div>
                <div><span className="muted">CAC</span><strong>{compact(dash.units?.cac, c)}</strong></div>
                <div><span className="muted">LTV : CAC</span><strong>{dash.units?.ltv_to_cac?.toFixed(1) ?? '—'}</strong></div>
              </div>
            )}
          </section>
        </>
      )}

      <div className="two-col" style={{ marginTop: 16 }}>
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
          <h2>Recent activity</h2>
          <ul className="activity">
            {activity.slice(0, 8).map((a, i) => (
              <li key={i}><span>{ACTIONS[a.action] ?? a.action}</span><time className="muted">{when(a.created_at)}</time></li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}
