import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Bars, MonthlyChart } from '../components/Charts'
import { ErrorBox } from '../components/Form'
import { Icon, type IconName } from '../components/Icons'
import { api, errorText, PRODUCT_TYPES, STAGES, type Activity, type Dashboard, type Runway, type Startup } from '../lib/api'
import { compact, count, monthLabel, pct } from '../lib/format'
import { StartupForm } from './StartupForm'

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

/** A supporting tile: label, the figure in the serif, and one small visual that belongs to this number. */
function Kpi({ label, value, note, tone, delta, meter, meterTone, meterLabel }: {
  label: string; value: string; note?: string; tone?: 'good' | 'bad'; icon?: IconName; color?: string; delta?: number | null
  meter?: number | null; meterTone?: 'good' | 'warn' | 'bad' | 'accent'; meterLabel?: string
}) {
  return (
    <div className="card stat">
      <span className="stat-label">{label}</span>
      <strong className={tone ?? ''}>{value}</strong>
      {meter != null && (
        <span className={`meter meter-${meterTone ?? 'accent'}`} role="img" aria-label={`${meterLabel ?? label} ${Math.round(Math.max(0, Math.min(1, meter)) * 100)}%`}>
          <i style={{ width: `${Math.max(2, Math.min(100, meter * 100))}%` }} />
        </span>
      )}
      {delta != null && <span className={`delta ${delta >= 0 ? 'up' : 'down'}`}>{delta >= 0 ? '▲' : '▼'} {pct(Math.abs(delta))} vs last month</span>}
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
    Promise.all([api.startup(id), api.dashboard(id), api.activity(id).catch(() => [])])
      .then(([s, d, a]) => { setStartup(s); setDash(d); setActivity(a) })
      .catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  if (error) return (
    <div className="narrow"><ErrorBox error={error} />
      <div className="form-actions"><button className="btn primary" onClick={() => { setError(null); load() }}>Try again</button><Link to="/">Back to your startups</Link></div>
    </div>
  )
  if (!startup || !dash) return <p className="muted" role="status" aria-busy="true">Loading the brief…</p>
  const c = dash.currency
  const k = dash.kpis
  const run = runwayText(dash.runway)
  const series = (dash.series ?? []).slice(-6)
  const maxRev = Math.max(1, ...series.map(p => p.revenue))
  const month = dash.month ? monthLabel(dash.month, true) : ''
  const prev = series.length > 1 ? monthLabel(series[series.length - 2].month, true).split(' ')[0] : null
  const burning = dash.runway?.status === 'burning'
  const toReview = dash.needs_review ?? 0
  const noCogs = (k?.revenue ?? 0) > 0 && k?.gross_margin === 1
  const deltaText = (g: number) => `${g >= 0 ? 'up' : 'down'} ${pct(Math.abs(g))} vs ${prev ?? 'last month'}`
  const headline = !dash.has_data
    ? `Welcome to ${startup.name}. Let’s get your numbers in.`
    : (k?.revenue ?? 0) > 0
      ? `${startup.name} earned ${compact(k?.revenue, c)} in ${month.split(' ')[0]}${burning ? ` and burned ${compact(k?.burn, c)}.` : ' and kept more cash than it spent.'}`
      : `${startup.name} had no revenue in ${month.split(' ')[0]}${burning ? ` and burned ${compact(k?.burn, c)}.` : '.'}`
  const lede = !dash.has_data
    ? `${PRODUCT_TYPES[startup.product_type]} · ${startup.sector}. Upload a bank statement and this page becomes your monthly brief: revenue, EBITDA, burn and runway.`
    : [
        k?.revenue_growth != null && prev ? `Revenue was ${k.revenue_growth >= 0 ? 'up' : 'down'} ${pct(Math.abs(k.revenue_growth))} on ${prev}.` : null,
        burning ? `At this pace, cash lasts about ${dash.runway?.months} months.` : dash.runway?.status === 'not_burning' ? 'Cash coming in covered everything going out.' : null,
        (dash.needs_review ?? 0) > 0 ? `${dash.needs_review} transactions still need sorting, so treat these figures as provisional.` : null,
      ].filter(Boolean).join(' ')

  return (
    <div>
      <header className="editorial">
        <div className="kicker">
          <span>{dash.has_data && dash.month ? `Monthly brief · ${monthLabel(dash.month, true)}` : 'Getting started'}</span>
          {toReview > 0 && <><span className="sep" /><span>Provisional · {toReview} to sort</span></>}
          <span className="sep kx" /><span className="kx">{STAGES[startup.stage]}</span><span className="sep kx" /><span className="kx">{startup.sector}</span>
        </div>
        <h1>{headline}</h1>
        <p className="lede">{lede}</p>
        <div className="actions">
          {toReview > 0
            ? <><Link className="btn primary" to={`/startups/${id}/transactions`}>Review {toReview} transaction{toReview === 1 ? '' : 's'}</Link>
                <Link className="btn" to={`/startups/${id}/transactions`}><Icon name="upload" size={16} /> Upload statement</Link></>
            : <Link className="btn primary" to={`/startups/${id}/transactions`}><Icon name="upload" size={16} /> Upload statement</Link>}
          {!editing && <button className="btn ghost" onClick={() => setEditing(true)}>Edit details</button>}
        </div>
      </header>

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
          <span className="empty-icon"><Icon name="upload" size={22} /></span>
          <h2>Start with your bank statement</h2>
          <p className="muted">Upload a statement and this page fills in: revenue, expenses, EBITDA, burn and how many months of cash you have.</p>
          <Link className="btn primary" to={`/startups/${id}/transactions`}>Go to Transactions</Link>
        </div>
      ) : (
        <>
          <div className="bento">
            <div className="card stat b-hero">
              <div>
                <span className="stat-label">Revenue · {monthLabel(dash.month!, true)}</span>
                <strong>{compact(k?.revenue, c)}</strong>
                {k?.revenue_growth != null && <span className={`delta ${k.revenue_growth >= 0 ? 'up' : 'down'}`}><span aria-hidden="true">{k.revenue_growth >= 0 ? '▲' : '▼'}</span> {deltaText(k.revenue_growth)}</span>}
              </div>
              <div>
                <div className="spark" role="img" aria-label={`Revenue by month: ${series.map(p => `${monthLabel(p.month)} ${compact(p.revenue, c)}`).join(', ')}`}>
                  {series.map(p => <span key={p.month} aria-hidden="true" style={{ height: `${Math.max(4, (p.revenue / maxRev) * 100)}%` }} title={`${monthLabel(p.month)} ${compact(p.revenue, c)}`} />)}
                </div>
                <div className="spark-labels" aria-hidden="true">{series.map(p => <span key={p.month}>{monthLabel(p.month)}</span>)}</div>
              </div>
            </div>
            {toReview > 0 && (
              <div className="card stat b-review">
                <span className="stat-label">Needs your review</span>
                <strong>{toReview}</strong>
                <span className="muted small">lines unsorted, so these figures are provisional</span>
                <Link className="btn small" to={`/startups/${id}/transactions`}>Review now</Link>
              </div>
            )}
            <Kpi label="EBITDA" value={compact(k?.ebitda, c)} tone={(k?.ebitda ?? 0) >= 0 ? 'good' : 'bad'}
              note={k?.revenue ? `${pct(Math.abs((k?.ebitda ?? 0) / k.revenue))} of revenue ${(k?.ebitda ?? 0) >= 0 ? 'kept as operating profit' : 'lost before tax and interest'}` : 'operating profit'} />
            <Kpi label="Gross margin" value={noCogs ? '—' : pct(k?.gross_margin)} meter={noCogs ? null : k?.gross_margin ?? null} meterLabel="Gross margin"
              note={k?.gross_margin == null ? 'no revenue this month' : noCogs ? 'No cost of goods filed yet. Mark stock and packaging lines as cost of goods in Transactions.' : 'of revenue left after cost of goods'} />
            <Kpi label={toReview > 0 ? 'Cash · runway' : 'Net burn'}
              value={toReview > 0 ? run.value : (k?.burn ?? 0) > 0 ? compact(k?.burn, c) : 'None'}
              meter={burning ? Math.min(1, (dash.runway?.months ?? 0) / 24) : null} meterLabel="Runway out of 24 months"
              meterTone={burning && (dash.runway?.months ?? 99) < 6 ? 'bad' : 'warn'}
              note={toReview > 0 ? ((k?.burn ?? 0) > 0 ? `${compact(k?.burn, c)} net burn this month` : 'no net burn this month') : (k?.burn ?? 0) > 0 ? 'excluding money raised' : 'more cash came in than went out'}
              tone={burning && (dash.runway?.months ?? 99) < 6 ? 'bad' : undefined} />
            {toReview === 0 && <Kpi label="Cash runway" value={run.value} note={run.note} meter={burning ? Math.min(1, (dash.runway?.months ?? 0) / 24) : null} meterLabel="Runway out of 24 months" meterTone="warn" />}
            <section className="card b-chart b-tall">
              <h2>Revenue, expenses and cash</h2>
              <MonthlyChart data={dash.series ?? []} currency={c} />
            </section>
            <section className="card b-tall">
              <h2>Where the money went</h2>
              {dash.top_expenses?.length ? <Bars items={dash.top_expenses} currency={c} /> : <p className="muted">No expenses this month.</p>}
              <Link className="small" to={`/startups/${id}/financials`}>Full P&amp;L →</Link>
            </section>
          </div>
          <section className="card" style={{ marginTop: 18 }}>
            <div className="page-head" style={{ marginBottom: 8 }}>
              <h2>Unit economics</h2>
              <Link className="btn small" to={`/startups/${id}/financials#units`}>{dash.units_entered ? 'Edit numbers' : 'Add this month’s numbers'}</Link>
            </div>
            {!dash.units_entered ? (
              <p className="muted" style={{ margin: 0 }}>A bank statement cannot see {startup.product_type === 'physical' ? 'orders, units sold or stock on hand' : 'customers, churn or MRR'}. Add them once a month and
                StartupIQ works out {startup.product_type === 'physical' ? 'average order value, cost per unit and months of stock' : 'churn, CAC, LTV and payback'} from your P&amp;L.</p>
            ) : startup.product_type === 'physical' ? (
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

      <div className="two-col" style={{ marginTop: 18 }}>
        <section className="card">
          <h2>{dash.month ? `Closing ${monthLabel(dash.month, true)}` : 'Getting set up'}</h2>
          <ul className="checklist">
            <li className={dash.has_data ? 'done' : ''}><span className="tick" aria-hidden="true">{dash.has_data ? '✓' : ''}</span>
              <span>{dash.has_data ? 'Bank statement uploaded' : <Link to={`/startups/${id}/transactions`}>Upload a bank statement</Link>}</span></li>
            <li className={dash.has_data && toReview === 0 ? 'done' : ''}><span className="tick" aria-hidden="true">{dash.has_data && toReview === 0 ? '✓' : ''}</span>
              <span>{toReview > 0 ? <Link to={`/startups/${id}/transactions`}>Sort {toReview} transaction{toReview === 1 ? '' : 's'}</Link> : 'Every transaction sorted'}</span></li>
            <li className={dash.runway?.status !== 'no_balance' && dash.has_data ? 'done' : ''}><span className="tick" aria-hidden="true">{dash.runway?.status !== 'no_balance' && dash.has_data ? '✓' : ''}</span>
              <span>{dash.runway?.status === 'no_balance' ? <Link to={`/startups/${id}/financials`}>Add an opening cash balance</Link> : 'Cash balance known'}</span></li>
            <li className={dash.units_entered ? 'done' : ''}><span className="tick" aria-hidden="true">{dash.units_entered ? '✓' : ''}</span>
              <span>{dash.units_entered ? 'Unit numbers added' : <Link to={`/startups/${id}/financials#units`}>Add this month’s unit numbers</Link>}</span></li>
          </ul>
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
