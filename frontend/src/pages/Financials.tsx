import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ErrorBox, Field } from '../components/Form'
import { Icon } from '../components/Icons'
import { api, errorText, type BankAccount, type Pnl, type PnlMonth, type Startup, type UnitMetricRow } from '../lib/api'
import { compact, count, money, monthLabel, pct, today } from '../lib/format'
import { runwayText } from './StartupOverview'

type Row = { label: string; value: (m: PnlMonth) => string; strong?: boolean; tone?: (m: PnlMonth) => string; indent?: boolean
  category?: string }  // the transactions behind each cell are one click away

const PHYSICAL = [['units_sold', 'Units sold'], ['orders', 'Orders'], ['inventory_value', 'Stock on hand (at cost)']] as const
const DIGITAL = [['customers', 'Paying customers at month end'], ['new_customers', 'New customers'],
  ['churned_customers', 'Customers lost'], ['mrr', 'MRR at month end']] as const

function UnitForm({ startup, rows, onSaved }: { startup: Startup; rows: UnitMetricRow[]; onSaved: () => void }) {
  const fields = startup.product_type === 'physical' ? PHYSICAL : DIGITAL
  const [month, setMonth] = useState(rows[0]?.month ?? today().slice(0, 7))
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const current = rows.find(r => r.month === month)

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    const data = Object.fromEntries(fields.map(([k]) => [k, f.get(k) === '' ? null : Number(f.get(k))]))
    setError(null); setSaved(false)
    try { await api.saveUnitMetrics(startup.id, month, data); setSaved(true); onSaved() } catch (err) { setError(errorText(err)) }
  }

  const c = current?.computed
  return (
    <section className="card" id="units">
      <h2>Unit economics</h2>
      <p className="muted">The few numbers a bank statement cannot know. About a minute a month; everything else is calculated.</p>
      <ErrorBox error={error} />
      <form className="unit-form" onSubmit={submit} key={month}>
        <Field label="Month"><input type="month" value={month} max={today().slice(0, 7)} onChange={e => setMonth(e.target.value)} /></Field>
        {fields.map(([k, l]) => (
          <Field key={k} label={l}><input name={k} type="number" min={0} step="any" defaultValue={(current?.[k] as number | null) ?? ''} /></Field>
        ))}
        <div className="form-actions"><button className="btn primary">Save</button>{saved && <span className="muted small"> Saved.</span>}</div>
      </form>
      {c && (
        <div className="metric-row" style={{ marginTop: 12 }}>
          {startup.product_type === 'physical' ? (<>
            <div><span className="muted">Average order value</span><strong>{compact(c.average_order_value, startup.currency)}</strong></div>
            <div><span className="muted">Revenue per unit</span><strong>{compact(c.revenue_per_unit, startup.currency)}</strong></div>
            <div><span className="muted">Cost of goods per unit</span><strong>{compact(c.cogs_per_unit, startup.currency)}</strong></div>
            <div><span className="muted">Months of stock</span><strong>{c.inventory_months?.toFixed(1) ?? '—'}</strong></div>
          </>) : (<>
            <div><span className="muted">Monthly churn</span><strong>{pct(c.churn_rate, 1)}</strong></div>
            <div><span className="muted">ARPU</span><strong>{compact(c.arpu, startup.currency)}</strong></div>
            <div><span className="muted">CAC</span><strong>{compact(c.cac, startup.currency)}</strong></div>
            <div><span className="muted">LTV</span><strong>{compact(c.ltv, startup.currency)}</strong></div>
            <div><span className="muted">LTV : CAC</span><strong>{c.ltv_to_cac?.toFixed(1) ?? '—'}</strong></div>
            <div><span className="muted">Payback</span><strong>{c.payback_months ? `${c.payback_months.toFixed(1)} mo` : '—'}</strong></div>
          </>)}
        </div>
      )}
      <p className="muted small">CAC uses this month’s marketing spend from the P&amp;L; LTV uses gross margin and churn.</p>
    </section>
  )
}

function OpeningBalance({ startupId, accounts, onSaved }: { startupId: string; accounts: BankAccount[]; onSaved: () => void }) {
  const [error, setError] = useState<string | null>(null)
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    try {
      await Promise.all(accounts.map(a => f.get(a.id) !== '' ? api.setOpeningBalance(startupId, a.id, Number(f.get(a.id))) : null))
      onSaved()
    } catch (err) { setError(errorText(err)) }
  }
  return (
    <form className="banner" onSubmit={submit}>
      <div>
        <strong>Your statement has no balance column.</strong> Enter each account’s balance before its first imported line, and cash and runway are worked out from there.
        <ErrorBox error={error} />
        <div className="inline-fields">
          {accounts.map(a => <Field key={a.id} label={a.name}><input name={a.id} type="number" step="any" /></Field>)}
          <button className="btn primary small">Save</button>
        </div>
      </div>
    </form>
  )
}

export function Financials() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [pnl, setPnl] = useState<Pnl | null>(null)
  const [units, setUnits] = useState<UnitMetricRow[]>([])
  const [accounts, setAccounts] = useState<BankAccount[]>([])
  const [showAll, setShowAll] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    Promise.all([api.startup(id), api.pnl(id), api.unitMetrics(id), api.bankAccounts(id)])
      .then(([s, p, u, a]) => { setStartup(s); setPnl(p); setUnits(u); setAccounts(a) })
      .catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  const months = useMemo(() => (pnl ? (showAll ? pnl.months : pnl.months.slice(-6)) : []), [pnl, showAll])
  const opexKeys = useMemo(() => {
    const seen = new Map<string, string>()
    months.forEach(m => m.opex.forEach(o => seen.set(o.key, o.label)))
    return [...seen.entries()]
  }, [months])

  if (error) return <div className="narrow"><ErrorBox error={error} /><Link to="/">Back</Link></div>
  if (!startup || !pnl) return <p className="muted" role="status" aria-busy="true">Loading…</p>
  const c = pnl.currency
  const m = (n: number | null) => money(n, c)
  const run = runwayText(pnl.runway)

  const rows: (Row | string)[] = [
    'Profit and loss',
    { label: 'Revenue', value: x => m(x.revenue), strong: true, category: 'revenue' },
    { label: 'Cost of goods', value: x => m(x.cogs), category: 'cogs' },
    { label: 'Gross profit', value: x => m(x.gross_profit), strong: true },
    { label: 'Gross margin', value: x => pct(x.gross_margin) },
    ...opexKeys.map(([key, label]): Row => ({ label, indent: true, value: x => m(x.opex.find(o => o.key === key)?.amount ?? 0),
      tone: () => (key === 'uncategorised' ? 'warn' : ''), category: key })),
    { label: 'Operating expenses', value: x => m(x.opex_total), strong: true },
    { label: 'EBITDA', value: x => m(x.ebitda), strong: true, tone: x => (x.ebitda >= 0 ? 'good' : 'bad') },
    { label: 'Other income (interest, dividends)', value: x => m(x.other_income) },
    'Cash flow',
    { label: 'Cash in', value: x => m(x.cash_in) },
    { label: 'Cash out', value: x => m(x.cash_out) },
    { label: 'Net burn', value: x => m(x.burn), tone: x => (x.burn > 0 ? 'bad' : 'good') },
    { label: 'Money raised', value: x => m(x.financing_in), indent: true },
    { label: 'GST and TDS paid', value: x => m(x.taxes_paid), indent: true },
    { label: 'Loan repayments', value: x => m(x.loan_repaid), indent: true },
    { label: 'Cash at month end', value: x => m(x.closing_cash), strong: true },
    { label: 'Lines needing review', value: x => count(x.needs_review), tone: x => (x.needs_review ? 'warn' : '') },
  ]

  return (
    <div>
      <div className="page-head">
        <div><h1>Financials</h1><p className="muted">Built from your categorised transactions. Correct a transaction and the numbers update.</p></div>
        <div className="card stat runway-tile"><span className="stat-label">Cash runway</span><strong>{run.value}</strong><span className="muted small">{run.note}</span></div>
      </div>
      {pnl.runway.status === 'no_balance' && accounts.length > 0 && <OpeningBalance startupId={id} accounts={accounts} onSaved={load} />}
      {pnl.months.length === 0 ? (
        <div className="card empty"><span className="empty-icon"><Icon name="financials" size={22} /></span><h2>No months to show yet</h2>
          <p className="muted">Upload a bank statement and every month it covers appears here as a P&amp;L and cash-flow column.</p>
          <Link className="btn primary" to={`/startups/${id}/transactions`}>Go to Transactions</Link></div>
      ) : (
        <section className="card table-card">
          <div className="page-head" style={{ marginBottom: 8 }}>
            <h2>Monthly P&amp;L and cash flow</h2>
            {pnl.months.length > 6 && <button className="btn small" onClick={() => setShowAll(s => !s)}>{showAll ? 'Last 6 months' : `All ${pnl.months.length} months`}</button>}
          </div>
          <div className="table-scroll">
            <table className="pnl">
              <thead><tr><th />{months.map(x => <th key={x.month}>{monthLabel(x.month)}</th>)}</tr></thead>
              <tbody>
                {rows.map(r => typeof r === 'string'
                  ? <tr key={r} className="section"><td colSpan={months.length + 1}>{r}</td></tr>
                  : (
                    <tr key={r.label} className={`${r.strong ? 'strong' : ''} ${r.indent ? 'indent' : ''}`}>
                      <td>{r.label}</td>
                      {months.map(x => <td key={x.month} className={r.tone?.(x) ?? ''}>{r.category
                        ? <Link className="cell-link" title="See these transactions"
                            to={`/startups/${id}/transactions?month=${x.month.slice(0, 7)}&category=${r.category}`}>{r.value(x)}</Link>
                        : r.value(x)}</td>)}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">GST, TDS, loan principal and money raised move cash but are not income or expenses, so they appear only under cash flow.
            Payments to people not yet reviewed count as people costs.</p>
        </section>
      )}
      <UnitForm startup={startup} rows={units} onSaved={load} />
    </div>
  )
}
