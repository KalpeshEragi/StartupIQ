import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ErrorBox } from '../components/Form'
import { Icon } from '../components/Icons'
import { api, errorText, type CheckStatus, type Startup, type Verification } from '../lib/api'
import { count, money, monthLabel, pct } from '../lib/format'

const CHECK: Record<CheckStatus, string> = { verified: 'Verified', warnings: 'Check needed', unverified: 'Not checked' }

export function CheckBadge({ status }: { status: CheckStatus }) {
  return <span className={`check check-${status}`}>{status === 'verified' ? '✓ ' : status === 'warnings' ? '! ' : ''}{CHECK[status]}</span>
}

function day(iso: string | null) {
  return iso ? new Date(iso + 'T00:00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—'
}
const share = (n: number, total: number) => (total ? pct(n / total) : '—')

export function Trust() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [data, setData] = useState<Verification | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.startup(id), api.verification(id)])
      .then(([s, v]) => { setStartup(s); setData(v) })
      .catch(e => setError(errorText(e)))
  }, [id])

  if (error) return <div className="narrow"><ErrorBox error={error} /><Link to="/">Back</Link></div>
  if (!startup || !data) return <p className="muted" role="status" aria-busy="true">Loading…</p>
  const { lines, audit } = data
  const c = startup.currency

  return (
    <div className="trust">
      <div className="page-head">
        <div>
          <h1>Trust</h1>
          <p className="muted">Whether your statements add up, and how every number was made. Print it for investors.</p>
        </div>
        <button className="btn no-print" onClick={() => window.print()}>Print report</button>
      </div>
      <p className="print-only"><strong>{startup.name}</strong>: verification report, {new Date(data.generated_at).toLocaleString('en-IN')}</p>

      <div className={`card trust-status trust-${data.status}`}>
        <Icon name={data.status === 'verified' ? 'shield' : 'alert'} size={22} />
        <div><CheckBadge status={data.status} /><div className="trust-headline">{data.headline}</div></div>
      </div>

      {lines.total > 0 && (
        <div className="stats trust-stats">
          <div className="card stat"><span className="stat-label">Lines from statements that add up</span>
            <strong>{share(lines.verified, lines.total)}</strong>
            <span className="muted small">{count(lines.verified)} of {count(lines.total)}</span></div>
          <div className="card stat"><span className="stat-label">Filed automatically</span>
            <strong>{share(lines.auto_filed, lines.total)}</strong>
            <span className="muted small">by your memory, bank rules or the ML model</span></div>
          <div className="card stat"><span className="stat-label">Set by you</span>
            <strong>{count(lines.founder_set)}</strong>
            <span className="muted small">{count(lines.changed)} changed from the app’s first choice · {count(lines.needs_review)} still to review</span></div>
        </div>
      )}

      <section className="card">
        <h2>Statements</h2>
        <p className="muted small">Each file’s fingerprint (SHA-256) changes if even one digit in the file changes. Each line’s balance
          must equal the previous balance plus money in, minus money out.</p>
        {data.imports.length === 0 ? <p className="muted">No statements uploaded yet. <Link to={`/startups/${id}/transactions`}>Upload one</Link>.</p> : (
          <ul className="checks">
            {data.imports.map(i => (
              <li key={i.id}>
                <div className="check-head">
                  <div><strong>{i.filename}</strong>
                    <div className="muted small">{i.account} · {day(i.period_start)} to {day(i.period_end)} · uploaded {new Date(i.uploaded_at).toLocaleDateString('en-IN')}
                      {i.uploaded_by ? ` by ${i.uploaded_by}` : ''} · {count(i.rows_added)} new lines</div></div>
                  <CheckBadge status={i.status} />
                </div>
                <div className="check-facts muted small">
                  {i.balance_checked > 0 && <span>{count(i.balance_checked - i.balance_breaks)} of {count(i.balance_checked)} balances add up</span>}
                  {i.opening_balance !== null && <span>Opening {money(i.opening_balance, c)}</span>}
                  {i.closing_balance !== null && <span>Closing {money(i.closing_balance, c)}</span>}
                  {i.file_sha256 && <span>Fingerprint <code className="hash" title={i.file_sha256}>{i.file_sha256}</code></span>}
                </div>
                {i.issues.length > 0 && <ul className="issues">{i.issues.map(t => <li key={t}>{t}</li>)}</ul>}
              </li>
            ))}
          </ul>
        )}
      </section>

      {data.continuity.length > 0 && (
        <section className="card">
          <h2><Icon name="alert" /> Possible missing statements</h2>
          <p className="muted small">Each statement should start at the balance the one before it ended with.</p>
          <ul className="issues">{data.continuity.map(t => <li key={t}>{t}</li>)}</ul>
        </section>
      )}

      {data.months.length > 0 && (
        <section className="card">
          <h2>Month by month</h2>
          <div className="table-scroll">
            <table className="pnl">
              <thead><tr><th>Month</th><th>Lines</th><th>From statements that add up</th><th>Filed automatically</th><th>Set by you</th><th>To review</th></tr></thead>
              <tbody>
                {data.months.map(m => (
                  <tr key={m.month}>
                    <td><Link to={`/startups/${id}/transactions?month=${m.month}`}>{monthLabel(m.month, true)}</Link></td>
                    <td>{count(m.total)}</td><td>{share(m.verified, m.total)}</td><td>{share(m.auto_filed, m.total)}</td>
                    <td>{count(m.founder_set)}</td><td className={m.needs_review ? 'warn' : ''}>{count(m.needs_review)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card">
        <h2>Activity history <CheckBadge status={audit.ok ? 'verified' : 'warnings'} /></h2>
        <p className="muted small">Every upload and change is logged. Each entry carries the fingerprint of the one before it, so
          changing or deleting an entry breaks the chain.</p>
        {audit.problem && <div className="banner"><Icon name="alert" /> <span className="grow">{audit.problem}</span></div>}
        <p>{count(audit.sealed)} entries in the chain{audit.unsealed ? ` (${count(audit.unsealed)} older entries were written before the chain existed)` : ''}.</p>
        {audit.head && <p className="small">Latest fingerprint: <code className="hash">{audit.head}</code><br />
          <span className="muted">Keep this with the printed report. If history is ever rewritten, this fingerprint will no longer match.</span></p>}
      </section>

      <p className="muted small trust-limits">{data.limits}</p>
    </div>
  )
}
