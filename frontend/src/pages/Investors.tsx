import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ErrorBox } from '../components/Form'
import { StartupNav } from '../components/StartupNav'
import { api, DEAL_STAGES, errorText, type CapTable, type Deal, type DealStage, type Startup } from '../lib/api'
import { compact, pct } from '../lib/format'

const ACTIVE: DealStage[] = ['contacted', 'meeting', 'due_diligence', 'term_sheet']

export function Investors() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [deals, setDeals] = useState<Deal[]>([])
  const [table, setTable] = useState<CapTable | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = useCallback(() => {
    Promise.all([api.startup(id), api.deals(id), api.capTable(id)])
      .then(([s, d, t]) => { setStartup(s); setDeals(d); setTable(t) }).catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  async function act(fn: () => Promise<unknown>) {
    setError(null)
    try { await fn(); load() } catch (err) { setError(errorText(err)) }
  }
  function add(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    const form = e.currentTarget
    act(() => api.addDeal(id, { investor_name: String(f.get('name')), firm: String(f.get('firm')) || null,
      amount: f.get('amount') ? Number(f.get('amount')) : null, next_step: String(f.get('next')) || null }).then(() => form.reset()))
  }

  if (!startup) return error ? <div className="narrow"><ErrorBox error={error} /><Link to="/">Back</Link></div> : <p className="muted">Loading…</p>
  const c = startup.currency
  const investors = table?.holders.filter(h => h.kind === 'investor') ?? []
  const open = deals.filter(d => ACTIVE.includes(d.stage))
  const pipelineTotal = open.reduce((s, d) => s + (d.amount ?? 0), 0)

  return (
    <div>
      <StartupNav id={id} name={startup.name} />
      <div className="page-head"><div><h1>Investors</h1><p className="muted">Who has invested, and who you are talking to about the next round.</p></div></div>
      <ErrorBox error={error} />

      <section className="card">
        <h2>Current investors</h2>
        {investors.length ? (
          <table className="pnl"><thead><tr><th>Investor</th><th>Invested</th><th>Ownership</th></tr></thead>
            <tbody>{investors.map(h => <tr key={h.id}><td>{h.name}</td><td>{h.invested ? compact(h.invested, c) : '—'}</td><td>{pct(h.ownership, 1)}</td></tr>)}</tbody></table>
        ) : <p className="muted">No investors on the cap table yet. Record a round or a SAFE in <Link to={`/startups/${id}/cap-table`}>Cap table</Link>.</p>}
      </section>

      <section className="card">
        <div className="page-head" style={{ marginBottom: 8 }}>
          <h2>Fundraising pipeline</h2>
          <span className="muted">{open.length} open · {compact(pipelineTotal, c)} in play</span>
        </div>
        <form className="mini-form" onSubmit={add}>
          <input name="name" placeholder="Investor name" required /><input name="firm" placeholder="Firm (optional)" />
          <input name="amount" type="number" min={0} placeholder="Possible cheque" /><input name="next" placeholder="Next step" />
          <button className="btn small primary">Add to pipeline</button>
        </form>
        <div className="kanban">
          {(Object.keys(DEAL_STAGES) as DealStage[]).map(stage => (
            <div key={stage} className="kanban-col">
              <h3>{DEAL_STAGES[stage]} <span className="muted">{deals.filter(d => d.stage === stage).length}</span></h3>
              {deals.filter(d => d.stage === stage).map(d => (
                <div key={d.id} className="deal">
                  <strong>{d.investor_name}</strong>
                  {d.firm && <span className="muted small">{d.firm}</span>}
                  {d.amount != null && <span className="small">{compact(d.amount, c)}</span>}
                  {d.next_step && <span className="muted small">Next: {d.next_step}</span>}
                  <div className="deal-actions">
                    <select value={d.stage} aria-label="Stage" onChange={e => act(() => api.updateDeal(id, d.id, { stage: e.target.value as DealStage }))}>
                      {Object.entries(DEAL_STAGES).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
                    </select>
                    <button className="btn ghost small" aria-label="Remove" onClick={() => act(() => api.remove(id, 'deals', d.id))}>✕</button>
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
