import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ErrorBox, Field } from '../components/Form'
import { StartupNav } from '../components/StartupNav'
import { api, errorText, HOLDER_KINDS, type CapTable as Table, type HolderKind, type Simulation, type Startup } from '../lib/api'
import { compact, count, pct, today } from '../lib/format'

const price = (n: number | null | undefined, c: string) => (n == null ? '—' : `${c === 'INR' ? '₹' : c + ' '}${n.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
const num = (f: FormData, k: string) => (f.get(k) === '' || f.get(k) === null ? null : Number(f.get(k)))

function Setup({ id, onDone }: { id: string; onDone: () => void }) {
  const [founders, setFounders] = useState([{ name: '', shares: 5000000 }, { name: '', shares: 5000000 }])
  const [error, setError] = useState<string | null>(null)
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    setError(null)
    try {
      await api.setupCapTable(id, { incorporated_on: String(f.get('incorporated_on')), pool_shares: num(f, 'pool') ?? 0,
        founders: founders.filter(x => x.name.trim()).map(x => ({ name: x.name.trim(), shares: x.shares })) })
      onDone()
    } catch (err) { setError(errorText(err)) }
  }
  return (
    <form className="card inline-form" onSubmit={submit}>
      <h2>Set up your cap table</h2>
      <p className="muted">How the shares were split when the company was formed. You can add investors, SAFEs and option grants afterwards.</p>
      <ErrorBox error={error} />
      <Field label="Incorporated on"><input name="incorporated_on" type="date" max={today()} required /></Field>
      {founders.map((fd, i) => (
        <div className="row-3" key={i}>
          <Field label={`Founder ${i + 1} name`}><input value={fd.name} required={i === 0}
            onChange={e => setFounders(fs => fs.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} /></Field>
          <Field label="Shares"><input type="number" min={1} value={fd.shares}
            onChange={e => setFounders(fs => fs.map((x, j) => (j === i ? { ...x, shares: Number(e.target.value) } : x)))} /></Field>
          <div className="field" style={{ justifyContent: 'flex-end' }}>
            {founders.length > 1 && <button type="button" className="btn ghost" onClick={() => setFounders(fs => fs.filter((_, j) => j !== i))}>Remove</button>}
          </div>
        </div>
      ))}
      <div><button type="button" className="btn small" onClick={() => setFounders(fs => [...fs, { name: '', shares: 1000000 }])}>Add a founder</button></div>
      <Field label="ESOP pool (shares reserved for employees, optional)" hint="Often 10–15% of the company.">
        <input name="pool" type="number" min={0} defaultValue={1000000} />
      </Field>
      <div><button className="btn primary">Create cap table</button></div>
    </form>
  )
}

function Simulator({ id, startup, onRecorded }: { id: string; startup: Startup; onRecorded: () => void }) {
  const [input, setInput] = useState({ new_money: 20000000, pre_money: 80000000, pool_target: 0.1, investor_name: 'New investor' })
  const [sim, setSim] = useState<Simulation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [record, setRecord] = useState({ name: 'Seed', closed_on: today() })
  const c = startup.currency

  async function run(e?: FormEvent) {
    e?.preventDefault()
    setError(null)
    try { setSim(await api.simulate(id, input)) } catch (err) { setError(errorText(err)) }
  }
  async function save() {
    if (!confirm(`Record the ${record.name} round in your cap table?`)) return
    try { await api.recordRound(id, { ...input, ...record }); setSim(null); onRecorded() } catch (err) { setError(errorText(err)) }
  }
  const set = (k: keyof typeof input) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setInput(v => ({ ...v, [k]: k === 'investor_name' ? e.target.value : k === 'pool_target' ? Number(e.target.value) / 100 : Number(e.target.value) }))

  return (
    <section className="card">
      <h2>Dilution simulator</h2>
      <p className="muted">What everyone owns after a priced round. Pool top-up and SAFE conversions come out of the pre-money (the usual Indian term-sheet convention).</p>
      <ErrorBox error={error} />
      <form className="unit-form" onSubmit={run}>
        <Field label="Raising"><input type="number" min={1} value={input.new_money} onChange={set('new_money')} /></Field>
        <Field label="Pre-money valuation"><input type="number" min={1} value={input.pre_money} onChange={set('pre_money')} /></Field>
        <Field label="ESOP pool after the round (%)"><input type="number" min={0} max={50} step={0.5} value={input.pool_target * 100} onChange={set('pool_target')} /></Field>
        <Field label="Lead investor"><input value={input.investor_name} onChange={set('investor_name')} /></Field>
        <div className="form-actions"><button className="btn primary">Simulate</button></div>
      </form>
      {sim && (
        <>
          <div className="metric-row" style={{ margin: '12px 0' }}>
            <div><span className="muted">Post-money</span><strong>{compact(sim.post_money, c)}</strong></div>
            <div><span className="muted">Price per share</span><strong>{price(sim.price_per_share, c)}</strong></div>
            <div><span className="muted">New shares</span><strong>{count(sim.new_shares)}</strong></div>
            <div><span className="muted">Pool top-up</span><strong>{count(sim.pool_top_up)}</strong></div>
          </div>
          <div className="table-scroll">
            <table className="pnl">
              <thead><tr><th>Holder</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
              <tbody>{sim.table.map(r => (
                <tr key={r.name}><td>{r.name} <span className="muted small">{HOLDER_KINDS[r.kind as HolderKind] ?? r.kind}</span></td>
                  <td>{pct(r.ownership_before, 1)}</td><td><strong>{pct(r.ownership_after, 1)}</strong></td>
                  <td className={r.ownership_after < r.ownership_before ? 'bad' : 'good'}>{r.ownership_before ? `${((r.ownership_after - r.ownership_before) * 100).toFixed(1)} pts` : 'new'}</td></tr>
              ))}</tbody>
            </table>
          </div>
          {sim.safe_conversions.length > 0 && <p className="muted small">SAFEs converting: {sim.safe_conversions.map(s => `${s.holder} ${compact(s.amount, c)} → ${count(s.shares)} shares`).join('; ')}.</p>}
          <div className="inline-fields">
            <Field label="Round name"><input value={record.name} onChange={e => setRecord(r => ({ ...r, name: e.target.value }))} /></Field>
            <Field label="Closed on"><input type="date" value={record.closed_on} onChange={e => setRecord(r => ({ ...r, closed_on: e.target.value }))} /></Field>
            <button className="btn" onClick={save}>Record this round</button>
          </div>
        </>
      )}
    </section>
  )
}

export function CapTable() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [table, setTable] = useState<Table | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = useCallback(() => {
    Promise.all([api.startup(id), api.capTable(id)]).then(([s, t]) => { setStartup(s); setTable(t) }).catch(e => setError(errorText(e)))
  }, [id])
  useEffect(load, [load])

  async function act(fn: () => Promise<unknown>) {
    setError(null)
    try { await fn(); load() } catch (err) { setError(errorText(err)) }
  }

  if (!startup || !table) return error ? <div className="narrow"><ErrorBox error={error} /><Link to="/">Back</Link></div> : <p className="muted">Loading…</p>
  const c = startup.currency
  const empty = table.total_fully_diluted === 0

  return (
    <div>
      <StartupNav id={id} name={startup.name} />
      <div className="page-head"><div><h1>Cap table</h1><p className="muted">Who owns the company, fully diluted.</p></div></div>
      <ErrorBox error={error} />
      {empty ? <Setup id={id} onDone={load} /> : (
        <>
          <section className="card">
            <div className="ownership-bar">
              {table.holders.filter(h => h.fully_diluted > 0).map(h => (
                <span key={h.id} className={`own-${h.kind}`} style={{ width: `${h.ownership * 100}%` }} title={`${h.name} ${pct(h.ownership, 1)}`} />
              ))}
            </div>
            <div className="table-scroll">
              <table className="pnl">
                <thead><tr><th>Holder</th><th>Shares</th><th>Options (vested)</th><th>Fully diluted</th><th>Ownership</th><th>Invested</th></tr></thead>
                <tbody>{table.holders.filter(h => h.fully_diluted > 0).map(h => (
                  <tr key={h.id}>
                    <td><span className={`dot own-${h.kind}`} /> {h.name} <span className="muted small">{HOLDER_KINDS[h.kind]}</span></td>
                    <td>{count(h.kind === 'esop_pool' ? h.fully_diluted : h.shares)}{h.kind === 'esop_pool' && <span className="muted small"> unallocated</span>}</td>
                    <td>{h.options ? `${count(h.options)} (${count(h.vested_options)})` : '—'}</td>
                    <td>{count(h.fully_diluted)}</td><td><strong>{pct(h.ownership, 1)}</strong></td>
                    <td>{h.invested ? compact(h.invested, c) : '—'}</td>
                  </tr>
                ))}</tbody>
                <tfoot><tr><td>Total</td><td /><td /><td>{count(table.total_fully_diluted)}</td><td>100%</td><td /></tr></tfoot>
              </table>
            </div>
          </section>

          {table.rounds.length > 0 && (
            <section className="card">
              <h2>Funding rounds</h2>
              <table className="pnl"><thead><tr><th>Round</th><th>Closed</th><th>Raised</th><th>Pre-money</th><th>Post-money</th><th>Price/share</th><th>Founders after</th><th /></tr></thead>
                <tbody>{table.rounds.map(r => (
                  <tr key={r.id}><td>{r.name}</td><td>{r.closed_on}</td><td>{compact(r.amount_raised, c)}</td><td>{compact(r.pre_money, c)}</td>
                    <td>{compact(r.post_money, c)}</td><td>{price(r.price_per_share, c)}</td><td>{pct(r.founders_after, 1)}</td>
                    <td><button className="btn ghost small" onClick={() => confirm(`Undo the ${r.name} round?`) && act(() => api.remove(id, 'rounds', r.id))}>Undo</button></td></tr>
                ))}</tbody></table>
            </section>
          )}

          <div className="two-col">
            <section className="card">
              <h2>SAFEs and convertible notes</h2>
              {table.safes.length ? (
                <ul className="activity">{table.safes.map(s => (
                  <li key={s.id}><span>{s.holder}: {compact(s.amount, c)}{s.valuation_cap ? ` · cap ${compact(s.valuation_cap, c)}` : ''}{s.discount ? ` · ${pct(s.discount)} discount` : ''}
                    {s.converted && <span className="pill">converted</span>}</span>
                    {!s.converted && <button className="btn ghost small" onClick={() => act(() => api.remove(id, 'safes', s.id))}>Remove</button>}</li>
                ))}</ul>) : <p className="muted">None yet.</p>}
              <form className="mini-form" onSubmit={e => { e.preventDefault(); const f = new FormData(e.currentTarget); const form = e.currentTarget
                act(() => api.addSafe(id, { investor_name: String(f.get('name')), amount: num(f, 'amount') ?? 0, valuation_cap: num(f, 'cap'),
                  discount: num(f, 'discount') === null ? null : (num(f, 'discount') ?? 0) / 100, signed_on: String(f.get('signed_on')) }).then(() => form.reset())) }}>
                <input name="name" placeholder="Investor" required /><input name="amount" type="number" min={1} placeholder="Amount" required />
                <input name="cap" type="number" min={1} placeholder="Valuation cap" /><input name="discount" type="number" min={0} max={90} placeholder="Discount %" />
                <input name="signed_on" type="date" max={today()} required /><button className="btn small">Add SAFE</button>
              </form>
            </section>
            <section className="card">
              <h2>Employee options (ESOP)</h2>
              {table.grants.length ? (
                <ul className="activity">{table.grants.map(g => (
                  <li key={g.id}><span>{g.holder}: {count(g.shares)} options · {count(g.vested)} vested <span className="muted small">({g.vesting_months / 12} yr, {g.cliff_months} mo cliff)</span></span>
                    <button className="btn ghost small" onClick={() => act(() => api.remove(id, 'option-grants', g.id))}>Remove</button></li>
                ))}</ul>) : <p className="muted">No grants yet.</p>}
              <form className="mini-form" onSubmit={e => { e.preventDefault(); const f = new FormData(e.currentTarget); const form = e.currentTarget
                act(() => api.addGrant(id, { employee_name: String(f.get('name')), shares: num(f, 'shares') ?? 0, granted_on: String(f.get('granted_on')),
                  vesting_months: num(f, 'vesting') ?? 48, cliff_months: num(f, 'cliff') ?? 12 }).then(() => form.reset())) }}>
                <input name="name" placeholder="Employee" required /><input name="shares" type="number" min={1} placeholder="Options" required />
                <input name="granted_on" type="date" max={today()} required />
                <input name="vesting" type="number" min={1} defaultValue={48} title="Vesting months" /><input name="cliff" type="number" min={0} defaultValue={12} title="Cliff months" />
                <button className="btn small">Grant options</button>
              </form>
            </section>
          </div>

          <Simulator id={id} startup={startup} onRecorded={load} />

          <details className="card">
            <summary>Add shares by hand (for example an angel who bought shares directly)</summary>
            <form className="mini-form" onSubmit={e => { e.preventDefault(); const f = new FormData(e.currentTarget); const form = e.currentTarget
              act(() => api.addShares(id, { name: String(f.get('name')), kind: f.get('kind') as HolderKind, shares: num(f, 'shares') ?? 0,
                price_per_share: num(f, 'price'), issued_on: String(f.get('issued_on')) }).then(() => form.reset())) }}>
              <input name="name" placeholder="Holder" required />
              <select name="kind" defaultValue="investor">{Object.entries(HOLDER_KINDS).map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
              <input name="shares" type="number" min={1} placeholder="Shares" required /><input name="price" type="number" min={0} step="any" placeholder="Price per share" />
              <input name="issued_on" type="date" max={today()} required /><button className="btn small">Add shares</button>
            </form>
          </details>
        </>
      )}
    </div>
  )
}
