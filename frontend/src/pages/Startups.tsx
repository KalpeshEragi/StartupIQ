import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { ErrorBox } from '../components/Form'
import { Icon } from '../components/Icons'
import { api, errorText, PRODUCT_TYPES, STAGES, type Startup } from '../lib/api'

export function Startups() {
  const [startups, setStartups] = useState<Startup[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { api.startups().then(setStartups).catch(e => setError(errorText(e))) }, [])

  if (error) return <ErrorBox error={error} />
  if (!startups) return <p className="muted" role="status" aria-busy="true">Loading…</p>
  if (startups.length === 0) return <Navigate to="/startups/new" replace />
  if (startups.length === 1) return <Navigate to={`/startups/${startups[0].id}`} replace />

  return (
    <div>
      <div className="page-head">
        <div><h1>Your startups</h1><p className="muted">Pick a company to open its workspace.</p></div>
        <Link to="/startups/new" className="btn primary"><Icon name="plus" size={16} /> New startup</Link>
      </div>
      <div className="grid">
        {startups.map(s => (
          <Link key={s.id} to={`/startups/${s.id}`} className="card tile">
            <span className="ws-avatar">{s.name.split(/\s+/).slice(0, 2).map(w => w[0]?.toUpperCase()).join('')}</span>
            <strong>{s.name}</strong>
            <span className="muted">{s.sector}</span>
            <span className="pills"><span className="pill">{STAGES[s.stage]}</span><span className="pill">{PRODUCT_TYPES[s.product_type]}</span></span>
          </Link>
        ))}
        <Link to="/startups/new" className="card tile add"><Icon name="plus" size={22} /> Add another startup</Link>
      </div>
    </div>
  )
}
