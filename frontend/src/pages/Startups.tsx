import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import { ErrorBox } from '../components/Form'
import { api, errorText, PRODUCT_TYPES, STAGES, type Startup } from '../lib/api'

export function Startups() {
  const [startups, setStartups] = useState<Startup[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { api.startups().then(setStartups).catch(e => setError(errorText(e))) }, [])

  if (error) return <ErrorBox error={error} />
  if (!startups) return <p className="muted">Loading…</p>
  if (startups.length === 0) return <Navigate to="/startups/new" replace />
  if (startups.length === 1) return <Navigate to={`/startups/${startups[0].id}`} replace />

  return (
    <div>
      <div className="page-head">
        <h1>Your startups</h1>
        <Link to="/startups/new" className="btn primary">New startup</Link>
      </div>
      <div className="grid">
        {startups.map(s => (
          <Link key={s.id} to={`/startups/${s.id}`} className="card tile">
            <strong>{s.name}</strong>
            <span className="muted">{s.sector}</span>
            <span className="pills"><span className="pill">{STAGES[s.stage]}</span><span className="pill">{PRODUCT_TYPES[s.product_type]}</span></span>
          </Link>
        ))}
      </div>
    </div>
  )
}
