import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { StartupForm } from './StartupForm'

export function NewStartup() {
  const navigate = useNavigate()
  return (
    <div className="narrow">
      <h1>Set up your startup</h1>
      <p className="muted">You can change any of this later.</p>
      <div className="card">
        <StartupForm submitLabel="Create startup" onSubmit={async data => {
          const s = await api.createStartup(data)
          navigate(`/startups/${s.id}`, { replace: true })
        }} />
      </div>
    </div>
  )
}
