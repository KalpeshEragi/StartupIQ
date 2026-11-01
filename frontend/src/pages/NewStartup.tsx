import { useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { StartupForm } from './StartupForm'

export function NewStartup() {
  const navigate = useNavigate()
  return (
    <div className="narrow">
      <div><h1>Set up your startup</h1>
      <p className="muted" style={{ margin: 0 }}>A few basics so the numbers make sense. You can change any of this later.</p></div>
      <div className="card">
        <StartupForm submitLabel="Create startup" onSubmit={async data => {
          const s = await api.createStartup(data)
          navigate(`/startups/${s.id}`, { replace: true })
        }} />
      </div>
    </div>
  )
}
