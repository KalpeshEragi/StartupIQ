import { NavLink } from 'react-router-dom'

const TABS: [string, string][] = [['', 'Overview'], ['transactions', 'Transactions'], ['financials', 'Financials'],
  ['cap-table', 'Cap table'], ['investors', 'Investors']]

export function StartupNav({ id, name }: { id: string; name: string }) {
  return (
    <div className="startup-nav">
      <div className="startup-name">{name}</div>
      <nav className="tabs-nav" aria-label="Startup sections">
        {TABS.map(([path, label]) => (
          <NavLink key={label} to={`/startups/${id}${path ? `/${path}` : ''}`} end={!path}
            className={({ isActive }) => (isActive ? 'on' : '')}>{label}</NavLink>
        ))}
      </nav>
    </div>
  )
}
