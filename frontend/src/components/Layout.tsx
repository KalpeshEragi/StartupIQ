import { useEffect, useState, type ReactNode } from 'react'
import { Link, Navigate, NavLink, Outlet, useLocation, useMatch, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { api, type Startup } from '../lib/api'
import { Icon, type IconName } from './Icons'
import { ThemeToggle } from './ThemeToggle'

function Logo() {
  return <div className="logo"><Icon name="sparkle" size={16} /></div>
}

function Splash() {
  return <div className="splash" aria-busy="true"><Logo /></div>
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth()
  const location = useLocation()
  if (!ready) return <Splash />
  if (!user) return <Navigate to={location.pathname === '/' ? '/welcome' : '/login'} replace state={{ from: location.pathname }} />
  return children
}

export function GuestOnly() {
  const { user, ready } = useAuth()
  if (!ready) return <Splash />
  if (user) return <Navigate to="/" replace />
  return (
    <div className="auth-page">
      <section className="auth-side">
        <div className="brand"><Logo /><span>StartupIQ</span></div>
        <div className="auth-pitch">
          <h2>The monthly operating system for founders.</h2>
          <p>Upload a bank statement and get your P&amp;L, burn and runway in minutes. Cap table, investors and your
            market’s outlook in one place.</p>
          <ul>
            <li><Icon name="check" size={16} /> Every transaction sorted automatically</li>
            <li><Icon name="check" size={16} /> Revenue, EBITDA, burn and runway every month</li>
            <li><Icon name="check" size={16} /> Cap table and dilution simulator</li>
          </ul>
        </div>
        <p className="fine">Built for Indian startups. Your data stays in your account.</p>
      </section>
      <section className="auth-main"><ThemeToggle /><Outlet /></section>
    </div>
  )
}

const NAV: [string, string, IconName][] = [
  ['', 'Overview', 'overview'], ['transactions', 'Transactions', 'transactions'], ['financials', 'Financials', 'financials'],
  ['cap-table', 'Cap table', 'captable'], ['investors', 'Investors', 'investors'], ['market', 'Market outlook', 'market'],
  ['trust', 'Trust', 'shield'],
]
const TITLES: Record<string, string> = Object.fromEntries(NAV.map(([p, l]) => [p, l]))

function initials(name: string | undefined) {
  return (name ?? '?').split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]?.toUpperCase()).join('')
}

export function AppShell() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const match = useMatch('/startups/:id/*')
  const id = match?.params.id && match.params.id !== 'new' ? match.params.id : null
  const [startup, setStartup] = useState<Startup | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!id) return
    let live = true
    api.startup(id).then(s => { if (live) setStartup(s) }).catch(() => {})
    return () => { live = false }
  }, [id])
  const current = id && startup?.id === id ? startup : null

  const section = id ? (match?.params['*'] ?? '').split('/')[0] : null
  const title = section !== null ? TITLES[section] ?? 'Overview' : location.pathname === '/startups/new' ? 'New startup' : 'Startups'
  useEffect(() => { document.title = [title, current?.name, 'StartupIQ'].filter(Boolean).join(' · ') }, [title, current?.name])

  return (
    <div className={`shell ${open ? 'nav-open' : ''}`}>
      <a className="skip" href="#main">Skip to content</a>
      <aside className="sidebar" id="sidebar" onClick={e => { if ((e.target as HTMLElement).closest('a')) setOpen(false) }}>
        <Link to="/" className="brand"><Logo /><span>StartupIQ</span></Link>

        {id ? (
          <>
            <Link to="/" className="workspace" title="Switch startup">
              <span className="ws-avatar">{initials(current?.name)}</span>
              <span className="ws-text"><strong>{current?.name ?? '…'}</strong><span>{current?.sector ?? ''}</span></span>
              <Icon name="chevron" size={16} />
            </Link>
            <nav className="side-nav" aria-label="Startup sections">
              <span className="side-label">Workspace</span>
              {NAV.map(([path, label, icon]) => (
                <NavLink key={label} to={`/startups/${id}${path ? `/${path}` : ''}`} end={!path}
                  className={({ isActive }) => (isActive ? 'on' : '')}>
                  <Icon name={icon} /> {label}
                </NavLink>
              ))}
            </nav>
          </>
        ) : (
          <nav className="side-nav" aria-label="Main">
            <span className="side-label">Menu</span>
            <NavLink to="/" end className={({ isActive }) => (isActive ? 'on' : '')}><Icon name="grid" /> Your startups</NavLink>
            <NavLink to="/startups/new" className={({ isActive }) => (isActive ? 'on' : '')}><Icon name="plus" /> New startup</NavLink>
          </nav>
        )}

        <div className="side-user">
          <span className="avatar">{initials(user?.full_name)}</span>
          <span className="ws-text"><strong>{user?.full_name}</strong><span>{user?.email}</span></span>
          <button className="icon-btn" title="Sign out" aria-label="Sign out"
            onClick={async () => { await logout(); navigate('/login') }}><Icon name="logout" /></button>
        </div>
      </aside>
      <div className="scrim" onClick={() => setOpen(false)} />

      <div className="main">
        <header className="topbar">
          <button className="icon-btn menu-btn" aria-label="Open menu" aria-expanded={open} aria-controls="sidebar" onClick={() => setOpen(true)}><Icon name="menu" /></button>
          <div className="crumbs">
            {current ? <><Link to={`/startups/${current.id}`}>{current.name}</Link><span>/</span></> : null}
            <strong>{title}</strong>
          </div>
          <div className="topbar-right">
            <ThemeToggle />
            <span className="avatar sm" title={user?.full_name}>{initials(user?.full_name)}</span>
          </div>
        </header>
        <main className="content" id="main" tabIndex={-1}><Outlet /></main>
      </div>
    </div>
  )
}
