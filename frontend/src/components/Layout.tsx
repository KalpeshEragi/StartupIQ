import type { ReactNode } from 'react'
import { Link, Navigate, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

function Splash() {
  return <div className="splash" aria-busy="true"><div className="logo">IQ</div></div>
}

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth()
  const location = useLocation()
  if (!ready) return <Splash />
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />
  return children
}

export function GuestOnly() {
  const { user, ready } = useAuth()
  if (!ready) return <Splash />
  if (user) return <Navigate to="/" replace />
  return (
    <div className="auth-page">
      <div className="auth-brand"><div className="logo">IQ</div><span>Startup IQ</span></div>
      <Outlet />
      <p className="fine">Your numbers, your cap table and your investors in one place.</p>
    </div>
  )
}

export function AppShell() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand"><div className="logo">IQ</div><span>Startup IQ</span></Link>
        <div className="topbar-right">
          <span className="muted">{user?.full_name}</span>
          <button className="btn ghost" onClick={async () => { await logout(); navigate('/login') }}>Sign out</button>
        </div>
      </header>
      <main className="content"><Outlet /></main>
    </div>
  )
}
