import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, onLogout, refreshSession, setAccessToken, type Session, type User } from '../lib/api'

interface AuthState {
  user: User | null
  ready: boolean // false until the first silent refresh has finished
  login: (email: string, password: string) => Promise<void>
  register: (email: string, fullName: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [ready, setReady] = useState(false)

  const start = useCallback((s: Session) => { setAccessToken(s.access_token); setUser(s.user) }, [])

  useEffect(() => {
    onLogout(() => { setAccessToken(null); setUser(null) })
    refreshSession().then(s => { if (s) setUser(s.user) }).finally(() => setReady(true))
  }, [])

  const value = useMemo<AuthState>(() => ({
    user, ready,
    login: async (email, password) => start(await api.login(email, password)),
    register: async (email, fullName, password) => start(await api.register(email, fullName, password)),
    logout: async () => { await api.logout().catch(() => {}); setAccessToken(null); setUser(null) },
  }), [user, ready, start])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
