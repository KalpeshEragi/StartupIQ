import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import { AppShell, GuestOnly, RequireAuth } from './components/Layout'
import { Login, Register } from './pages/Auth'
import { NewStartup } from './pages/NewStartup'
import { StartupOverview } from './pages/StartupOverview'
import { Startups } from './pages/Startups'

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<GuestOnly />}>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Route>
          <Route element={<RequireAuth><AppShell /></RequireAuth>}>
            <Route path="/" element={<Startups />} />
            <Route path="/startups/new" element={<NewStartup />} />
            <Route path="/startups/:id" element={<StartupOverview />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
