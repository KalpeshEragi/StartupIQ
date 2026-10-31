import { useState, type FormEvent } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { ErrorBox, Field } from '../components/Form'
import { errorText } from '../lib/api'

export function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const from = (useLocation().state as { from?: string } | null)?.from ?? '/'
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    setBusy(true); setError(null)
    try {
      await login(String(f.get('email')), String(f.get('password')))
      navigate(from, { replace: true })
    } catch (err) { setError(errorText(err)) } finally { setBusy(false) }
  }

  return (
    <form className="card auth-card" onSubmit={submit}>
      <h1>Sign in</h1>
      <ErrorBox error={error} />
      <Field label="Email"><input name="email" type="email" autoComplete="email" required autoFocus /></Field>
      <Field label="Password"><input name="password" type="password" autoComplete="current-password" required /></Field>
      <button className="btn primary" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'}</button>
      <p className="muted center">New here? <Link to="/register">Create an account</Link></p>
    </form>
  )
}

export function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    setBusy(true); setError(null)
    try {
      await register(String(f.get('email')), String(f.get('full_name')), String(f.get('password')))
      navigate('/startups/new', { replace: true })
    } catch (err) { setError(errorText(err)) } finally { setBusy(false) }
  }

  return (
    <form className="card auth-card" onSubmit={submit}>
      <h1>Create your account</h1>
      <ErrorBox error={error} />
      <Field label="Full name"><input name="full_name" autoComplete="name" required maxLength={120} autoFocus /></Field>
      <Field label="Work email"><input name="email" type="email" autoComplete="email" required /></Field>
      <Field label="Password" hint="At least 10 characters.">
        <input name="password" type="password" autoComplete="new-password" required minLength={10} maxLength={128} />
      </Field>
      <button className="btn primary" disabled={busy}>{busy ? 'Creating…' : 'Create account'}</button>
      <p className="muted center">Already have an account? <Link to="/login">Sign in</Link></p>
    </form>
  )
}
