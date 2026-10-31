// Typed API client. The access token lives only in memory; the refresh token is an httpOnly cookie the browser sends
// to /api/v1/auth. On a 401 the client refreshes once and retries, so a 15-minute token never logs the user out.

export type Stage = 'idea' | 'pre_seed' | 'seed' | 'series_a' | 'series_b_plus'
export type ProductType = 'physical' | 'digital'

export interface User { id: string; email: string; full_name: string; created_at: string }
export interface Session { access_token: string; expires_in: number; user: User }
export interface Startup {
  id: string; name: string; sector: string; stage: Stage; product_type: ProductType; founded_on: string | null
  currency: string; fy_start_month: number; created_at: string; role: 'founder'
}
export type StartupInput = Omit<Startup, 'id' | 'created_at' | 'role'>
export interface Activity { created_at: string; action: string; user_id: string | null; details: Record<string, unknown> }

export const STAGES: Record<Stage, string> = {
  idea: 'Idea', pre_seed: 'Pre-seed', seed: 'Seed', series_a: 'Series A', series_b_plus: 'Series B+',
}
export const PRODUCT_TYPES: Record<ProductType, string> = { physical: 'Physical product', digital: 'Digital product' }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

export function errorText(e: unknown): string {
  return e instanceof Error ? e.message : 'Something went wrong. Please try again.'
}

let accessToken: string | null = null
let onSessionLost: () => void = () => {}
export const setAccessToken = (t: string | null) => { accessToken = t }
export const onLogout = (fn: () => void) => { onSessionLost = fn }

function message(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown })?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail[0]?.msg) {
    const d = detail[0] as { loc?: string[]; msg: string }
    const field = d.loc?.at(-1)
    return field ? `${field.replaceAll('_', ' ')}: ${d.msg.replace(/^Value error, /, '')}` : d.msg
  }
  return status >= 500 ? 'Something went wrong on our side. Please try again.' : `Request failed (${status})`
}

async function raw<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body) headers.set('Content-Type', 'application/json')
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  const res = await fetch(`/api/v1${path}`, { ...init, headers, credentials: 'same-origin' })
  if (res.status === 204) return undefined as T
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new ApiError(res.status, message(body, res.status))
  return body as T
}

let refreshing: Promise<Session | null> | null = null
export function refreshSession(): Promise<Session | null> {
  refreshing ??= raw<Session>('/auth/refresh', { method: 'POST' })
    .then(s => { setAccessToken(s.access_token); return s })
    .catch(() => { setAccessToken(null); return null })
    .finally(() => { refreshing = null })
  return refreshing
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  try {
    return await raw<T>(path, init)
  } catch (e) {
    if (!(e instanceof ApiError) || e.status !== 401 || path.startsWith('/auth/')) throw e
    if (!(await refreshSession())) { onSessionLost(); throw e }
    return raw<T>(path, init)
  }
}

const json = (method: string, data: unknown): RequestInit => ({ method, body: JSON.stringify(data) })

export const api = {
  register: (email: string, full_name: string, password: string) =>
    raw<Session>('/auth/register', json('POST', { email, full_name, password })),
  login: (email: string, password: string) => raw<Session>('/auth/login', json('POST', { email, password })),
  logout: () => raw<void>('/auth/logout', { method: 'POST' }),
  startups: () => call<Startup[]>('/startups'),
  startup: (id: string) => call<Startup>(`/startups/${id}`),
  createStartup: (data: StartupInput) => call<Startup>('/startups', json('POST', data)),
  updateStartup: (id: string, data: Partial<StartupInput>) => call<Startup>(`/startups/${id}`, json('PATCH', data)),
  activity: (id: string) => call<Activity[]>(`/startups/${id}/activity`),
}
