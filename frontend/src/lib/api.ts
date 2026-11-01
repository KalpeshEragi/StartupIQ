// Typed API client. The access token lives only in memory; the refresh token is an httpOnly cookie the browser sends
// to /api/v1/auth. On a 401 the client refreshes once and retries, so a 15-minute token never logs the user out.

export type Stage = 'idea' | 'pre_seed' | 'seed' | 'series_a' | 'series_b_plus'
export type ProductType = 'physical' | 'digital'

export interface User { id: string; email: string; full_name: string; created_at: string }
export interface Session { access_token: string; expires_in: number; user: User }
export interface Startup {
  id: string; name: string; sector: string; stage: Stage; product_type: ProductType; founded_on: string | null
  currency: string; fy_start_month: number; created_at: string; role: 'founder'; market_sector?: string | null
}
export type StartupInput = Omit<Startup, 'id' | 'created_at' | 'role'>
export interface ForecastWeek { week: string; p10: number; p50: number; p90: number }
export interface MarketOutlook {
  key: string; label: string; history: { week: string; index: number }[]; forecast: ForecastWeek[]
  model: 'network' | 'smoothing'; model_label: string; backtest: { network_mase: number; smoothing_mase: number }
  change_4w: number; change_12w: number; change_12w_low: number; change_12w_high: number; peak_week: string
  monthly: Record<string, number>
}
export interface Market {
  sector: string | null; chosen: string | null; suggested: string | null; available: boolean; message?: string
  generated_at?: string; data_through?: string; about?: string; outlook?: MarketOutlook
  you_vs_market?: { months: { month: string; revenue: number; market: number }[]; your_change: number | null
    market_change: number | null; correlation: number | null; summary: string | null }
}
export interface Activity { created_at: string; action: string; user_id: string | null; details: Record<string, unknown> }

export interface Category { key: string; label: string; group: 'expense' | 'structural' }
export interface BankAccount { id: string; name: string; bank_name: string | null; last4: string | null; currency: string }
export type Source = 'founder' | 'memory' | 'rule' | 'model' | 'none'
export interface Transaction {
  id: string; bank_account_id: string; txn_date: string; description: string; direction: 'debit' | 'credit'
  amount: number; balance: number | null; category: string | null; category_source: Source; confidence: number | null
  needs_review: boolean; reason: string | null; suggestions: string[]; source_row: number | null
}
export interface TransactionPage { items: Transaction[]; total: number }
export interface ImportResult {
  id: string; filename: string; rows_read: number; rows_added: number; rows_duplicate: number; rows_skipped: number
  period_start: string | null; period_end: string | null; columns: Record<string, string>; notes: string[]
  auto_filed: number; needs_review: number; by_source: Record<string, number>
  file_sha256: string; balance_status: CheckStatus; balance_checked: number; balance_breaks: number; issues: string[]
  same_file_as: string | null
}
export type CheckStatus = 'verified' | 'warnings' | 'unverified'
export interface TrustCounts { total: number; verified: number; auto_filed: number; founder_set: number; changed: number; needs_review: number }
export interface ImportCheck {
  id: string; filename: string; account: string; uploaded_at: string; uploaded_by: string | null; period_start: string | null
  period_end: string | null; rows_added: number; file_sha256: string | null; status: CheckStatus; balance_checked: number
  balance_breaks: number; opening_balance: number | null; closing_balance: number | null; issues: string[]
}
export interface Verification {
  generated_at: string; status: CheckStatus; headline: string; lines: TrustCounts; months: (TrustCounts & { month: string })[]
  imports: ImportCheck[]; continuity: string[]; limits: string
  audit: { ok: boolean; sealed: number; unsealed: number; head: string | null; problem: string | null }
}
export interface TransactionHistory {
  transaction: Transaction; file: string | null; file_sha256: string | null; statement_status: CheckStatus | null
  source_row: number | null; first_category: string | null; first_source: string | null; first_reason: string | null
  events: { at: string; what: string; by: string | null }[]
}
export interface ReviewSummary {
  total: number; needs_review: number; auto_filed_share: number; by_source: Record<string, number>; model_version: string | null
}
export interface TransactionFilter { needs_review?: boolean; category?: string; month?: string; q?: string; limit?: number; offset?: number }

export interface PnlLine { key: string; label: string; amount: number }
export interface PnlMonth {
  month: string; revenue: number; other_income: number; cogs: number; gross_profit: number; gross_margin: number | null
  opex: PnlLine[]; opex_total: number; expenses: number; ebitda: number; cash_in: number; cash_out: number
  net_cash_flow: number; burn: number; financing_in: number; taxes_paid: number; loan_repaid: number
  closing_cash: number | null; lines: number; needs_review: number; uncategorised: number
}
export interface Runway { cash: number | null; avg_burn: number | null; months: number | null; status: 'burning' | 'not_burning' | 'no_balance' | 'no_data' }
export interface Pnl { currency: string; months: PnlMonth[]; runway: Runway }
export type Units = Record<string, number | null>
export interface Dashboard {
  currency: string; has_data: boolean; month?: string
  kpis?: { revenue: number; revenue_growth: number | null; gross_margin: number | null; ebitda: number; burn: number }
  runway?: Runway; needs_review?: number; uncategorised?: number
  series?: { month: string; revenue: number; expenses: number; ebitda: number; closing_cash: number | null }[]
  top_expenses?: PnlLine[]; units?: Units; units_entered?: boolean
}
export interface UnitMetricRow extends Record<string, unknown> { month: string; computed?: Units }
export type HolderKind = 'founder' | 'investor' | 'employee' | 'advisor' | 'esop_pool' | 'other'
export interface Holder {
  id: string; name: string; kind: HolderKind; email: string | null; shares: number; options: number; vested_options: number
  fully_diluted: number; ownership: number; invested: number
}
export interface CapTable {
  total_fully_diluted: number; holders: Holder[]
  rounds: { id: string; name: string; closed_on: string; pre_money: number; amount_raised: number; post_money: number; price_per_share: number; founders_after: number }[]
  safes: { id: string; holder: string; amount: number; valuation_cap: number | null; discount: number | null; signed_on: string; converted: boolean }[]
  grants: { id: string; holder: string; shares: number; granted_on: string; vesting_months: number; cliff_months: number; vested: number }[]
}
export interface RoundInput { new_money: number; pre_money: number; pool_target: number; investor_name: string }
export interface Simulation {
  price_per_share: number; post_money: number; new_shares: number; pool_top_up: number
  safe_conversions: { holder: string; amount: number; shares: number }[]
  table: { name: string; kind: string; before: number; after: number; ownership_before: number; ownership_after: number }[]
}
export type DealStage = 'contacted' | 'meeting' | 'due_diligence' | 'term_sheet' | 'closed' | 'passed'
export interface Deal {
  id: string; investor_name: string; firm: string | null; stage: DealStage; amount: number | null
  next_step: string | null; next_date: string | null; notes: string | null
}
export const DEAL_STAGES: Record<DealStage, string> = {
  contacted: 'Contacted', meeting: 'Meeting', due_diligence: 'Due diligence', term_sheet: 'Term sheet', closed: 'Closed', passed: 'Passed',
}
export const HOLDER_KINDS: Record<HolderKind, string> = {
  founder: 'Founder', investor: 'Investor', employee: 'Employee', advisor: 'Advisor', esop_pool: 'ESOP pool', other: 'Other',
}

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
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')  // the browser sets multipart
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
  market: (id: string) => call<Market>(`/startups/${id}/market`),
  marketSectors: () => call<{ key: string; label: string }[]>('/market/sectors'),
  setMarketSector: (id: string, market_sector: string) => call<Startup>(`/startups/${id}`, json('PATCH', { market_sector })),
  register: (email: string, full_name: string, password: string) =>
    raw<Session>('/auth/register', json('POST', { email, full_name, password })),
  login: (email: string, password: string) => raw<Session>('/auth/login', json('POST', { email, password })),
  logout: () => raw<void>('/auth/logout', { method: 'POST' }),
  startups: () => call<Startup[]>('/startups'),
  startup: (id: string) => call<Startup>(`/startups/${id}`),
  createStartup: (data: StartupInput) => call<Startup>('/startups', json('POST', data)),
  updateStartup: (id: string, data: Partial<StartupInput>) => call<Startup>(`/startups/${id}`, json('PATCH', data)),
  activity: (id: string) => call<Activity[]>(`/startups/${id}/activity`),
  categories: (id: string) => call<Category[]>(`/startups/${id}/categories`),
  bankAccounts: (id: string) => call<BankAccount[]>(`/startups/${id}/bank-accounts`),
  createBankAccount: (id: string, data: { name: string; bank_name?: string | null; last4?: string | null }) =>
    call<BankAccount>(`/startups/${id}/bank-accounts`, json('POST', data)),
  importStatement: (id: string, accountId: string, file: File) => {
    const body = new FormData()
    body.append('file', file)
    return call<ImportResult>(`/startups/${id}/bank-accounts/${accountId}/imports`, { method: 'POST', body })
  },
  transactions: (id: string, f: TransactionFilter = {}) => {
    const qs = new URLSearchParams(Object.entries(f).filter(([, v]) => v !== undefined && v !== '').map(([k, v]) => [k, String(v)]))
    return call<TransactionPage>(`/startups/${id}/transactions?${qs}`)
  },
  transactionHistory: (id: string, txnId: string) => call<TransactionHistory>(`/startups/${id}/transactions/${txnId}/history`),
  verification: (id: string) => call<Verification>(`/startups/${id}/verification`),
  reviewSummary: (id: string) => call<ReviewSummary>(`/startups/${id}/transactions/summary`),
  categorise: (id: string, txnId: string, category: string, remember: boolean) =>
    call<{ transaction: Transaction; also_updated: number }>(`/startups/${id}/transactions/${txnId}`, json('PATCH', { category, remember })),
  pnl: (id: string) => call<Pnl>(`/startups/${id}/financials/pnl`),
  dashboard: (id: string) => call<Dashboard>(`/startups/${id}/financials/dashboard`),
  unitMetrics: (id: string) => call<UnitMetricRow[]>(`/startups/${id}/unit-metrics`),
  saveUnitMetrics: (id: string, month: string, data: Record<string, number | null>) =>
    call<unknown>(`/startups/${id}/unit-metrics/${month}`, json('PUT', data)),
  setOpeningBalance: (id: string, accountId: string, opening_balance: number | null) =>
    call<unknown>(`/startups/${id}/bank-accounts/${accountId}`, json('PATCH', { opening_balance })),
  capTable: (id: string) => call<CapTable>(`/startups/${id}/cap-table`),
  setupCapTable: (id: string, data: { incorporated_on: string; founders: { name: string; shares: number }[]; pool_shares: number }) =>
    call<CapTable>(`/startups/${id}/cap-table/setup`, json('POST', data)),
  addSafe: (id: string, data: { investor_name: string; amount: number; valuation_cap: number | null; discount: number | null; signed_on: string }) =>
    call<unknown>(`/startups/${id}/safes`, json('POST', data)),
  addGrant: (id: string, data: { employee_name: string; shares: number; granted_on: string; vesting_months: number; cliff_months: number }) =>
    call<unknown>(`/startups/${id}/option-grants`, json('POST', data)),
  addShares: (id: string, data: { name: string; kind: HolderKind; shares: number; price_per_share: number | null; issued_on: string }) =>
    call<unknown>(`/startups/${id}/share-issues`, json('POST', data)),
  remove: (id: string, kind: 'safes' | 'option-grants' | 'rounds' | 'shareholders' | 'deals', objId: string) =>
    call<void>(`/startups/${id}/${kind}/${objId}`, { method: 'DELETE' }),
  simulate: (id: string, data: RoundInput) => call<Simulation>(`/startups/${id}/cap-table/simulate`, json('POST', data)),
  recordRound: (id: string, data: RoundInput & { name: string; closed_on: string }) =>
    call<CapTable>(`/startups/${id}/rounds`, json('POST', data)),
  deals: (id: string) => call<Deal[]>(`/startups/${id}/deals`),
  addDeal: (id: string, data: Partial<Deal> & { investor_name: string }) => call<Deal>(`/startups/${id}/deals`, json('POST', data)),
  updateDeal: (id: string, dealId: string, data: Partial<Deal>) => call<Deal>(`/startups/${id}/deals/${dealId}`, json('PATCH', data)),
}
