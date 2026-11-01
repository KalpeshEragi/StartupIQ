import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ErrorBox, Field } from '../components/Form'
import { StartupNav } from '../components/StartupNav'
import {
  api, errorText, type BankAccount, type Category, type ImportResult, type ReviewSummary, type Source, type Startup,
  type Transaction,
} from '../lib/api'

const PAGE = 50
const SOURCE: Record<Source, string> = { founder: 'You', memory: 'Remembered', rule: 'Bank format', model: 'ML model', none: 'Needs review' }

function money(n: number, currency: string) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency, maximumFractionDigits: 2 }).format(n)
}
function day(iso: string) {
  return new Date(iso + 'T00:00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
}

function AddAccount({ startupId, onAdded }: { startupId: string; onAdded: (a: BankAccount) => void }) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    setBusy(true); setError(null)
    try {
      onAdded(await api.createBankAccount(startupId, {
        name: String(f.get('name')).trim(), bank_name: String(f.get('bank_name')).trim() || null,
        last4: String(f.get('last4')).trim() || null,
      }))
    } catch (err) { setError(errorText(err)) } finally { setBusy(false) }
  }
  return (
    <form className="card inline-form" onSubmit={submit}>
      <h2>Add your bank account</h2>
      <p className="muted">Then upload its statement. We only ask for the last four digits, never the full number.</p>
      <ErrorBox error={error} />
      <div className="row-3">
        <Field label="Account name"><input name="name" required maxLength={80} placeholder="Current account" /></Field>
        <Field label="Bank (optional)"><input name="bank_name" maxLength={80} placeholder="HDFC Bank" /></Field>
        <Field label="Last 4 digits (optional)"><input name="last4" pattern="\d{4}" maxLength={4} inputMode="numeric" /></Field>
      </div>
      <div><button className="btn primary" disabled={busy}>{busy ? 'Adding…' : 'Add account'}</button></div>
    </form>
  )
}

function CategoryPicker({ value, categories, onPick, disabled }: {
  value: string | null; categories: Category[]; onPick: (key: string) => void; disabled: boolean
}) {
  return (
    <select className="cat-select" value={value ?? ''} disabled={disabled} aria-label="Category"
      onChange={e => e.target.value && onPick(e.target.value)}>
      {!value && <option value="">Choose a category…</option>}
      <optgroup label="Money in and bank items">
        {categories.filter(c => c.group === 'structural').map(c => <option key={c.key} value={c.key}>{c.label}</option>)}
      </optgroup>
      <optgroup label="Expenses">
        {categories.filter(c => c.group === 'expense').map(c => <option key={c.key} value={c.key}>{c.label}</option>)}
      </optgroup>
    </select>
  )
}

export function Transactions() {
  const { id = '' } = useParams()
  const [startup, setStartup] = useState<Startup | null>(null)
  const [accounts, setAccounts] = useState<BankAccount[] | null>(null)
  const [accountId, setAccountId] = useState('')
  const [categories, setCategories] = useState<Category[]>([])
  const [summary, setSummary] = useState<ReviewSummary | null>(null)
  const [items, setItems] = useState<Transaction[]>([])
  const [total, setTotal] = useState(0)
  const [tab, setTab] = useState<'review' | 'all'>('review')
  const [month, setMonth] = useState('')
  const [category, setCategory] = useState('')
  const [q, setQ] = useState('')
  const [remember, setRemember] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<ImportResult | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const [saving, setSaving] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const labels = useMemo(() => Object.fromEntries(categories.map(c => [c.key, c.label])), [categories])
  const currency = startup?.currency ?? 'INR'

  useEffect(() => {
    Promise.all([api.startup(id), api.bankAccounts(id), api.categories(id)])
      .then(([s, a, c]) => { setStartup(s); setAccounts(a); setAccountId(a[0]?.id ?? ''); setCategories(c) })
      .catch(e => setError(errorText(e)))
  }, [id])

  const load = useCallback(async (offset = 0) => {
    const filter = { needs_review: tab === 'review' ? true : undefined, month: month || undefined,
      category: category || undefined, q: q.trim() || undefined, limit: PAGE, offset }
    const [page, s] = await Promise.all([api.transactions(id, filter), api.reviewSummary(id)])
    setItems(prev => offset ? [...prev, ...page.items] : page.items)
    setTotal(page.total)
    setSummary(s)
  }, [id, tab, month, category, q])

  useEffect(() => {
    const t = setTimeout(() => { load().catch(e => setError(errorText(e))) }, q ? 250 : 0)  // wait for typing to pause
    return () => clearTimeout(t)
  }, [load, q])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 4000)
    return () => clearTimeout(t)
  }, [toast])

  async function upload(file: File) {
    if (!accountId) return
    setUploading(true); setError(null); setResult(null)
    try {
      const r = await api.importStatement(id, accountId, file)
      setResult(r)
      setTab(r.needs_review ? 'review' : 'all')
      await load()
    } catch (e) { setError(errorText(e)) } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function pick(t: Transaction, key: string) {
    setSaving(t.id)
    try {
      const r = await api.categorise(id, t.id, key, remember)
      const others = r.also_updated ? ` and ${r.also_updated} other line${r.also_updated > 1 ? 's' : ''} from the same vendor` : ''
      setToast(`Filed as ${labels[key]}${others}.`)
      if (tab === 'review') setItems(prev => prev.filter(x => x.id !== t.id))
      await load()
    } catch (e) { setError(errorText(e)) } finally { setSaving(null) }
  }

  if (error && !startup) return <div className="narrow"><ErrorBox error={error} /><Link to="/">Back</Link></div>
  if (!startup || accounts === null) return <p className="muted">Loading…</p>

  return (
    <div>
      <StartupNav id={id} name={startup.name} />
      <div className="page-head">
        <div>
          <h1>Transactions</h1>
        </div>
        {accounts.length > 0 && (
          <div className="upload-bar">
            {accounts.length > 1 && (
              <select value={accountId} onChange={e => setAccountId(e.target.value)} aria-label="Bank account">
                {accounts.map(a => <option key={a.id} value={a.id}>{a.name}{a.last4 ? ` ··${a.last4}` : ''}</option>)}
              </select>
            )}
            <input ref={fileInput} type="file" accept=".csv,.xlsx,.xls,.xlsm,.txt" hidden
              onChange={e => e.target.files?.[0] && upload(e.target.files[0])} />
            <button className="btn primary" disabled={uploading} onClick={() => fileInput.current?.click()}>
              {uploading ? 'Reading statement…' : 'Upload statement'}
            </button>
          </div>
        )}
      </div>

      <ErrorBox error={error} />

      {accounts.length === 0 ? (
        <AddAccount startupId={id} onAdded={a => { setAccounts([a]); setAccountId(a.id) }} />
      ) : (
        <>
          {result && (
            <div className="banner" role="status">
              <div>
                <strong>{result.filename}</strong>: {result.rows_added} new line{result.rows_added === 1 ? '' : 's'}
                {result.rows_duplicate > 0 && <> ({result.rows_duplicate} already imported, skipped)</>}.{' '}
                {result.auto_filed} filed automatically, {result.needs_review} need{result.needs_review === 1 ? 's' : ''} your review.
                {result.notes.map(n => <div key={n} className="muted small">{n}</div>)}
              </div>
              <button className="btn ghost" onClick={() => setResult(null)} aria-label="Dismiss">✕</button>
            </div>
          )}

          {summary && summary.total > 0 && (
            <div className="stats">
              <div className="card stat"><span className="muted">Transactions</span><strong>{summary.total.toLocaleString('en-IN')}</strong></div>
              <div className="card stat"><span className="muted">Filed automatically</span><strong>{Math.round(summary.auto_filed_share * 100)}%</strong></div>
              <div className={`card stat ${summary.needs_review ? 'attention' : ''}`}>
                <span className="muted">Need your review</span><strong>{summary.needs_review}</strong>
              </div>
            </div>
          )}

          {summary?.total === 0 ? (
            <div className="card empty">
              <h2>Upload your first statement</h2>
              <p className="muted">Download the statement from your bank's net banking as CSV or Excel and upload it here.
                Every line is sorted into a category; you only check the ones we are unsure about.</p>
              <button className="btn primary" onClick={() => fileInput.current?.click()}>Upload statement</button>
            </div>
          ) : (
            <>
              <div className="toolbar">
                <div className="tabs" role="tablist">
                  <button role="tab" aria-selected={tab === 'review'} className={tab === 'review' ? 'on' : ''} onClick={() => setTab('review')}>
                    Needs review{summary ? ` (${summary.needs_review})` : ''}
                  </button>
                  <button role="tab" aria-selected={tab === 'all'} className={tab === 'all' ? 'on' : ''} onClick={() => setTab('all')}>All</button>
                </div>
                <div className="filters">
                  <input type="month" value={month} onChange={e => setMonth(e.target.value)} aria-label="Month" />
                  <select value={category} onChange={e => setCategory(e.target.value)} aria-label="Category filter">
                    <option value="">All categories</option>
                    <option value="uncategorised">Uncategorised</option>
                    {categories.map(c => <option key={c.key} value={c.key}>{c.label}</option>)}
                  </select>
                  <input type="search" value={q} onChange={e => setQ(e.target.value)} placeholder="Search description" aria-label="Search" />
                </div>
              </div>
              <label className="remember">
                <input type="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} />
                When I choose a category, also file this vendor's other and future transactions the same way
              </label>

              {items.length === 0 ? (
                <div className="card empty"><p className="muted">{tab === 'review' ? 'Nothing to review. Every transaction has a category.' : 'No transactions match these filters.'}</p></div>
              ) : (
                <ul className="txns">
                  {items.map(t => (
                    <li key={t.id} className={`card txn ${t.needs_review ? 'review' : ''}`}>
                      <div className="txn-main">
                        <div className="txn-desc">{t.description}</div>
                        <div className="muted small">{day(t.txn_date)} · <span className={`src src-${t.category_source}`}>{SOURCE[t.category_source]}</span> · {t.reason}</div>
                        {t.needs_review && t.suggestions.length > 0 && (
                          <div className="chips">
                            {t.suggestions.map(s => (
                              <button key={s} className="chip" disabled={saving === t.id} onClick={() => pick(t, s)}>{labels[s] ?? s}</button>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className={`txn-amount ${t.direction}`}>{t.direction === 'debit' ? '−' : '+'}{money(t.amount, currency)}</div>
                      <div className="txn-cat">
                        <CategoryPicker value={t.category} categories={categories} disabled={saving === t.id} onPick={k => pick(t, k)} />
                        {t.needs_review && t.category && t.category !== 'person_payment' && (
                          <button className="btn small" disabled={saving === t.id} onClick={() => pick(t, t.category!)}>Confirm</button>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
              {items.length < total && (
                <div className="center"><button className="btn" onClick={() => load(items.length).catch(e => setError(errorText(e)))}>
                  Show more ({total - items.length} left)</button></div>
              )}
            </>
          )}
        </>
      )}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  )
}
