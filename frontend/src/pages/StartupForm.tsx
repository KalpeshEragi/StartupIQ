import { useState, type FormEvent } from 'react'
import { ErrorBox, Field } from '../components/Form'
import { errorText, PRODUCT_TYPES, STAGES, type ProductType, type Stage, type StartupInput } from '../lib/api'

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const today = () => new Date().toISOString().slice(0, 10)

export function StartupForm({ initial, submitLabel, onSubmit }: {
  initial?: Partial<StartupInput>
  submitLabel: string
  onSubmit: (data: StartupInput) => Promise<void>
}) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const f = new FormData(e.currentTarget)
    const data: StartupInput = {
      name: String(f.get('name')).trim(),
      sector: String(f.get('sector')).trim(),
      stage: f.get('stage') as Stage,
      product_type: f.get('product_type') as ProductType,
      founded_on: String(f.get('founded_on')) || null,
      currency: String(f.get('currency')),
      fy_start_month: Number(f.get('fy_start_month')),
    }
    setBusy(true); setError(null)
    try { await onSubmit(data) } catch (err) { setError(errorText(err)) } finally { setBusy(false) }
  }

  return (
    <form className="form-grid" onSubmit={submit}>
      <ErrorBox error={error} />
      <Field label="Startup name"><input name="name" required maxLength={120} defaultValue={initial?.name} /></Field>
      <Field label="Sector" hint="For example: D2C beverages, B2B SaaS, fintech">
        <input name="sector" required maxLength={80} defaultValue={initial?.sector} />
      </Field>
      <Field label="Stage">
        <select name="stage" defaultValue={initial?.stage ?? 'pre_seed'}>
          {Object.entries(STAGES).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      </Field>
      <Field label="What do you sell?" hint="Decides which unit metrics you will track.">
        <div className="choice">
          {Object.entries(PRODUCT_TYPES).map(([v, l]) => (
            <label key={v}><input type="radio" name="product_type" value={v} required
              defaultChecked={(initial?.product_type ?? 'digital') === v} /> {l}</label>
          ))}
        </div>
      </Field>
      <Field label="Founded on (optional)"><input name="founded_on" type="date" max={today()} defaultValue={initial?.founded_on ?? ''} /></Field>
      <Field label="Currency">
        <select name="currency" defaultValue={initial?.currency ?? 'INR'}>
          {['INR', 'USD', 'EUR', 'GBP', 'SGD', 'AED'].map(c => <option key={c}>{c}</option>)}
        </select>
      </Field>
      <Field label="Financial year starts in" hint="April for most Indian companies.">
        <select name="fy_start_month" defaultValue={initial?.fy_start_month ?? 4}>
          {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
        </select>
      </Field>
      <div className="form-actions"><button className="btn primary" disabled={busy}>{busy ? 'Saving…' : submitLabel}</button></div>
    </form>
  )
}
