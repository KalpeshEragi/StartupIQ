import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import '../landing.css'

/* A few lines in the shapes Indian bank statements really use, and where StartupIQ files them. */
const STRIP = [
  { line: 'NEFT-RAZORPAY SETTLEMENT 3534', amt: '+₹2,03,443', cat: 'Revenue' },
  { line: 'ACH D- AMAZON WEB SERVICES', amt: '−₹38,210', cat: 'Software & IT' },
  { line: 'UPI/98xxxxxx10@ybl/RAVI KUMAR', amt: '−₹15,000', cat: 'Payment to a person' },
  { line: 'GST PAYMENT CPIN 2611', amt: '−₹54,900', cat: 'Tax · cash flow only' },
]

type Answer = { cat: string; where: string; how: string; tone: 'auto' | 'ask' }

/** The same idea as the real categoriser, in miniature: known formats first, then known vendors. */
function classify(raw: string): Answer {
  const s = raw.toLowerCase()
  const rule = (re: RegExp) => re.test(s)
  if (rule(/razorpay|cashfree|payu|stripe|settlement/)) return { cat: 'Revenue', where: 'P&L · revenue', how: 'Payment gateway settlement: money your customers paid.', tone: 'auto' }
  if (rule(/\bgst\b|cpin|\btds\b/)) return { cat: 'Tax payment', where: 'Cash flow only, not an expense', how: 'GST and TDS move cash but are not costs of running the business.', tone: 'auto' }
  if (rule(/\bemi\b|loan/)) return { cat: 'Loan repayment', where: 'Cash flow only', how: 'Repaying principal is not an expense.', tone: 'auto' }
  if (rule(/salary|payroll/)) return { cat: 'People', where: 'P&L · operating expenses', how: 'Salaries and payroll.', tone: 'auto' }
  if (rule(/aws|amazon web|google cloud|azure|github|figma|notion|zoho|slack/)) return { cat: 'Software & IT', where: 'P&L · operating expenses', how: 'A software vendor the model has seen thousands of times.', tone: 'auto' }
  if (rule(/google ads|facebook|meta|instagram|linkedin ads/)) return { cat: 'Marketing', where: 'P&L · operating expenses', how: 'Advertising spend.', tone: 'auto' }
  if (rule(/rent|wework|awfis|91springboard/)) return { cat: 'Rent & premises', where: 'P&L · operating expenses', how: 'Office rent or coworking.', tone: 'auto' }
  if (rule(/uber|\bola\b|indigo|makemytrip|irctc|vistara/)) return { cat: 'Travel', where: 'P&L · operating expenses', how: 'Flights, trains and cabs.', tone: 'auto' }
  if (rule(/swiggy|zomato|blinkit|zepto/)) return { cat: 'Meals & groceries', where: 'P&L · operating expenses', how: 'Food delivery and quick commerce.', tone: 'auto' }
  if (rule(/upi\/\d|upi\/9|@ybl|@okaxis|@oksbi|@paytm/)) return { cat: 'Payment to a person', where: 'Waits for you', how: 'A phone-number UPI ID is a person. Salary? Freelancer? You say once, it remembers.', tone: 'ask' }
  return { cat: 'New vendor', where: 'Waits for you', how: 'Words it has never seen are never filed on a guess. You pick, it remembers the vendor.', tone: 'ask' }
}

const EXAMPLES = ['NEFT-RAZORPAY SETTLEMENT 3534', 'GST PAYMENT CPIN 2611', 'POS 4021 WEWORK INDIA BLR']

const STAGES = [
  { name: 'Upload', clause: 'First, upload the statement' },
  { name: 'Check', clause: 'Then, settle the few it asks' },
  { name: 'Read', clause: 'Next, read your month' },
  { name: 'Plan', clause: 'Finally, plan the next round' },
]

function StageFragment({ stage }: { stage: number }) {
  if (stage === 0) return (
    <div className="lp-ui" data-ui>
      <div className="lp-ui-head"><strong>HDFC current account</strong><span>Sep–Nov 2026.csv</span></div>
      <p className="lp-ui-big">312 lines read</p>
      <ul className="lp-ui-list">
        <li><span>Filed automatically</span><b>251</b></li>
        <li><span>Already imported, skipped</span><b>28</b></li>
        <li><span>Need your review</span><b className="lp-warn">33</b></li>
      </ul>
    </div>
  )
  if (stage === 1) return (
    <div className="lp-ui" data-ui>
      <div className="lp-ui-head"><strong>Needs review</strong><span>33 left</span></div>
      <p className="lp-ui-line">UPI/98xxxxxx10@ybl/RAVI KUMAR <b>−₹15,000</b></p>
      <p className="lp-ui-note">A phone-number UPI ID is a person. Best guesses:</p>
      <div className="lp-ui-chips"><span className="on">People</span><span>Professional services</span></div>
      <p className="lp-ui-note">Remember: file Ravi’s other and future payments the same way</p>
    </div>
  )
  if (stage === 2) return <Brief compact />
  return (
    <div className="lp-ui" data-ui>
      <div className="lp-ui-head"><strong>Raise ₹2 Cr at ₹8 Cr pre</strong><span>10% pool</span></div>
      <table className="lp-ui-table"><tbody>
        <tr><td>Asha (founder)</td><td>50.0% → 34.3%</td></tr>
        <tr><td>Ravi (founder)</td><td>40.0% → 27.5%</td></tr>
        <tr className="b"><td>New investor</td><td>20.0%</td></tr>
        <tr><td>Angel SAFE converts</td><td>6.9%</td></tr>
        <tr><td>Price per share</td><td>₹6.86</td></tr>
      </tbody></table>
    </div>
  )
}

function Brief({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`lp-brief-card${compact ? ' compact' : ''}`} data-ui>
      <p className="lp-brief-kick">Monthly brief · November 2026 · 33 lines to sort</p>
      <p className="lp-brief-head">Chai Labs earned ₹17 L in November and kept more cash than it spent.</p>
      {!compact && <p className="lp-brief-lede">Revenue was down 14% on October’s festive peak. Cash coming in covered everything going out.</p>}
      <dl className="lp-brief-figs">
        <div><dt>EBITDA</dt><dd className="lp-good">₹8.5 L</dd></div>
        <div><dt>Net burn</dt><dd>None</dd></div>
        <div><dt>Biggest cost</dt><dd>People, ₹4.0 L</dd></div>
        {!compact && <div><dt>Runway</dt><dd>Cash positive</dd></div>}
      </dl>
    </div>
  )
}

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>('.lp [data-reveal]')
    if (!('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('in')); return }
    const io = new IntersectionObserver(entries => entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target) }
    }), { threshold: 0.15 })
    els.forEach(e => io.observe(e))
    return () => io.disconnect()
  }, [])
}

export function Landing() {
  const [strip, setStrip] = useState(0)
  const [line, setLine] = useState('UPI/98xxxxxx10@ybl/RAVI KUMAR')
  const [stage, setStage] = useState(0)
  const [paused, setPaused] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const answer = classify(line)
  useReveal()

  useEffect(() => {
    if (paused || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    const t = setInterval(() => setStrip(i => (i + 1) % STRIP.length), 2600)
    return () => clearInterval(t)
  }, [paused])

  const row = STRIP[strip]
  return (
    <div className="lp">
      <nav className="lp-nav" aria-label="Main">
        <Link to="/welcome" className="lp-word">StartupIQ</Link>
        <div><Link to="/login">Sign in</Link><Link to="/register">Create account</Link></div>
      </nav>

      <header className="lp-hero">
        <h1 className="lp-sentence">
          <span>Statement in,</span>
          <span className="lp-strip" data-ui aria-hidden="true">
            <span className="lp-strip-row" key={strip}>
              <span className="lp-strip-line">{row.line}</span>
              <span className="lp-strip-amt">{row.amt}</span>
              <span className="lp-strip-cat">→ {row.cat}</span>
            </span>
          </span>
          <span>brief out.</span>
        </h1>
        <p className="lp-sub">Upload your bank statement. StartupIQ sorts every line and writes your month: revenue, burn, runway and who owns what.</p>
        <Link to="/register" className="lp-btn">Create free account</Link>
        <button type="button" className="lp-pause" onClick={() => setPaused(p => !p)} aria-pressed={paused}>
          {paused ? 'Play the example' : 'Pause the example'}
        </button>
      </header>

      <section className="lp-show" aria-labelledby="try-h">
        <h2 id="try-h" className="lp-kick">Type any line from your statement</h2>
        <label className="lp-sr" htmlFor="try">Type a line from a bank statement</label>
        <input id="try" ref={input} className="lp-type" data-ui data-reveal value={line} spellCheck={false} autoComplete="off"
          onChange={e => setLine(e.target.value)} />
        <div className="lp-answer" data-ui aria-live="polite" data-reveal>
          <p className="lp-answer-cat">{answer.cat}</p>
          <p className="lp-answer-where">{answer.where} <span className={answer.tone === 'auto' ? 'lp-good' : 'lp-warn'}>{answer.tone === 'auto' ? 'Filed automatically' : 'You confirm once'}</span></p>
          <p className="lp-answer-how">{answer.how}</p>
        </div>
        <div className="lp-examples" data-reveal>
          <p>Try another line from a statement:</p>
          {EXAMPLES.map(x => <button key={x} type="button" onClick={() => { setLine(x); input.current?.focus() }}>{x}</button>)}
        </div>
      </section>

      <section className="lp-brief" aria-labelledby="brief-h">
        <div className="lp-brief-text">
          <h2 id="brief-h">This is what comes out.</h2>
          <p>Every month opens with one sentence written from your own ledger, the few figures investors ask for, and an honest
            note when lines are still unsorted.</p>
        </div>
        <Brief />
      </section>

      <section className="lp-stages" aria-labelledby="stage-h">
        <div className="lp-stage-left">
          <p className="lp-kick">How it works</p>
          <h2 id="stage-h" className="lp-clause"><span key={stage} className="lp-swap">{STAGES[stage].clause}</span></h2>
        </div>
        <div className="lp-stage-right"><div key={stage} className="lp-fade"><StageFragment stage={stage} /></div></div>
        <div className="lp-slider" role="tablist" aria-label="How it works">
          {STAGES.map((s, i) => (
            <button key={s.name} type="button" role="tab" aria-selected={i === stage} className={i === stage ? 'on' : ''}
              onClick={() => setStage(i)} onKeyDown={e => {
                if (e.key === 'ArrowRight') setStage(x => Math.min(3, x + 1))
                if (e.key === 'ArrowLeft') setStage(x => Math.max(0, x - 1))
              }}><span className="lp-step-bar" />{s.name}</button>
          ))}
        </div>
        <ol className="lp-steps-sm">
          {STAGES.map((s, i) => <li key={s.name}><h3>{s.clause}</h3><StageFragment stage={i} /></li>)}
        </ol>
      </section>

      <section className="lp-para" aria-labelledby="what-h">
        <h2 id="what-h" className="lp-kick">What it does</h2>
        <p data-reveal>
          Every line of your statement lands in a category <span className="lp-slot" data-ui><i>People</i></span>, your P&amp;L closes itself
          each month <span className="lp-slot" data-ui><i>₹8.5 L</i></span>, the cap table shows who owns what after the next round <span className="lp-slot lp-slot-bar" data-ui aria-label="ownership bar"><i style={{ width: '34%' }} /><i style={{ width: '28%' }} /><i style={{ width: '20%' }} /></span>, and your
          market’s outlook says where your sector is heading <span className="lp-slot lp-slot-spark" data-ui aria-label="rising trend"><i style={{ height: '40%' }} /><i style={{ height: '55%' }} /><i style={{ height: '50%' }} /><i style={{ height: '78%' }} /></span>.
        </p>
        <div className="lp-para-foot" data-reveal>
          <p>Built for Indian statements: UPI, NEFT, IMPS, GST and TDS lines, lakh and crore, and the layouts of HDFC, ICICI, SBI, Axis and Kotak.</p>
          <Link to="/register" className="lp-btn">Upload your first statement</Link>
        </div>
      </section>

      <section className="lp-ledger" aria-labelledby="ledger-h">
        <figure className="lp-folio" data-reveal>
          <img src="/landing/assets/folio-of-a-genealogical-register-bahi-khata.jpg" alt="A handwritten page of a bahi, a register kept in Haridwar" width="1480" height="2331" loading="lazy" />
          <figcaption>A page from a bahi, the handwritten register kept for generations. Haridwar, c. 1700–1920.</figcaption>
        </figure>
        <div className="lp-ledger-text">
          <h2 id="ledger-h">An old habit, kept monthly.</h2>
          <p data-reveal>Indian families and traders have kept a bahi for generations, and many businesses still open a new account book at Diwali.
            StartupIQ keeps the same habit for a startup: every month closes into one clean page you can read in a minute and hand to an investor.</p>
          <figure className="lp-script" data-reveal>
            <img src="/landing/assets/mahajani-varnamala.png" alt="Letters and numerals of the Mahajani script" width="1501" height="2415" loading="lazy" />
            <figcaption>Mahajani, the script north Indian merchants used for their accounts.</figcaption>
          </figure>
        </div>
      </section>

      <section className="lp-faq" aria-labelledby="faq-h">
        <div className="lp-row"><h2 id="faq-h" className="lp-q lp-soft">Fair questions</h2><span /></div>
        {[
          ['Which banks does it read?', 'CSV and Excel exports from HDFC, ICICI, SBI, Axis and Kotak, including title rows, Withdrawal/Deposit or Dr/Cr columns and Indian number formats. Re-uploading the same months skips lines you already have.'],
          ['What does the machine learning do?', 'It reads each line and picks one of 14 expense categories. On 18 synthetic test statements in real Indian bank layouts it filed 80% of lines on the first upload, and 97% of what it filed by itself was right. Lines with words it has never seen wait for you.'],
          ['What if it files a line wrong?', 'Change it in one click. Tick “remember” and that vendor’s past and future lines follow, so each month needs fewer fixes.'],
          ['Does it replace my CA?', 'No. It gives you the month while it is still useful, so the conversation with your CA and your investors starts from numbers you already understand.'],
          ['Who can see my data?', 'Only people you add to your startup. Every change is logged, and nothing is shared with anyone else.'],
          ['How is my account protected?', 'Passwords are stored as Argon2 hashes, never in plain text. Sign-in tokens expire after 15 minutes and are renewed through a secure cookie, and a stolen renewal token stops working the moment it is reused.'],
          ['Who builds StartupIQ?', 'It is an open-source project: every line of the code, including the categoriser and how it was tested, is public on GitHub.'],
        ].map(([q, a]) => (
          <div className="lp-row" key={q} data-reveal><h3 className="lp-q">{q}</h3><p className="lp-a">{a}</p></div>
        ))}
      </section>

      <section className="lp-cta" aria-labelledby="cta-h">
        <h2 id="cta-h">Close your month in minutes.</h2>
        <div className="lp-cta-foot" data-reveal>
          <p>One bank statement is enough to see revenue, burn and runway for every month it covers.</p>
          <Link to="/register" className="lp-btn lp-btn-invert">Create free account</Link>
        </div>
      </section>

      <footer className="lp-foot">
        <div className="lp-foot-top">
          <p>The monthly operating system for Indian founders.<br /><a href="https://github.com/KalpeshEragi/StartupIQ">Source code on GitHub</a></p>
          <a href="#top" onClick={e => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>Back to top</a>
        </div>
        <p className="lp-contact" data-reveal>StartupIQ</p>
        <div className="lp-meta"><span><Link to="/login">Sign in</Link> · <Link to="/register">Create account</Link></span><span className="lp-credits">Pictures via Wikimedia Commons: <a href="https://commons.wikimedia.org/wiki/File:Folio_of_a_genealogical_register_(%27Bahi-Khata%27)_of_Haridwar_for_a_local_village_near_Kanpur.jpg">bahi folio</a> (public domain) and <a href="https://commons.wikimedia.org/wiki/File:Mahajani_Varnamala.png">Mahajani Varnamala</a> by Symphonicallyu (CC0)</span><span>India · 2026</span></div>
      </footer>
    </div>
  )
}
