# Startup IQ

**The founder's monthly operating system.** A founder uploads their bank statement; Startup IQ sorts every
transaction into a P&L automatically, keeps the monthly numbers (revenue, gross profit, EBITDA, burn, runway), the cap
table and investors in one place, forecasts where its market is heading, and warns early when something looks wrong. It learns the
business from the founder's corrections every month.

India-first: Indian bank statement formats (HDFC, SBI, ICICI, Axis, Kotak), NEFT / IMPS / RTGS / UPI narrations, GST
and TDS, INR, and an April financial year by default.

- Full product spec: [docs/PRD.md](docs/PRD.md)
- ML results: [docs/results/categoriser.md](docs/results/categoriser.md)
- Earlier versions of the project: the `legacy` branch

## How it works

```
bank statement (CSV / Excel)
   -> parse: find the table, columns, dates, debit/credit, skip duplicates
   -> categorise each line:  1. founder's vendor memory
                             2. Indian bank-format rules (receipts, GST/TDS, EMIs, bank charges, cash, people)
                             3. ML model (kind of expense), filed only when confident
                             4. otherwise the review inbox, with suggestions
   -> founder reviews and corrects; corrections are remembered and become training data
   -> monthly P&L, cash, runway, unit economics, dashboard
   -> market forecasts (deep learning), alerts and recommendations
```

Most numbers come from the bank statement. The founder only types what a statement cannot know:

| Data | Source | Founder effort |
| --- | --- | --- |
| Revenue, expenses, salaries, GST, EMIs | Bank statement, categorised automatically | None |
| Cash in bank | Balance column of the statement (or one opening balance) | None, or one number |
| Cost of goods (for gross profit) | Founder marks supplier lines once; memory repeats it | A few clicks, once |
| EBITDA, burn, runway, margins | Calculated | None |
| Units, orders, customers, churn | Small monthly form (later: Razorpay / Shopify / Stripe integrations) | About a minute a month |
| Cap table | One-time setup, then each new round | Ten minutes once |
| Investors and pipeline | Added as conversations happen | As needed |

## Roadmap

### Done

**Phase 1: Foundation**
- Sign up, sign in, secure sessions (argon2 passwords, 15-minute access tokens, rotating refresh cookies)
- Create and edit a startup (sector, stage, physical or digital product, currency, financial year)
- Membership-based access check on every startup endpoint (founder role now; built for more roles later)
- Audit log; React frontend; Docker Compose; GitHub Actions CI with tests on PostgreSQL

**Phase 2: Transactions and ML categorisation**
- Statement import for CSV and Excel: title rows, different column names, Withdrawal/Deposit or Amount + Dr/Cr,
  Indian number formats, duplicate detection on re-upload
- Indian bank-format rules: customer receipts, payment-gateway settlements, GST/TDS, EMIs, bank charges, cash, payments
  to people (phone-number UPI IDs, bare names after NEFT)
- ML expense categoriser trained on about a million real UK MP office expense claims (IPSA, Open Government Licence),
  mapped to 14 startup expense categories; 6 approaches compared, Linear SVM chosen
  (test macro-F1 0.866 vs 0.506 for keyword rules)
- Safety: lines whose words the model has never seen are never auto-filed, only suggested
- Review inbox with one-click suggestions; vendor memory ("file this vendor the same way from now on")
- 18-statement synthetic demo pack across 5 bank layouts (kept outside the repo)

**Phase 3: Financials and dashboard**
- Monthly P&L from categorised transactions: revenue, other income, cost of goods, gross profit and margin, operating
  expenses by category, EBITDA
- Cash flow: cash in and out, money raised, tax paid, loan principal; closing cash per month from statement balances
- Burn and runway (average operating burn of the last three months)
- "Cost of goods" category the founder assigns (vendor memory repeats it)
- Unit economics from a small monthly form:
  - physical products: units, orders, inventory, average order value, cost of goods per unit, months of stock
  - digital products: customers, new and churned customers, MRR, churn rate, ARPU, CAC, LTV, LTV:CAC, payback months
- Dashboard: KPI tiles, 12-month revenue / expenses / cash chart, top expenses, what still needs review

**Phase 4: Cap table and investors**
- Shareholders (founders, investors, employees, ESOP pool), share issues, funding rounds
- SAFEs and convertible notes (valuation cap, discount), converted at the next priced round
- ESOP grants with vesting (default 4 years, 1-year cliff): vested vs unvested
- Fully diluted ownership and its history round by round
- Dilution simulator: "raise X at a pre-money of Y with a Z% pool top-up: what does everyone own after?"
- Quick setup for a new company (founders and pool at incorporation)
- Investor list and fundraising pipeline (contacted, meeting, due diligence, term sheet, closed, passed)
- Record a simulated round in one click (investor shares, SAFE conversions, pool top-up), and undo it

**Data trust and verification**
- Every uploaded statement gets a file fingerprint (SHA-256) and a running-balance check: each line's balance must
  equal the previous balance plus money in, minus money out. Breaks are shown with their line number
- Gaps between statements of the same account are flagged (one should start where the last one ended)
- Each import's stored lines are sealed with one hash and rechecked on every report, so edits made directly in the
  database are caught
- Raw lines are never edited; the app's first category decision is kept beside the founder's corrections; each line
  has a history (file, line number, every change)
- The startup's audit log is a hash chain; the Trust page shows whether it is intact and prints its latest hash
- Trust page: overall status, share of lines from statements that add up, auto-filed vs set by the founder, month by
  month; printable as an investor report. P&L cells link to the transactions behind them
- Statements uploaded before these checks show "Not checked" until the same file is uploaded again
- Odd inputs handled (fuzzed on SQLite and PostgreSQL): UTF-16 / Windows-1252 files, dates before 2000 or in the
  future, lines with both a withdrawal and a deposit, impossible amounts, files over 50,000 lines (refused, not cut),
  lines without a balance (checked through the next one), numbers beyond the database columns (rejected with 422)
- Limits: cannot prove a file came from the bank (only bank-signed data can); details in docs/PRD.md section 9

### Planned

**Phase 5: Market behaviour forecasting (deep learning)**
- Forecast where interest in the startup's sector is heading over the next 4-12 weeks, with P10 / P50 / P90
  ranges ("interest in your market is expected to rise about 12% by October")
- Market interest index per sector from daily Wikipedia page views (CC0) of about 170 topic and brand articles in
  English and Hindi, 19 sectors, 2023-2026 only (post-COVID, UPI and AI-era). Code in ml/market/
- Corrected for the chatbot effect: share of all Wikipedia views, then relative to the median of all sectors
- Trained on GPU: LSTM with attention test MASE 1.367 vs 1.477 for the best baseline (exponential smoothing),
  P10-P90 coverage 91% vs 69%; wins 10 of 19 sectors. Details: ml/market/RESULTS.md. Next: Market tab in the app
- Models compared: MLP, LSTM (with attention), 1D CNN (TCN), against statistical baselines (naive, seasonal naive, ETS);
  every model backtested on the same hidden recent period, the best one used per sector
- "You vs your market": how the startup's revenue moves with its market and whether it is beating it (simple statistics)
- Training on GPU (PyTorch). Why deep learning and not rules or statistics: see docs/WHY_DEEP_LEARNING.md

**Phase 6: Insights**
- Anomaly alerts: spend outside its expected range, duplicate payments, sudden revenue drops (Isolation Forest plus
  forecast-range checks)
- Rule-based health checks and recommendations (runway, burn multiple, margin trend, LTV:CAC, churn), each showing the
  numbers behind it
- Monthly investor update drafted from the real numbers (every number comes from the database)
- Month close: lock a month once its numbers are final

**Phase 7: Hardening**
- Security review and tests, performance, error tracking, structured logs
- Deployment (managed PostgreSQL, HTTPS), backups, publishing the trained model as a release file
- Final demo script

**Phase 8: Roles**
- Team members and read-only investors, invites by email
- Investor portal: an investor sees only what the founder shares (their holding, updates, chosen KPIs)

### Future: market intelligence agents

**Competitor watch agent**: tracks the startup's named competitors (new products and features, pricing changes, sales
and promotions, funding rounds, launches, hiring) from their websites, changelogs, pricing pages, app-store listings,
news and job boards. Output: a weekly competitor digest plus instant alerts for big moves, every item linked to its
source.

**Customer voice agent**: reads customer reviews (app stores, Google, review sites), Reddit, Quora and other public
forums about competitors and the market. Output: top complaints, most-requested features, what customers love, and gaps
no competitor fills, with example quotes, links and how often each comes up; feeds goals and planning.

Ground rules for both: public information only, follow each site's terms and use official APIs, every claim linked to
its source, no invented facts, no storing reviewers' personal data.

Also later: live bank connection through India's Account Aggregator framework (needs regulatory onboarding), data room,
churn prediction per customer, invoice payment-date prediction, investor matching.

## Where the ML and DL are (term work)

| Part | What it does | Data | Status |
| --- | --- | --- | --- |
| ML: expense categorisation | Bank line -> kind of expense; 6 approaches compared | ~1M real IPSA expense claims; time-based split | Done |
| DL: market behaviour forecasting | Sector market interest with a range; MLP vs LSTM vs 1D CNN vs statistics | Wikipedia page views (CC0), 2023-2026 only | Phase 5 |
| Unsupervised ML: anomalies | Unusual spend, duplicates | Same data plus injected anomalies to measure precision | Phase 6 |
| Rules | Indian bank formats, health checks, recommendations | Written from finance practice, each rule tested | Done, growing |

Deliberately not ML: P&L arithmetic, runway, cap table and dilution (they must be exact).

## Tech stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.12+, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL (SQLite for quick local runs and tests) |
| Auth | argon2, JWT access tokens, rotating httpOnly refresh cookies, per-startup membership checks |
| ML | scikit-learn, LightGBM (`ml/`), model exported as a plain scikit-learn artefact for the backend |
| DL (Phase 5) | PyTorch on GPU |
| Frontend | React, TypeScript, Vite |
| Ops | Docker Compose (PostgreSQL, API, web via nginx), GitHub Actions CI, ruff, oxlint |

## Layout

```
backend/
  app/core/            settings, database, password hashing and tokens
  app/models/          users, startups, memberships, audit log, bank accounts, transactions, unit metrics
  app/api/             routes, and the access check every startup endpoint uses (deps.py)
  app/categorisation/  statement parsing, Indian bank-format rules, text cleaning, model loading, the categoriser
  app/finance/         P&L, cash, runway and unit economics
  app/services/        audit log (hash chain) and statement verification
  alembic/             database migrations
  tests/               API, security, import, categorisation and migration tests
frontend/              React + TypeScript + Vite
ml/                    training the expense categoriser: data download, labelled dataset, model comparison, export
docs/                  product spec (PRD), results, a sample statement
```

## Getting started

### With Docker

```bash
cp .env.example .env          # then set SIQ_SECRET_KEY
docker compose up --build     # app on http://localhost:8080, API docs on http://localhost:8000/api/docs
```

### Without Docker (development)

Backend, using a local SQLite file instead of PostgreSQL:

```bash
cd backend
python -m venv ../.venv
../.venv/Scripts/pip install -e ".[dev]"            # macOS/Linux: ../.venv/bin/pip
export SIQ_DATABASE_URL=sqlite:///./local.db       # Windows cmd: set SIQ_DATABASE_URL=sqlite:///./local.db
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Frontend (proxies `/api` to port 8000):

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

### The trained model

The categoriser model file is not committed (`backend/models/categoriser.joblib`). Without it the app still runs and
sends expense lines to the review inbox. To build it, follow [ml/README.md](ml/README.md) (downloads public data, trains,
exports; about 30 minutes on a laptop CPU).

Try it with the sample statement in `docs/samples/`.

## Checks

```bash
cd backend && ruff check . && pytest        # set SIQ_TEST_DATABASE_URL to a PostgreSQL URL to test on PostgreSQL
cd ml && ruff check categoriser tests && pytest
cd frontend && npm run lint && npm run build
```

CI runs the same checks on every push, with the backend tests on PostgreSQL.

## Working together

- Branch from `main`, one topic per branch, open a pull request; CI must pass before merging.
- Small, focused commits with a clear subject line saying what changed and a body saying why.
- Every model must beat a stated baseline; choose on validation, score the test set once, and write results to
  `docs/results/`. Synthetic data is for demos only, never for accuracy claims.
- Never commit secrets, `.env` files, datasets or model files (`.gitignore` covers them).
- Money is stored as decimals, never floats; every startup-scoped query goes through the membership check.

## Security notes

- Passwords hashed with argon2; an unknown email takes as long to reject as a wrong password.
- Access tokens last 15 minutes and live only in browser memory. Refresh tokens are httpOnly, SameSite=Strict cookies,
  stored server-side only as SHA-256 hashes and rotated on every use; reusing a rotated token ends the whole session.
- Every startup endpoint checks membership; non-members get 404, so other startups' existence is not revealed.
- Sign-ups, logins, failed logins, imports and every change are written to the audit log.
- Bank statements are processed into transactions; only the last four digits of an account number are kept.
- In production the API refuses to start without a real `SIQ_SECRET_KEY`.
