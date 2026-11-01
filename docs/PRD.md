# Startup IQ: Product Requirements Document

Version 1.0 · 1 November 2026 · Status: draft for build

## 1. Summary

Startup IQ is the founder's monthly operating system. A founder connects their bank transaction history; the platform
categorises every transaction into a P&L automatically, keeps month-by-month financials (revenue, gross profit, EBITDA,
burn, runway), the cap table and investors in one place, forecasts where its market is heading with a range, flags anomalies, and
gives investors a read-only view of what the founder chooses to share.

Machine learning is used only where it removes manual work or catches problems earlier:

| Intelligence | Job | Kind |
| --- | --- | --- |
| Transaction categorisation | Turn raw bank lines into P&L categories | Supervised ML (text + amount), learns from founder corrections |
| Market behaviour forecasting | Sector market interest, next 4-12 weeks, P10/P50/P90 | Deep learning (MLP, LSTM, 1D CNN) vs statistical baselines |
| Anomaly detection | Flag unusual spend, duplicates, revenue drops | Unsupervised ML + usual-range checks |
| Rules engine | Health checks, alerts, recommendations | Deterministic rules |

Everything that must be exact (P&L arithmetic, cap table, dilution, vesting, SAFE conversion) is deterministic by design.

## 2. Problem

- Early-stage founders keep finances in spreadsheets and bank statements. Sorting hundreds of transactions into a P&L
  takes hours a month, so numbers are 1-2 months late and often wrong.
- Runway is computed as cash / burn, which ignores seasonality, growth and planned hires.
- Cap tables live in spreadsheets; dilution from a new round or ESOP top-up is guesswork.
- Investor updates are assembled by hand from several sources, so they are irregular.
- Problems (a doubled subscription, a duplicate payment, a revenue drop from the top customer) are noticed weeks late.

## 3. Users and roles

| Role | Can do |
| --- | --- |
| Founder (owner) | Everything in their startup workspace; invite and remove members; share with investors |
| Team member | Edit financials, transactions, goals; cap table only if granted |
| Investor (viewer) | Read-only: their holding, shared updates, chosen KPIs |
| Platform admin | Support, monitoring; no access to startup data without an audit-logged grant |

A user can belong to several startups (e.g. an investor in many) with a different role in each.

**Rollout:** v1 ships the founder role only. The data model is multi-role from day one: every user-startup link is a row
in `memberships(user, startup, role)` and every endpoint goes through one access check, so adding team and investor
roles later means new role values, permissions in that one check, and invite/portal pages, with no change to existing
tables or queries.

Startup types supported from day one: **physical product** (D2C, hardware, FMCG) and **digital product** (SaaS, apps,
platforms). The type changes which unit metrics are shown, not the P&L.

## 4. Goals and success metrics

| Goal | Metric | Target (demo / pilot) |
| --- | --- | --- |
| Remove bookkeeping work | Share of transactions auto-categorised at >= 95% precision | >= 80% |
| Categorisation quality | Macro-F1 on held-out transactions | Report vs keyword-rule baseline |
| Trustworthy forecasts | MASE vs statistical baselines on held-out series; P10-P90 coverage | Beat best baseline; coverage 70-90% |
| Useful alerts | Precision of anomaly flags on injected anomalies | >= 70% at <= 1 false alarm per month |
| Timely numbers | Days from month-end to closed month | < 3 days |
| Engagement (pilot) | Founders closing a month every month | 3 consecutive months |

Non-goals: tax filing, payroll, accounting-standard statements (Ind AS), legal documents, startup success prediction.

## 5. Modules and pages

### A. Account and workspace
- **Landing** - value proposition, sign up, log in.
- **Sign up / log in** - email + password (argon2 hash), email verification, password reset, session refresh.
- **Create startup** - name, sector, stage (idea, pre-seed, seed, Series A+), physical or digital, founding date,
  currency (INR default), financial year start (April default).
- **Team and access** (later phase) - invite by email with role; revoke; every change written to the audit log.

### B. Dashboard
- KPI tiles: revenue (month), MoM growth, gross margin, EBITDA, net burn, cash, runway.
- 12-month revenue, expenses and cash chart.
- Goal progress, top recommendations, open anomalies, uncategorised transactions waiting for review.

### C. Transactions (ML: categorisation)
- **Connect / import** - upload bank statement (CSV, Excel, PDF export) in v1; bank connection through India's Account
  Aggregator framework planned (see 9).
- **Inbox** - each transaction with predicted category and confidence. High confidence auto-filed; low confidence
  queued for review with the top 3 suggestions.
- **Review** - accept, change category, or create a rule ("always file RAZORPAY SETTLEMENT as Revenue"). Every change
  is stored as a labelled example.
- **Categories** - default chart of P&L categories, editable per startup.

### D. Financials (monthly P&L)
- Built from categorised transactions, or entered/imported per month.
- Per month: revenue, COGS, gross profit, operating expenses by category, EBITDA, depreciation, interest, tax, net
  profit, cash balance, net burn.
- Derived: margins, MoM/YoY growth, burn multiple, runway.
- **Month close** locks a month; later edits require reopening and are audit-logged.

### E. Unit economics
- Physical: units sold, average order value, COGS per unit, inventory value, days of stock, return rate.
- Digital: MRR, new/expansion/churned MRR, customers, churn %, CAC, LTV, LTV:CAC, payback months.

### F. Cap table and equity
- **Shareholders** - founders, investors, ESOP pool; share classes.
- **Rounds** - amount, pre-money, price per share, participants.
- **Instruments** - SAFEs and convertible notes (cap, discount) with conversion at a priced round.
- **Ownership** - fully diluted %, round-by-round history chart.
- **Dilution simulator** - new round size, pre-money, ESOP top-up -> ownership after.
- **ESOP** - grants, vesting (default 4 years, 1-year cliff), vested vs unvested.

### G. Investors
- **Investor list** - firm, contact, amount, round, ownership.
- **Pipeline** - prospects by stage: contacted, meeting, due diligence, term sheet, closed / passed.
- **Monthly update** - generated from closed-month numbers, goals and anomalies; founder edits; share by link or email.
- **Investor portal** - what an invited investor sees, read-only.

### H. Goals and plan
- Goals with owner, target, deadline; progress computed from actuals.
- Quarterly plan: milestones, budget by category, planned hires (feeds the cash forecast).

### I. Market outlook and scenarios
- Market interest forecast for the startup's sector, next 4-12 weeks, P10/P50/P90 (deep learning).
- "You vs your market": how the startup's revenue moves with its market and whether it is beating it (statistics).
- Simple revenue estimate = own trend adjusted by the market forecast, labelled as an estimate.
- Scenarios: hire, cut spend, raise on a date -> new runway and EBITDA path (exact calculation).
- **Track record** - past forecasts against actuals, shown on the page.

### J. Insights (anomalies + rules)
- Anomalies: category spend outside its forecast range; Isolation Forest on transactions; duplicate payments; revenue
  concentration changes.
- Rules: health checks and recommendations with the numbers behind each (runway, burn multiple, margin trend,
  LTV:CAC, churn).

### K. Settings
Company profile, categories, integrations, data export (CSV/Excel), account deletion, audit log viewer.

### Later (phase 2)
Data room; churn prediction per customer (digital); invoice payment-date prediction (B2B); investor matching; LLM
drafting of updates grounded in stored numbers.

## 6. Intelligence specifications

### 6.1 Transaction categorisation (ML)
- **Input**: description, amount, debit/credit, day of month, recurrence (same counterparty in previous months),
  counterparty if parseable.
- **Features**: character n-gram TF-IDF on a cleaned description (UPI/NEFT/IMPS prefixes, reference numbers removed),
  log amount, flags.
- **Models compared**: keyword rules (baseline), multinomial Naive Bayes, logistic regression, linear SVM, LightGBM, a
  small neural text classifier.
- **Evaluation**: split by account holder (no user in two splits); choose on validation, score test once; macro-F1,
  per-category F1, confusion matrix, coverage at 95% precision.
- **Serving**: auto-file above a confidence threshold chosen on validation for 95% precision; otherwise ask.
- **Learning loop**: founder corrections stored per startup; per-startup overrides apply immediately; global model
  retrained on a schedule, promoted only if it beats the current model on the held-out set.
- **Data**: public labelled transaction dataset (to be selected; licence checked before use) for training and
  evaluation; generated Indian transactions for the demo startups.

### 6.2 Market behaviour forecasting (DL)
- **Why the market, not one startup's sales**: startup financials are private, so no open training data exists;
  market signals are public and recent, and training and testing use the same kind of data. Full reasoning:
  docs/WHY_DEEP_LEARNING.md.
- **Data**: daily Wikipedia page views (Wikimedia Analytics, CC0) of about 170 topic and brand articles in
  English and Hindi across 19 sectors, 2023-2026 only (post-COVID, UPI and AI-era). Views summed over redirects,
  scaled per article, median per sector. Google Trends ruled out (closed API, scraping not allowed); NPCI, GST and
  RBI data are monthly and can be added later as extra inputs.
- **Models**: MLP, LSTM (+attention), 1D CNN (TCN) as one global model across series, quantile outputs;
  baselines: naive, seasonal naive, ETS.
- **Inputs**: past index window, calendar (week, month, festival season), sector, language.
- **Evaluation**: time-based split, rolling origins, choose on validation, test once on the latest hidden period;
  MASE, P10-P90 coverage. Known limit: about three yearly cycles, so festive-month ranges are wider.
- **Serving**: best model per sector; the comparison with baselines is shown.

### 6.3 Anomaly detection (ML)
- Category level: spend outside its usual range from the startup's own history.
- Transaction level: Isolation Forest on amount, counterparty, timing, frequency; exact and near-duplicate checks.
- Evaluation: precision and recall on anomalies injected into held-out data.

### 6.4 Rules engine
- Thresholds from finance practice (runway < 6 months, burn multiple > 2, falling gross margin 3 months running,
  LTV:CAC < 3, monthly churn > 3%).
- Each rule tested with an example and a counter-example.

## 7. Data model (main tables)

`users`, `startups`, `memberships(user, startup, role)`, `bank_accounts`, `transactions`, `categories`,
`category_corrections`, `monthly_financials`, `unit_metrics`, `shareholders`, `share_classes`, `funding_rounds`,
`holdings`, `instruments`, `esop_grants`, `investors`, `pipeline_deals`, `goals`, `plans`, `forecasts`,
`anomalies`, `recommendations`, `investor_updates`, `shares_with_investor`, `audit_log`, `model_versions`.

All startup-owned rows carry `startup_id`; every query is scoped by membership (row-level tenancy).

## 8. Architecture

| Layer | Choice |
| --- | --- |
| Backend | FastAPI, modular routers, Pydantic schemas |
| Database | PostgreSQL, SQLAlchemy 2, Alembic migrations |
| Auth | argon2 password hashes, JWT access + rotating refresh tokens, role checks per startup |
| Jobs | Redis + RQ worker: imports, categorisation, forecasts, update generation |
| ML/DL | Offline training scripts; versioned artefacts in `model_versions`; scoring in the worker |
| Frontend | React + Vite + TypeScript, router, component library, charts |
| Ops | Git, Docker Compose (api, worker, db, redis, web), pytest + CI, structured logs, `.env` config |

## 9. Security, privacy, compliance
- Passwords never stored in plain text; tokens short-lived; HTTPS in deployment.
- Tenant isolation enforced in the data layer and tested.
- Audit log of logins, shares, role changes, month reopenings.
- **Data trust** (Trust page, printable for investors):
  - Each upload: SHA-256 file fingerprint, running-balance check per line (oldest- or newest-first), opening/closing
    balance; gaps between statements of one account flagged.
  - Each import's stored lines sealed with one hash, rechecked on every report: catches lines edited or deleted
    directly in the database.
  - Raw lines have no edit API; the app's first category decision is kept; every founder change is recorded.
  - Startup audit log is a hash chain (each entry holds the previous entry's hash); the latest hash is printed on the
    report so a holder of the report can detect a later rewrite.
  - Limits, stated on the page: a file edited consistently before upload cannot be detected; an attacker with full
    database write access could rebuild hashes after the last report was shared. Real proof of origin needs
    bank-signed data (signed PDF statements, Account Aggregator). Planned: verify signed PDFs, anchor the chain head
    outside the database, 2FA.
- Investors see only what is explicitly shared.
- India's DPDP Act 2023: consent, purpose limitation, deletion on request, data export.
- **Bank connection**: v1 imports statements the founder downloads. Live connection requires the RBI Account
  Aggregator framework (registration as a Financial Information User or a licensed partner). The demo must say
  "upload your statement; live bank connection planned", not claim a live connection.

## 10. Demo data
Three generated startups, 24-36 months each, consistent across all modules, clearly labelled as generated:
1. D2C snacks brand (physical): seasonal sales, inventory, a seed round.
2. B2B SaaS (digital): MRR growth, churn, a SAFE converting at seed, ESOP pool.
3. Marketplace: fast growth, high burn, short runway, fundraise in progress (triggers alerts).
Each comes with raw bank transactions (so categorisation runs live), cap table, investors, goals and past updates.
Models are trained and evaluated on public data only; demo data is never used to report accuracy.

## 11. Release plan

| Phase | Scope | Done when |
| --- | --- | --- |
| 1 Foundation | Repo, Docker Compose, Postgres, auth (founder accounts), startups, membership-based access check, audit log | Sign up, log in, create and edit a startup, other users get 404, tests green |
| 2 Transactions + ML | Import, categorisation models + evaluation, review inbox, corrections | Results report; inbox works on demo data |
| 3 Financials | P&L from transactions, month close, unit economics, dashboard | Dashboard KPIs match hand-computed values |
| 4 Equity + investors | Cap table, rounds, SAFEs, ESOP, dilution simulator, investor list, pipeline | Dilution matches worked examples |
| 5 Forecast + DL | Forecasters, cash forecast, scenarios, track record | Results report; forecast page live |
| 6 Insights | Anomaly detection, rules, recommendations, monthly update | Injected-anomaly precision reported |
| 7 Hardening | Security tests, performance, deployment, demo script | End-to-end demo on a clean machine |
| 8 Roles | Team and investor roles, invites, investor portal, sharing settings | Investor sees only shared data; role tests green |

## 12. Risks and open questions
- Choice and licence of the public transaction dataset; Indian descriptions may differ from it (mitigation: Indian
  cleaning rules + per-startup learning loop).
- Small startups have short histories; forecasts must widen their range and say so.
- Live bank connection needs regulatory onboarding; out of scope for v1.
- Scope is large for one builder; phases 1-3 and 5 are the term-work core.

## 13. Future plans: market intelligence agents

Planned after the core product (phases 1-8). Recorded here so they shape design decisions now.

### 13.1 Competitor watch agent
Keeps track of what the startup's named competitors are doing, and tells the founder what changed and why it matters.
- **Watches:** new products and features, pricing and discount changes, sales and promotions, funding rounds, hiring,
  launches and press coverage.
- **Sources:** competitors' websites, changelogs, pricing pages, app-store listings, press releases, news, job boards.
- **Output:** a weekly "competitor digest" plus instant alerts for big moves (a price cut, a funding round, a launch),
  each with a link to the source.

### 13.2 Customer voice agent
Finds what customers want by reading what they say about competitors and the market.
- **Sources:** customer reviews (app stores, Google, product review sites), Reddit, Quora, and other public forums.
- **Output:** recurring complaints, most-requested features, what customers love, and gaps no competitor fills, each
  with example quotes and links, and how often it comes up.
- **Use in the product:** feeds goals and planning ("customers keep asking for X").

### 13.3 Ground rules for both agents
- Only public information; respect each site's terms of use and use official APIs where they exist (e.g. Reddit's API).
- Every claim links to its source; summaries never invent facts.
- Personal data of reviewers is not stored beyond what is needed to quote a public comment.

More future ideas from the founder will be added to this section as they come up.
