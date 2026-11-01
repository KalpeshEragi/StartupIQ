# StartupIQ: full context for the presentation

Everything needed to present StartupIQ end to end: what it is, how the whole system works, the datasets and why,
how data is cleaned, every model with its real results, how the models are used, every screen, the rule-based
engines, and how security and ethics are handled. Numbers below come from the repo's own result files
(`docs/results/categoriser.md`, `docs/results/categoriser_indian_check.md`, `ml/market/RESULTS.md`).

Legend: **Built** = implemented and tested in the repo. **Planned** = on the roadmap, not built yet.

---

## 1. What StartupIQ is

**One line:** the monthly operating system for Indian founders. Upload a bank statement; StartupIQ sorts every
transaction, builds the P&L, burn and runway, keeps the cap table and investors, and forecasts where the founder's
market is heading.

**The problem:** early-stage founders run on spreadsheets. Every month someone sorts hundreds of bank lines by hand
(UPI, NEFT, GST, TDS, EMIs), the P&L is weeks late, runway is a guess, the cap table lives in a lawyer's PDF, and
investors ask the same questions every update.

**Who it is for:** Indian founders from pre-seed to Series A who are numerate but not accountants, and who open the
tool once a week and properly once a month after downloading the bank statement.

**What makes it different:**
- Built for Indian bank statements: UPI handles, NEFT/IMPS/RTGS narrations, GST and TDS, lakh/crore, the layouts of
  HDFC, ICICI, SBI, Axis and Kotak.
- A machine-learning categoriser that never guesses on words it has not seen and learns each startup's vendors.
- Every month opens as a written brief ("Chai Labs earned ₹17 L in November and kept more cash than it spent.").
- A deep-learning market outlook per sector, with a likely range and an honest track record.

**Term-work components:** ML = expense categorisation (Linear SVM, chosen from 6 approaches). DL = market behaviour
forecasting (MLP, 1D CNN, LSTM with attention). Rule-based = Indian bank formats, accounting rules, cap-table maths,
validation and (planned) health checks.

---

## 2. Whole-system workflow

### 2.1 The founder's month

```
Download bank statement (CSV/Excel) from net banking
        |
        v
Upload in Transactions  -->  parse (any bank layout)  -->  de-duplicate (re-uploads are safe)
        |
        v
Categorise each line:  vendor memory  ->  Indian bank-format rules  ->  ML model  ->  else "needs review"
        |
        v
Review inbox: one-click suggestions; "remember this vendor" files past and future lines the same way
        |
        v
Financials: monthly P&L, cash flow, burn, runway (+ a 1-minute unit-economics form)
        |
        v
Overview: the monthly brief, KPI tiles, chart, where the money went, month-close checklist
        |
        +--> Cap table: ownership, SAFEs, ESOP, dilution simulator, record a round
        +--> Investors: current investors + fundraising pipeline
        +--> Market outlook: 12-week sector forecast (deep learning) + "you vs your market"
```

### 2.2 System architecture

```
 Browser (React + TypeScript + Vite)
    |  HTTPS, JSON; access token in memory, refresh token in an httpOnly cookie
    v
 FastAPI backend (Python)
    |-- auth (argon2, JWT, rotating refresh tokens)        |-- audit log of every change
    |-- membership check on every startup endpoint         |-- categorisation (rules + ML model)
    |-- finance engine (P&L, cash, runway, units)          |-- equity engine (cap table, SAFEs, dilution)
    |-- market service (reads the DL forecast file)
    v
 PostgreSQL (SQLite for local runs and tests), migrations with Alembic

 Offline (ml/ folder, run by the team, never on user data):
    categoriser training  -->  backend/models/categoriser.joblib
    market forecasting    -->  backend/models/market_forecast.json  (refreshed weekly)
```

**Tech stack:** FastAPI, SQLAlchemy 2, Alembic, PostgreSQL; scikit-learn, LightGBM; PyTorch (GPU, RTX 2050);
React 19, TypeScript, Vite; Docker Compose, GitHub Actions CI; ruff and oxlint; pytest (44 backend + 72 ML tests).

---

## 3. Datasets: which, why, licence

| Dataset | Used for | Size | Why this one | Licence |
| --- | --- | --- | --- | --- |
| **IPSA UK MP office expenses** (via `kieraisverybored/uk-mp-expenses`) | Training and testing the expense categoriser | 1.33 million real expense claims, 2010-2027 | No public labelled dataset of startup business transactions exists. IPSA is real, large, labelled by cost type, and its cost types (rent, software, travel, phone, office supplies...) map onto startup expense categories | Open Government Licence |
| **AgamiAI Indian Bank Statements** | Checking the Indian bank-format rules and the model on Indian narrations | 200 statements, 32,349 lines | Real Indian current-account narration formats (NEFT/IMPS/RTGS/UPI/cheque) | Apache-2.0 (synthetic) |
| **DoDataThings US bank transaction categories** | Optional external benchmark | 68,000 labelled lines | Cross-check on bank-style descriptions | MIT |
| **Wikipedia page views (Wikimedia Analytics API)** | Training the deep-learning market forecaster | ~170 articles x ~1,400 days (2023-2026), 19 sectors | Free, daily, recent and openly licensed signal of what people in a market pay attention to; 2023-2026 only so the model learns the post-COVID, UPI and AI-era market | CC0 (public domain) |
| **Demo statement pack (ours)** | Demo and end-to-end accuracy check | 18 statements, 5 bank layouts | Exercises the real product flow | Synthetic, built from public formats; always described as synthetic |

**Ruled out, and why** (good viva material):
- Real startup bank data: private; never used for training.
- Kaggle "startup" datasets: funding rounds (Crunchbase-style), not transactions or monthly revenue.
- Google Trends: official API is a closed alpha; the free library (pytrends) was archived in 2025; scraping breaks
  Google's terms.
- NPCI UPI, GST and RBI payment data: public but monthly (~45 points for 2023-2026), too few for deep learning; kept as
  possible extra inputs later.
- Older sales datasets (Rossmann 2013-15, Walmart M5 2011-16, Olist 2016-18): predate COVID, UPI and the AI boom.
- Open-startup revenue dashboards, TrustMRR: a few dozen companies, or terms that do not allow scraping.

---

## 4. Preprocessing and cleaning

### 4.1 Bank statements (in the product, every upload)
1. **Read any layout:** CSV or Excel; title rows above the table are skipped; the header row is detected; columns are
   recognised by role (date, narration, withdrawal/deposit or amount + Dr/Cr, balance), whatever the bank calls them.
2. **Normalise values:** Indian number formats (1,23,456.00), Dr/Cr signs, many date formats; rows with an unreadable
   date or amount are counted as skipped, never guessed.
3. **De-duplicate:** each line gets a fingerprint (date, amount, direction, narration, account), so uploading
   overlapping months never double-counts.
4. **Clean the narration for matching** (`backend/app/categorisation/text.py`): remove redaction marks, bracketed
   references, IFSC codes, UPI IDs, bank prefixes and long reference numbers; numbers become 0; drop bank-only tokens
   (UPI, NEFT, IMPS, RTGS, DR, CR, TRF, POS...); lower-case.
5. **Vendor key:** a stable short key per counterparty, so "remember this vendor" can match future lines.

### 4.2 Categoriser training data (IPSA)
1. Map IPSA's **344 cost types onto 14 startup expense categories** (`ml/categoriser/taxonomy.py`).
2. Clean descriptions with the **same text cleaner the product uses** (imported from the backend), so training and
   production can never clean text differently.
3. **Split by financial year, not randomly** (no future leaks into training): train 2010-11 to 2021-22, validation
   2022-23 to 2023-24, test 2024-25 to 2026-27. 41% of test rows use wording never seen in training.
4. **Balance:** cap 25,000 rows per category (326,256 training rows used of 782,429).
5. **Features:** TF-IDF word 1-2 grams + character 3-5 grams (handles typos and abbreviations) + amount bins.

### 4.3 Market data (Wikipedia page views)
1. **Canonical titles only:** every title is checked against Wikipedia; redirects and deleted articles are fixed
   (e.g. "Saree" redirects to "Sari" and would collect almost nothing).
2. **Views summed across renames:** an article's views are added up over its title and every redirect, so renamed
   articles keep their history (e.g. "Gemini (chatbot)" became "Google Gemini" in late 2025).
3. **Humans only:** the API's `user` agent filter excludes known bots.
4. **Share of attention:** divide by all human views on that Wikipedia that day (removes Wikipedia's overall decline).
5. **Scale per article:** divide by the article's own median, so big and small articles count equally; cap viral
   one-day spikes at the 99.5th percentile; drop articles with fewer than 20 views on a normal day.
6. **Sector index:** median of its articles (one viral article cannot move a sector).
7. **Chatbot correction:** divide each sector by the median of all 19 sectors (100 = a typical market). Topic pages
   lost readers faster than Wikipedia overall because people ask chatbots; this cancels the shared drop.
8. **Weekly:** mean of 7 days, weeks ending Sunday, partial weeks dropped.
9. **Sanity checks passed:** AI tools rises most (61 to 153), fashion peaks every Diwali, edtech peaks in
   exam-results season, edtech falls and travel cools after its 2023 peak.

---

## 5. Models, training and results

### 5.1 ML: expense categoriser (Built)

**Task:** bank line -> one of 14 expense categories, or "needs review".

**Six approaches compared** (validation picks the model; test is scored once):

| Model | Test accuracy | Test macro-F1 | Auto-filed | Precision when auto-filed |
| --- | --- | --- | --- | --- |
| Keyword rules (baseline) | 44.2% | 50.6% | 61.6% | 71.7% |
| Naive Bayes | 86.6% | 81.7% | 70.4% | 95.2% |
| Logistic regression | 90.8% | 85.9% | 94.7% | 93.5% |
| **Linear SVM (chosen)** | **91.2%** | **86.6%** | **95.3%** | **93.5%** |
| LightGBM | 89.5% | 84.0% | 92.6% | 93.4% |
| Neural network (MLP) | 88.5% | 82.7% | 91.2% | 93.5% |

- Chosen on validation macro-F1 (89.4%). On wording never seen in training: 85.6% accuracy.
- **Safety threshold:** chosen on validation so auto-filed lines are at least 95% precise; below it, the line goes to
  the founder with suggestions.
- **Unseen-words guard:** if a line has no word the model knows, it is never auto-filed, only suggested.
- **Indian narration check:** 611 of 638 purpose-stated Indian lines correct (95.8%); all 27 misses fell below the
  threshold and would have gone to review, not been filed wrongly.
- **End-to-end on the 18-statement demo pack:** 80% of lines filed on first upload, 96.8% of auto-filed lines
  correct; 100% filed automatically on a repeat month after learning.
- Weakest categories: meals (58% F1) and marketing (71%), often confused with travel and office supplies.

### 5.2 DL: market behaviour forecaster (Built)

**Task:** each sector's weekly market-interest index for the next 12 weeks, as P10 / P50 / P90.

**Why deep learning:** three years of weekly data is too short for statistics to learn seasonality on one series. A
single network trained on 189 related series (19 sectors + ~170 articles) learns shared patterns: a festive ramp-up
seen in fashion helps food. Rules cannot adapt per sector or give a range.

**Inputs:** last 52 weeks (scaled by their own level, log), chatbot-interest change, week of year, Diwali and Holi
weeks (known in advance), sector embedding. **Outputs:** 12 weeks x (P10, P50, P90), ordered by construction.

**Models:** MLP (flat view), 1D CNN / temporal convolution (local shapes), LSTM with attention (sequence memory;
attention shows which weeks mattered). **Loss:** pinball (quantile) loss. **Tuning:** dropout {0.1, 0.3} x epochs
{10, 20, 40, 60}, 3 seeds each; all chose dropout 0.3, 10 epochs. **Hardware:** RTX 2050, ~7 minutes.

**Split by time:** train Jan 2023 - May 2026 (175 weeks, 20,638 windows), validation May - Aug 2026, test Aug -
Oct 2026 (locked, scored once).

| Model | Validation MASE | Test MASE | 5-8 weeks ahead | P10-P90 coverage (target ~80%) |
| --- | --- | --- | --- | --- |
| **LSTM + attention** | 1.283 | **1.367** | **1.53** | 91% |
| Ensemble (mean of 3) | **1.281** | 1.385 | 1.54 | 90% |
| 1D CNN | 1.281 | 1.405 | 1.57 | 86% |
| MLP | 1.355 | 1.436 | 1.58 | 95% |
| Exponential smoothing (best baseline) | 1.562 | 1.477 | 1.80 | 69% |
| Naive | 1.846 | 1.502 | 1.85 | 70% |
| Seasonal naive | 2.402 | 1.934 | 2.03 | 56% |

(MASE = mean absolute error divided by the average week-to-week change; lower is better; multi-week forecasts
naturally exceed 1.)

- All three networks beat every baseline; LSTM 7.4% better than smoothing on test, 18% on validation.
- Ranges are honest: 86-95% coverage vs 69-70% for baselines (which were too narrow).
- Per sector the LSTM wins 10 of 19: biggest wins in festival-driven sectors (food and beverage 0.55 vs 1.26, food
  delivery 1.41 vs 2.92), losses in smooth-trend sectors (AI tools 2.43 vs 1.99, real estate 1.14 vs 0.79).

### 5.3 Deliberately not ML
P&L arithmetic, cash, runway, cap table and dilution: these must be exact, so they are calculations, not predictions.

---

## 6. How the models are used in the product

**Categoriser (every upload):**
1. Vendor memory first (the founder's own past decisions win).
2. Indian bank-format rules (structural lines: receipts, GST/TDS, EMIs, charges, cash, people).
3. ML model, only if confident and the words are known.
4. Otherwise "needs review" with the model's best guesses as one-click chips, plus a plain reason
   ("New vendor. Best guesses: ...").
Every line shows where its category came from (you, memory, rule, model) and why.

**Market forecaster (weekly):**
1. `python -m market.pageviews` extends the data; `market.index` rebuilds the index.
2. `python -m market.forecast` retrains the three networks on all weeks to date and writes each sector's 12-week
   forecast, smoothing's forecast and both models' held-out scores to `market_forecast.json`.
3. The backend serves, **per sector, whichever model was more accurate on the held-out weeks** (deep learning in
   12 sectors, exponential smoothing in 7) and says which one it used.
4. The startup's sector is suggested from its own description ("D2C beverages" -> Food and beverage) and can be
   changed; "you vs your market" compares the startup's monthly revenue with its market's index.

---

## 7. Every screen

| Screen | What it does |
| --- | --- |
| **Landing (`/welcome`)** | "Statement in, brief out." A real statement line flips into its category inside the headline; a live box where visitors type any statement line and see where it lands; how it works in 4 steps; the monthly brief; the bahi (traditional Indian account book) story; honest FAQ. |
| **Sign in / Create account** | Email and password; split screen with the product promise. |
| **Your startups** | Pick or add a company (one user can run several). |
| **Overview** | The monthly brief (sentence written from the ledger), primary action changes to "Review N transactions" when lines are unsorted, revenue hero tile with 6-month bars, EBITDA, gross margin (shows "-" with a fix when no cost of goods is filed), cash and runway, revenue/expenses/cash chart, where the money went, unit economics, month-close checklist, recent activity. |
| **Transactions** | Add bank account, upload statements, import summary, review inbox with suggestion chips, "remember this vendor", filters by month, category and text, provenance of every category. |
| **Financials** | Monthly P&L and cash flow table, runway tile, opening-balance prompt when a statement has no balance column, unit-economics form (physical: units, orders, stock; digital: customers, churn, MRR) with derived metrics. |
| **Cap table** | One-time setup, fully diluted ownership bar and table, rounds history, SAFEs, ESOP grants with vesting, add shares by hand, dilution simulator, record a round in one click, undo a round. |
| **Investors** | Current investors (from the cap table) and a fundraising pipeline by stage (contacted, meeting, due diligence, term sheet, closed, passed). |
| **Market outlook** | Written headline for the sector's next 12 weeks, chart (26 weeks of history, dashed likely path inside its range), next 4 / 12 weeks, peak week, track record, "you vs your market", how the forecast is made, sector picker. |

Design: one design system (`.interface-design/system.md`): indigo and emerald on lavender paper, Newsreader serif for
headlines and figures, Work Sans for the interface, light and dark themes, every text colour WCAG AA.

---

## 8. Rule-based engines

### 8.1 Indian bank-format rules (`backend/app/categorisation/formats.py`) (Built)
Applied before the model, in order:
- **Failed / reversed lines** -> excluded.
- **Refunds** -> refund (cash, not revenue).
- **Tax:** GST payment, CPIN, TDS payment/challan -> tax (cash flow only). "GST filing" fees are not tax.
- **Loans:** EMI / loan -> loan repayment (cash flow only).
- **Bank charges**, **cash withdrawals**, **interest / other income**.
- **Financing:** equity or loan money coming in -> financing (cash in, not revenue).
- **Payment-gateway settlements** (Razorpay, Cashfree, PayU, Stripe...) -> revenue.
- **Payments to people:** a phone-number UPI ID, or a bare name after NEFT/IMPS with no business word -> "payment to a
  person" (asks: salary? freelancer?). A business UPI handle or a business/supplier word (Pvt, Ltd, traders, supplies,
  packaging, works, known brands...) skips this rule.

### 8.2 Accounting rules (`backend/app/finance/pnl.py`) (Built)
- Revenue = customer receipts; other income separate.
- **Not in the P&L (cash flow only):** GST/TDS, loan principal, money raised, cash withdrawals, refunds.
- Cost of goods is a category the founder assigns (vendor memory repeats it); gross profit = revenue - COGS.
- Unreviewed payments to people count as people costs; unrecognised money in is not counted as revenue until confirmed.
- EBITDA = gross profit - operating expenses.
- Closing cash = last balance per account per month, else opening balance + flows.
- Net burn = cash out - (cash in - money raised). Runway = closing cash / average burn of the last 3 months.

### 8.3 Unit economics (`backend/app/finance/units.py`) (Built)
- Physical: average order value, revenue per unit, cost of goods per unit, months of stock.
- Digital: churn, ARPU, CAC (marketing spend / new customers), LTV (ARPU x gross margin / churn), LTV:CAC, payback.

### 8.4 Cap-table maths (`backend/app/finance/equity.py`) (Built)
- Fully diluted = issued shares + granted options + unallocated pool.
- Pre-money method; pool top-up and SAFE conversion come out of the pre-money (Indian term-sheet convention).
- SAFE converts at the lower of the cap price (cap / pre-money shares) and the discounted round price.
- Price per share and pool top-up solved by fixed-point iteration so the post-round pool hits its target exactly.
- Vesting: monthly after the cliff (default 4 years, 1-year cliff). Grants cannot exceed the pool.

### 8.5 Validation rules (Built)
Future months rejected; cap table set up once; grants larger than the pool rejected; dates cannot be in the future;
unknown market sectors rejected; uploads over 10 MB rejected.

### 8.6 Health checks and recommendations (Planned, Phase 6)
Runway under 6 months, burn multiple over 2, gross margin falling 3 months running, LTV:CAC under 3, monthly churn
over 3%; each with the numbers behind it. Plus anomaly alerts (Isolation Forest + out-of-range checks).

---

## 9. Security (this is private financial data)

### Built
- **Passwords:** argon2 hashing (memory-hard); never stored or logged in plain text; re-hashed when parameters change.
- **Sessions:** 15-minute JWT access token kept only in browser memory (not localStorage); 30-day refresh token in an
  httpOnly, SameSite=Strict cookie scoped to the auth path; stored server-side only as a SHA-256 hash.
- **Refresh rotation with reuse detection:** every refresh issues a new token; if an old one is reused (a sign of
  theft), the whole token family is revoked.
- **Tenant isolation:** every startup endpoint goes through one access check (`app/api/deps.py`); non-members get
  404 (not 403), so other startups' existence is never revealed. Tested.
- **Roles:** founder role today; the data model (memberships) is ready for team members and read-only investors.
- **Audit log:** every change (imports, categorisations, cap-table edits, rounds) records who, what and when.
- **Minimal bank data:** only the last 4 digits of an account number are stored, never the full number; the uploaded
  file itself is not kept, only the parsed lines and import counts.
- **Input limits:** 10 MB upload cap, validated schemas (Pydantic) on every request, length limits on text fields.
- **Production guard:** the app refuses to start in production without a random 32+ character secret.
- **CORS** restricted to configured origins; HTTPS-only cookies when deployed behind HTTPS.
- **Secrets and data out of git:** no .env files, datasets, statements or model files are committed.

### Planned (Phase 7 hardening)
- Rate limiting on sign-in and uploads; account lockout after repeated failures.
- Encryption at rest (managed PostgreSQL with disk encryption) and encrypted backups.
- Data export and account/startup deletion on request.
- Security review, dependency scanning, error tracking without personal data in logs.
- Live bank connection only through India's Account Aggregator framework (consent-based, regulated), never by
  collecting net-banking passwords.

---

## 10. Ethics and responsible AI

- **No user data trains the models.** The categoriser is trained on public IPSA data; the market model on public
  Wikipedia page views. A startup's corrections only teach *its own* vendor memory, never a shared model.
- **Human in the loop:** the model never files a line it is unsure about or whose words it has never seen; the
  founder decides, and one correction can be applied to the vendor's other lines.
- **Explainability:** every category shows its source (you, memory, rule, model) and a plain reason; forecasts show
  which model was used and its track record against the alternative.
- **Honest uncertainty:** forecasts are ranges (P10-P90), not single numbers; provisional figures are labelled while
  lines are unsorted; implausible numbers (e.g. 100% gross margin with no cost of goods) are shown as "-" with a fix.
- **Honest claims:** the demo data is always described as synthetic; the market index is labelled as interest
  (attention on Wikipedia), not sales; accuracy claims state what they were measured on.
- **Licences respected and credited:** OGL, Apache-2.0, MIT, CC0; landing-page pictures credited (public domain, CC0).
  Google Trends was rejected rather than scraped against its terms.
- **Privacy by design:** minimal data (last 4 digits, no stored files), access only for members, audit trail,
  designed to align with India's Digital Personal Data Protection Act, 2023 (consent, purpose limitation, deletion);
  formal compliance work is part of Phase 7.
- **Future agents** (competitor watch, customer voice): public information only, official APIs, every claim linked
  to its source, no storing of reviewers' personal data.
- **Limits stated openly:** categoriser trained on UK expense wording (Indian vendors learned per startup); market
  model has about three yearly cycles of data; Wikipedia readers lean English-speaking and urban.

---

## 11. Roadmap

| Phase | Status |
| --- | --- |
| 1. Foundation: auth, startups, memberships, audit | Built |
| 2. Transactions and ML categorisation | Built |
| 3. Financials and dashboard | Built |
| 4. Cap table and investors | Built |
| 5. Market behaviour forecasting (DL) + Market outlook tab | Built |
| 6. Insights: anomaly alerts, health checks, investor update drafts, month close | Planned |
| 7. Hardening: rate limits, encryption at rest, deletion, deployment, backups | Planned |
| 8. Roles: team members, read-only investor portal | Planned |
| Future: competitor watch and customer voice agents; Account Aggregator bank connection | Planned |

---

## 12. Suggested slide order

1. The problem (spreadsheets, late P&L, runway guessed)
2. StartupIQ in one line + the monthly brief screenshot
3. Whole-system workflow (section 2.1 diagram)
4. Architecture (section 2.2)
5. Datasets and why (section 3) + what was ruled out
6. Cleaning and preprocessing (section 4)
7. ML categoriser: 6 models, results table, safety threshold, Indian check
8. DL market forecaster: why DL, inputs/outputs, 3 architectures, results table, per-sector wins
9. How models are used in the product (section 6)
10. Screens tour (Overview, Transactions, Financials, Cap table, Investors, Market outlook)
11. Rule-based engines (bank formats, accounting, cap-table maths)
12. Security (built vs planned)
13. Ethics and responsible AI
14. Limits and roadmap
15. Live demo: upload a statement, review a line, open Overview, Cap table simulator, Market outlook
