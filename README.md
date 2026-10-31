# Startup IQ

The founder's monthly operating system: bank transactions categorised into a P&L automatically, monthly financials
(revenue, gross profit, EBITDA, burn, runway), cap table and investors in one place, revenue and cash forecasts with a
range, and alerts when something looks wrong. Product spec: [docs/PRD.md](docs/PRD.md).

Earlier versions of the project live on the `legacy` branch.

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 Foundation | Accounts, sign-in, startups, membership-based access, audit log, Docker, CI | Done |
| 2 Transactions + ML | Statement import, automatic categorisation, review inbox | Next |
| 3-8 | Financials, equity, forecasting, insights, hardening, roles | Planned |

## Layout

```
backend/    FastAPI + SQLAlchemy + Alembic (Python 3.12+)
  app/core/       settings, database, password hashing and tokens
  app/models/     users, refresh tokens, startups, memberships, audit log
  app/api/        routes, and the access check every startup endpoint uses (deps.py)
  alembic/        database migrations
  tests/          API, security and migration tests
frontend/   React + TypeScript + Vite
docs/       product spec
```

## Run with Docker

```bash
cp .env.example .env          # then set SIQ_SECRET_KEY
docker compose up --build     # app on http://localhost:8080, API docs on http://localhost:8000/api/docs
```

## Run without Docker (development)

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

## Checks

```bash
cd backend && ruff check . && pytest        # set SIQ_TEST_DATABASE_URL to a PostgreSQL URL to test on PostgreSQL
cd frontend && npm run lint && npm run build
```

CI runs the same checks on every push, with the backend tests on PostgreSQL.

## Security notes

- Passwords hashed with argon2; an unknown email takes as long to reject as a wrong password.
- Access tokens last 15 minutes and live only in browser memory. Refresh tokens are httpOnly, SameSite=Strict cookies,
  stored server-side only as SHA-256 hashes and rotated on every use; reusing a rotated token ends the whole session.
- Every startup endpoint checks membership; non-members get 404, so other startups' existence is not revealed.
- Sign-ups, logins, failed logins and every startup change are written to the audit log.
- In production the API refuses to start without a real `SIQ_SECRET_KEY`.
