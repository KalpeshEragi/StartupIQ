# Task 4 — EDA findings (actual data)

Figures in `Task4/outputs/figures/`, tables in `Task4/outputs/`.

- **Target (`status`):** operating 41,827 · acquired 3,692 · closed 2,603 · missing 1,314. There is no IPO class. Only acquired + closed (6,295) are resolved outcomes; `operating` is unresolved (right-censored).
- **Missingness:** `state_code` 44% (only recorded for USA and Canada, so structural), founded date 29%, country 19%, category/market 16%, funding total 17% (`-`).
- **Funding:** extremely right-skewed (skew 142 raw → -0.27 after log1p). Median $2.0M, p99 $200M, max $30B.
- **Outcome vs funding:** median total funding is $8.4M for acquired vs $1.0M for closed. 80% of acquired companies have venture funding vs 54% of closed; seed-only history is far more common among closed (36% vs 18%).
- **Rounds:** acquired share rises from 49% (1 round) to 80%+ (3+ rounds). Round count is strongly predictive but is future information (see Task 5).
- **Founding year:** acquired share among resolved companies falls from ~85% (1990s cohorts) to ~45% (2010s). This is a cohort/observation-window effect, not a startup quality signal. New startups will fall outside this range.
- **Market/country:** acquired share varies by market (Enterprise Software 80%, Semiconductors 70%, Social Media 41%, Clean Tech 40%; overall ~59%). Country effects exist but sample sizes outside the US are small.
- **Dates:** first funding happens before the recorded founding date for 2,739 companies (493 by more than a year); founding dates are often the 1st of the month/year, so they are imprecise.
