# Task 4: Data cleaning and exploration (EDA)

All four images are in this folder. Code: `src/startupiq/clean/` (cleaning rules), `src/startupiq/pipelines/task4.py` (runs it), `src/startupiq/pipelines/output_figures.py` (draws these images). Full step-by-step logs: `docs/results/task4/cleaning_log_yc.json` and `cleaning_log_formd.json`.

## What Task 4 is, in one line
Take two raw public datasets (Y Combinator company descriptions, SEC Form D funding filings), find what is wrong with them, fix or flag each problem with a logged rule, and look at the cleaned data before any modelling.

## Why these two datasets
YC gives company descriptions (text) to learn what a good startup description contains and which gaps are common. Form D gives how much companies raised, for the funding benchmark. Neither is merged with the other; they serve different modules.

---

## Image 1: `1_cleaning_pipeline_funnel.png` (rows kept at each cleaning step)
**What it shows.** Left: YC stays at 6,288 rows because nothing was deleted; 6,233 have usable text and 6,215 also have a known industry. Right: Form D goes 107,709 filings, then 64,594 after removing amendments, then 32,206 after removing investment funds, then 10,321 start-up-like filings with a usable amount.
**Why we did it.** Amendments (D/A) restate an earlier filing, so keeping them counts one raise twice. Pooled investment funds raise money to invest, not to run a business, so they are not startups.
**What to say to the mentor.** "Cleaning removed 70% of Form D on purpose, with a reason for each cut. YC was already clean, so we flagged problems instead of deleting rows."
**Likely question.** *Why only 10,321 at the end?* Only six industry groups can contain operating startups (technology, computers, biotech, pharma, other health care, business services), and many filings say the offering amount is "Indefinite".

## Image 2: `2_problems_found_and_fixes.png` (data issues found and handled)
**What it shows.** Counts of each problem. YC: 370 empty long descriptions, 137 team sizes of 0 or -1, 55 rows with no usable text, 18 industries marked "Unspecified", 199 launch years far from the batch year, 863 with no tags. Form D: 25,713 filings decline to disclose revenue, 8,233 have no issuer age, 2,843 offering amounts are "Indefinite", 25 amounts are zero or negative, 6 are $10 billion or more.
**What we did about each.** Empty long description: use the one-liner instead. Impossible team size: set to missing and flag. No usable text: flag, excluded from text models. "Unspecified" industry: set to missing. Year mismatch and no tags: kept (tags are never a model input). "Decline to Disclose": kept as its own category, because refusing to disclose is information. "Indefinite" and zero/negative amounts: set to missing. $10bn or more: flagged and left out of amount models.
**What to say.** "Every issue has a rule, a count, and a logged result. Nothing is silently deleted or silently invented. We never filled in a missing value with a guess."
**Likely question.** *Why not fill missing values?* Our project rule is never to invent startup information. Missing stays missing, with an indicator column when it might matter.

## Image 3: `3_yc_data_shape.png` (YC dataset overview)
**What it shows.** Left: companies per industry. B2B has 3,202 of 6,288 (51%); Consumer 887; Healthcare 703; Fintech 663; Industrials 484; Real Estate and Construction 163; Education 125; Government 43; Unspecified 18. Right: description length in words; the median is 62 words, with a long tail.
**Why it matters.** Because B2B is half the data, a model that always says B2B would already score 51% accuracy. That is why we report macro-F1 (average over classes) and compare against a baseline. Short descriptions mean many pitches simply do not say much, which later became a feature for the readiness check.
**What to say.** "Class imbalance is the main reason I do not trust accuracy alone. Text length is short, which limits what any model can learn."

## Image 4: `4_formd_amount_raw_vs_log.png` (offering amount distribution)
**What it shows.** Left: raw offering amount in millions of dollars (capped at 200M for display): one tall spike near zero and a long tail. Right: the same amounts on a log10 scale, which spread out into a clear bell shape. For start-up-like filings with a usable amount, the median is about $3,500,000.
**Why we did it.** Raise sizes run from thousands to billions of dollars. On the raw scale a few huge filings dominate every error. On the log scale, "twice as big" is the same distance everywhere, which is how funding sizes behave.
**What to say.** "I model the log of the amount, not the raw amount, because the raw data is extremely skewed."

---

## Honest limits to mention
- Issuer age only exists for companies incorporated within the last five years (the chart for it shows values 0 to 5), so 25.6% of ages are missing. This explanation is my reading of the data and is not confirmed against SEC documentation.
- Zero duplicates were found by id, by website domain, and by near-identical text. We checked; nothing needed removing.
- YC's licence is not confirmed, so this is for experiments until we do.
