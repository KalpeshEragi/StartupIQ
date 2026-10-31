# 0011: Intake design (reading any sales upload)

- Status: Accepted
- Date: 2026-10-24
- Module / task: M0; Tasks 4, 5, 10

## Problem

Founders upload whatever they have: Excel with title rows, CSVs with ₹ and lakh commas, no headers, Hindi headers, one column per month, several sheets, US or Indian dates. Intake must find the date and amount, build a regular series, and say honestly what it is unsure of.

## Options considered

| Option | For | Against |
| --- | --- | --- |
| Fixed template only | Zero ambiguity | Most users will not reformat; high drop-off |
| **Rules: name synonyms + content checks + confidence** | Predictable, explainable ("we picked 'Bill Date' because 100% of values parse as dates and the name contains 'date'"); testable file by file | Synonym lists need upkeep |
| Learned column classifier (e.g. gradient boosting on column statistics) | Could generalise to unseen names | Needs hundreds of labelled real uploads we do not have; errors are silent and hard to explain |
| LLM column mapping | Handles odd names | Non-deterministic, needs network and a paid API, hard to test, can invent mappings |

## Decision

Rules, with the template as a fallback and a confirmation step. The template (`configs/intake_template.csv`) is offered whenever a file is refused.

## How it works

1. **Read** (`reader.py`): encodings utf-8, utf-8-sig, cp1252, latin-1; delimiter sniffing; blank rows and columns removed; header row = first wide row within 15 whose cells are mostly labels (text or period names), else the file has no header; "Total" rows removed; every sheet is tried and the one with the best tier wins.
2. **Layout** (`layout.py`): no date column but 3+ period headers -> periods across, melted. Date column but no confidently named amount and 3+ unnamed numeric columns -> products across, melted, and the user is asked to confirm.
3. **Roles** (`columns.py`): date needs at least 80% of values to parse; the name breaks ties, with "order/purchase/invoice/bill" preferred and "ship/deliver/estimated/due" avoided. Money and quantity roles need a name match of at least 0.5 after penalties ("No", "ID", "GST", "freight", ...) and at least 80% numeric content. ID-like (unique, increasing integers) and year-like columns are never amounts. If nothing is named, the numeric column with the largest typical value is guessed as revenue with confidence 0.5. User overrides always win.
4. **Amount**: revenue column; else quantity x price; else quantity (forecast in units); else price per row (marketplace exports, asks to confirm).
5. **Series** (`series.py`): dates parsed by rule (ISO, day-first, month-first, Excel serial, month-only); if every value fits both day orders we read day-first and ask. Future and pre-1990 rows removed. Duplicates judged on the **whole uploaded row**: kept in a shop log without times (repeat purchases are normal), removed when rows carry a time stamp or one row per period is expected. Negatives netted within the period. Partial first and last weeks dropped. Weekly series, or monthly if the upload is monthly. Top 5 products get their own weekly series.
6. **Checks and tier**: refuse below 20 rows (daily or transactions), fewer than 4 periods with sales, or all-zero sales. Ask about 3+ consecutive empty periods inside the history (closed or not recorded?) and about periods with robust z above 4 on log sales (one-off bulk order?). Cap at Basic when more than half the periods are empty. Tier thresholds: decision 0004.

## Confidence and confirmation

Below 0.75 the UI asks the user. Questions currently produced: `revenue`/`date`/`product` (guessed column), `date_order` (ambiguous day/month), `gaps`, `outliers`, `layout` (products-as-columns reshape).

## Evidence

See `../evaluation_log.md` (2026-10-24 intake rows): 25 of 25 pack files handled as expected; real Rossmann, UCI and Olist exports mapped correctly. The pack and real exports found five bugs, all fixed before acceptance: identical shop-log rows dropped as duplicates; wide monthly sheets read as sparse weekly; products-as-columns not reshaped; Olist shipping date and freight picked over purchase date and price; marketplace rows that differ only in item number treated as duplicates.

## Known limits

- Fees and postage lines are not recognised in unknown files (UCI weekly totals differ by 3.3% from the dataset-specific cleaner, which removes them).
- Synonyms cover English and common Hindi words only.
- Product series are built for weekly uploads only.
