# SEC Form D: Phase 1 intake report

Source: SEC Form D Data Sets, quarterly ZIPs `<quarter>_d.zip` (URL pattern confirmed working). Downloaded 2026-10-13 for 2023 Q1 to 2024 Q4 (8 files). Sizes and SHA-256 are in `data/raw/MANIFEST.md`. All numbers below were computed from the downloaded files. Each ZIP also contains the official `FormD_readme.html` and `FormD_metadata.json`, which are the authority for field meanings and should be read before Phase 7.

## Tables and columns (official, from the files)

FORMDSUBMISSION, ISSUERS, OFFERING, RECIPIENTS, RELATEDPERSONS, SIGNATURES, joined by `ACCESSIONNUMBER`.

- FORMDSUBMISSION: FILING_DATE, SIC_CODE, SUBMISSIONTYPE (D or D/A), and others.
- ISSUERS: IS_PRIMARYISSUER_FLAG, CIK, ENTITYNAME, STATEORCOUNTRY, ENTITYTYPE, YEAROFINC_TIMESPAN_CHOICE, YEAROFINC_VALUE_ENTERED, and others.
- OFFERING: INDUSTRYGROUPTYPE, INVESTMENTFUNDTYPE, REVENUERANGE, FEDERALEXEMPTIONS_ITEMS_LIST, ISAMENDMENT, PREVIOUSACCESSIONNUMBER, SALE_DATE, security-type flags, TOTALOFFERINGAMOUNT, TOTALAMOUNTSOLD, TOTALNUMBERALREADYINVESTED, HASNONACCREDITEDINVESTORS, and others.
- There is no business-description text. Company name, SIC code and industry group are the only descriptive fields.

## Differences from the proposal (which relied on a third-party field reference)

| Proposal said | Found |
| --- | --- |
| Indefinite offering amount encoded as -1 | Encoded as the string **"Indefinite"** (49,318 rows) |
| Year of incorporation has "within five years" and "yet to be formed" fields | Two columns: YEAROFINC_TIMESPAN_CHOICE (withinFiveYears, overFiveYears, yetToBeFormed) and YEAROFINC_VALUE_ENTERED (blank in about 25% of issuer rows) |
| Industry group lists "Technology" | Values are finer: Other Technology, Computers, Biotechnology, Other Health Care, and so on (35 values) |

## Volume and scope (8 quarters)

| Step | Filings |
| --- | --- |
| All submissions (accession numbers unique) | 107,709 |
| of which amendments (D/A) | 43,115 (40%); every one links back through PREVIOUSACCESSIONNUMBER |
| Pooled investment funds (by industry group) | 64,160 (60%) |
| Non-fund | 40,192 |
| Non-fund, initial filings only (type D) | 32,206 |
| ...with a positive numeric offering amount | **29,338**, from 24,676 distinct issuers (2,971 filed more than once) |
| Technology, health and business-services groups within that | 10,321 |

Largest non-fund industry groups (initial, numeric amount): Other 6,851; Other Technology 6,419; Commercial 3,034; Other Real Estate 1,893; Other Health Care 1,728; Residential 1,632; Biotechnology 1,311. Many of these are not startups (real estate, oil and gas). Whether enough filings are real startups is still **unknown**.

## Target quality (TOTALOFFERINGAMOUNT)

- "Indefinite": 2,843 among initial non-fund filings. Treat as missing target.
- Zero: 33 across all filings.
- Distribution across all positive amounts: 10th percentile $125,000; median $3,000,000; 90th percentile $75,000,000; 99th percentile $1 billion; maximum about $1 trillion. **Non-fund maximums reach $500 billion**, which are almost certainly entry errors or placeholders. Cap or exclude in modelling, work in log scale.
- Median by industry group (initial, non-fund): Other Technology $3.5M; Biotechnology $5.8M; Other Health Care $3.0M; Business Services $1.5M; Other $2.0M.
- TOTALAMOUNTSOLD is zero in 18% of all filings (early filings report partial sales), 7% in the technology, health and services subset. Amount sold exceeds amount offered in 0 filings. Offering amount remains the better target.

## Feature quality

- REVENUERANGE: "Decline to Disclose" is 88% of the technology, health and services subset (9,047 of 10,321). Expect this feature to carry little signal. It is blank for most fund filings.
- Year of incorporation value is missing for 3,844 of 10,321 in that subset (37%); the timespan choice is always present and is the safer age proxy.
- STATEORCOUNTRY includes non-state codes (for example "E9", 6,087 rows); a code lookup is needed.
- Entity types: Limited Partnership 43,995; LLC 33,632; Corporation 20,741; Other 11,255 (issuer rows).
- SIC_CODE sits in the submission table; fill rate not yet checked. It may help the sector crosswalk.

## Decisions taken from this report

1. Use initial filings (type D); keep amendments only to correct or link to their original.
2. Exclude pooled investment funds.
3. Treat "Indefinite" as a missing target; cap or drop extreme amounts and model log amounts.
4. Primary target stays TOTALOFFERINGAMOUNT.
5. Use YEAROFINC_TIMESPAN_CHOICE as the main age proxy; use the year value only where present.
6. Expect REVENUERANGE to be a weak feature; keep it for the ablation.
7. Group splits by issuer CIK; temporal split by filing date.
8. The funding benchmark should be limited to a defined startup-like subset; defining it is the first job of Phase 7.

## Still open

Licence wording (the files carry the SEC's disclaimer; confirm before commercial use). Whether the startup-like subset really contains startups. Coverage before 2023 (only 2023 Q1 to 2024 Q4 downloaded). Fill rate and usefulness of SIC_CODE.
