# Data card: market signals (FRED and Wikipedia)

| Item | Value |
| --- | --- |
| Used for | Market pulse (M5): is the category growing, flat or falling |
| Sources | FRED (St. Louis Fed, from US Census): RSXFS (retail ex food services), RSCCAS (clothing stores), MRTSSM4541USN (electronic shopping, not seasonally adjusted). Wikimedia page-views API: Coffee, Smartphone, Wireless_earbuds |
| Licence | Public data; cite FRED/Census and Wikimedia |
| Unit of a row | One series in one month |
| Coverage | FRED monthly from 1992 (about 417 rows each); Wikipedia monthly from 2015-07 (Wireless_earbuds from 2022-09) |

## Cleaning (`python -m startupiq.clean signals`)

Non-numeric FRED values dropped (none in the current files). A trailing Wikipedia month that the API returns when it has barely started is dropped. Output: `data/processed/signals_monthly.parquet` with columns month, value, source, series.

## Known limits

- US data, while the target users may be Indian sellers. A real deployment would need local category data.
- Page views measure interest, not purchases, and move differently from sales; treat as a weak, optional input.
- Online retail grew 81% from 2019 to 2023 versus 32% for all retail: a category's trend can differ a lot from the overall market.
