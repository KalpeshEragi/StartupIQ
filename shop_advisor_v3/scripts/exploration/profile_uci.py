"""Exploration script (2026-10-23/24). Run from the project root. Numbers are logged in devdocs/evaluation_log.md."""
import zipfile, sys, time
import pandas as pd, numpy as np

interim = "data/interim/uci"
t = time.time()
zipfile.ZipFile("data/raw/uci_retail/online_retail_ii.zip").extractall(interim)
sheets = pd.read_excel(interim + "/online_retail_II.xlsx", sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)
print("read secs", round(time.time() - t))
print(df.shape); print(df.dtypes)
print(df.isna().mean().round(3))
print("date range", df.InvoiceDate.min(), df.InvoiceDate.max())
print("cancel invoices", df.Invoice.astype(str).str.startswith("C").mean().round(3))
print("neg qty", (df.Quantity < 0).mean().round(3), "price<=0", (df.Price <= 0).mean().round(3))
print("dups", df.duplicated().mean().round(3))
print("customers", df["Customer ID"].nunique(), "products", df.StockCode.nunique(), "countries", df.Country.nunique())
d = df[(df.Quantity > 0) & (df.Price > 0)].copy()
d["rev"] = d.Quantity * d.Price
d["day"] = d.InvoiceDate.dt.normalize()
daily = d.groupby("day").rev.sum()
print("days with sales", len(daily), "of", (daily.index.max() - daily.index.min()).days + 1)
wk = d.groupby(pd.Grouper(key="InvoiceDate", freq="W")).rev.sum()
print("weeks", len(wk), "zero weeks", (wk == 0).sum())
print(wk.describe().round(0))
# series available for a global model: product-weekly
pw = d.groupby(["StockCode", pd.Grouper(key="InvoiceDate", freq="W")]).Quantity.sum().reset_index()
cnt = pw.groupby("StockCode").size()
print("products with >=52 active weeks", (cnt >= 52).sum(), ">=78", (cnt >= 78).sum(), ">=26", (cnt >= 26).sum())
# competitor/share structure? none: single retailer
print("top-10 product revenue share", (d.groupby("StockCode").rev.sum().nlargest(10).sum() / d.rev.sum()).round(3))
print("country revenue share top3", (d.groupby("Country").rev.sum().nlargest(3) / d.rev.sum()).round(3).to_dict())

# quick forecastability: weekly total revenue, last 12 weeks holdout
y = wk.copy()
y = y[y.index < y.index.max()]  # drop partial last week
train, test = y[:-12], y[-12:]
snaive = pd.Series(y.shift(52).values[-12:], index=test.index)
naive = pd.Series(train.iloc[-1], index=test.index)
ma = pd.Series(train.iloc[-4:].mean(), index=test.index)
def mape(a, f):
    m = a.notna() & f.notna()
    return float((abs(a[m] - f[m]) / a[m]).mean())
print("weekly total revenue holdout 12w MAPE: naive %.3f, ma4 %.3f, seasonal-naive %.3f" % (mape(test, naive), mape(test, ma), mape(test, snaive)))
# product-level: how noisy is a typical product?
top = d.groupby("StockCode").rev.sum().nlargest(200).index
ps = pw[pw.StockCode.isin(top)].pivot(index="InvoiceDate", columns="StockCode", values="Quantity").fillna(0)
cv = (ps.std() / ps.mean()).median()
print("median coefficient of variation, top-200 products weekly qty", round(float(cv), 2))
print("share of zero weeks, top-200 products", round(float((ps == 0).mean().mean()), 3))
