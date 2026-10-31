"""Exploration script (2026-10-23/24). Run from the project root. Numbers are logged in devdocs/evaluation_log.md."""
import pandas as pd, numpy as np
root = "data/raw"

print("######## ROSSMANN (HF mirror AiiN-aini)")
tr = pd.read_csv(root + r"\rossmann_hf\train.csv", low_memory=False, parse_dates=["Date"])
st = pd.read_csv(root + r"\rossmann_hf\store.csv")
print(tr.shape, tr.Date.min().date(), tr.Date.max().date(), "stores", tr.Store.nunique())
print(tr.dtypes.to_dict())
print("missing:", tr.isna().sum().sum(), "| dup (Store,Date):", tr.duplicated(["Store", "Date"]).sum())
print("StateHoliday values:", tr.StateHoliday.astype(str).value_counts().to_dict())
print("Open=0 share", round((tr.Open == 0).mean(), 3), "| Open=1 but Sales=0:", ((tr.Open == 1) & (tr.Sales == 0)).sum())
o = tr[(tr.Open == 1) & (tr.Sales > 0)]
print("sales open days: mean %.0f median %.0f p99 %.0f max %.0f" % (o.Sales.mean(), o.Sales.median(), o.Sales.quantile(.99), o.Sales.max()))
print("promo lift (mean sales promo vs not):", round(o[o.Promo == 1].Sales.mean() / o[o.Promo == 0].Sales.mean(), 3))
print("store.csv missing:", st.isna().sum()[st.isna().sum() > 0].to_dict())
print("StoreType", st.StoreType.value_counts().to_dict(), "| CompetitionDistance median", st.CompetitionDistance.median())
j = o.merge(st[["Store", "CompetitionDistance"]], on="Store")
j["near"] = j.CompetitionDistance < j.CompetitionDistance.median()
print("mean sales, rival closer than median vs farther:", round(j[j.near].Sales.mean()), round(j[~j.near].Sales.mean()))
print("days per store (min/median)", tr.groupby("Store").size().min(), tr.groupby("Store").size().median())

# baseline backtest: last 42 days held out; metric RMSPE on open days with sales>0
last = tr.Date.max()
cut = last - pd.Timedelta(days=41)
piv = tr.pivot(index="Date", columns="Store", values="Sales")
opn = tr.pivot(index="Date", columns="Store", values="Open")
test_idx = piv.index[piv.index >= cut]
def rmspe(act, pred):
    m = (act > 0) & pred.notna()
    a, p = act[m].values, pred[m].values
    return float(np.sqrt(np.mean(((a - p) / a) ** 2)))
act = piv.loc[test_idx]
# 1) last-year same day (364 days earlier)
ly = piv.shift(364).loc[test_idx]
# 2) same weekday mean of 4 weeks before cut
hist = piv[piv.index < cut]
pred_wd = pd.DataFrame(index=test_idx, columns=piv.columns, dtype=float)
for d in test_idx:
    wd = hist[hist.index.dayofweek == d.dayofweek].tail(4)
    pred_wd.loc[d] = wd.replace(0, np.nan).mean().values
# 3) store overall mean of open days
pred_mean = pd.DataFrame(np.tile(hist.replace(0, np.nan).mean().values, (len(test_idx), 1)), index=test_idx, columns=piv.columns)
print("42-day holdout RMSPE (open days) | store-mean %.3f | same-weekday-4w %.3f | same-day-last-year %.3f" % (
    rmspe(act, pred_mean), rmspe(act, pred_wd), rmspe(act, ly)))

print()
print("######## OLIST (HF mirror miminmoons, original CC BY-NC-SA 4.0)")
od = pd.read_csv(root + r"\olist_hf\olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
it = pd.read_csv(root + r"\olist_hf\olist_order_items_dataset.csv")
pr = pd.read_csv(root + r"\olist_hf\olist_products_dataset.csv")
tl = pd.read_csv(root + r"\olist_hf\product_category_name_translation.csv")
sl = pd.read_csv(root + r"\olist_hf\olist_sellers_dataset.csv")
rv = pd.read_csv(root + r"\olist_hf\olist_order_reviews_dataset.csv")
print("orders", od.shape, od.order_purchase_timestamp.min().date(), od.order_purchase_timestamp.max().date())
print("status", od.order_status.value_counts().to_dict())
print("items", it.shape, "sellers", it.seller_id.nunique(), "products", it.product_id.nunique(), "| reviews", rv.shape,
      "| review text present", round(rv.review_comment_message.notna().mean(), 3))
d = it.merge(od[["order_id", "order_purchase_timestamp", "order_status"]], on="order_id")
d = d[d.order_status.isin(["delivered", "shipped", "invoiced", "approved", "processing"])]
d = d.merge(pr[["product_id", "product_category_name"]], on="product_id", how="left")
d = d.merge(tl, on="product_category_name", how="left")
d["cat"] = d.product_category_name_english.fillna(d.product_category_name)
print("items with no category", round(d.cat.isna().mean(), 3))
d["month"] = d.order_purchase_timestamp.dt.to_period("M")
d = d[(d.month >= "2017-01") & (d.month <= "2018-08")]  # drop sparse 2016 and truncated tail
print("months used", d.month.nunique(), "rows", len(d))
m = d.groupby(["cat", "month", "seller_id"]).agg(rev=("price", "sum"), n=("order_id", "nunique"), price=("price", "median")).reset_index()
sellers_per = m.groupby(["cat", "month"]).seller_id.nunique().reset_index(name="k")
print("category-months", len(sellers_per), "with >=5 sellers", (sellers_per.k >= 5).sum(), "with >=10", (sellers_per.k >= 10).sum())
cats_ok = sellers_per.groupby("cat").k.median()
print("categories total", cats_ok.size, "| median sellers/month >=5:", (cats_ok >= 5).sum(), ">=10:", (cats_ok >= 10).sum(), ">=20:", (cats_ok >= 20).sum())
print("top categories by median sellers:", cats_ok.sort_values(ascending=False).head(6).round(0).to_dict())
tot = d.groupby("seller_id").price.sum().sort_values(ascending=False)
print("seller revenue: top 1%% = %.2f of revenue, top 10%% = %.2f; sellers with <10 orders overall: %.2f" % (
    tot.head(int(len(tot) * .01)).sum() / tot.sum(), tot.head(int(len(tot) * .1)).sum() / tot.sum(),
    (d.groupby("seller_id").order_id.nunique() < 10).mean()))
# seller-category panel continuity
act = m.groupby(["cat", "seller_id"]).month.nunique()
print("seller-category pairs", len(act), "active >=6 months:", (act >= 6).sum(), ">=10:", (act >= 10).sum())
# can we label competitor moves? price cut / expansion / exit at seller-category-month level for pairs active >=6 months
full = m.set_index(["cat", "seller_id", "month"])
lab = []
for (c, s), g in m.groupby(["cat", "seller_id"]):
    if len(g) < 6: continue
    g = g.set_index("month").reindex(pd.period_range("2017-01", "2018-08", freq="M"))
    for t in range(2, len(g) - 1):
        cur, nxt = g.iloc[t], g.iloc[t + 1]
        if pd.isna(cur.rev): continue
        if pd.isna(nxt.rev): y = "no_orders_next"
        elif nxt.price < cur.price * 0.95: y = "price_cut"
        elif nxt.n > cur.n * 1.3 and nxt.n >= 3: y = "expansion"
        else: y = "stagnate"
        lab.append(y)
lab = pd.Series(lab)
print("labelled seller-category-months", len(lab)); print(lab.value_counts(normalize=True).round(3).to_dict())
# share predictability: persistence baseline within categories having >=5 sellers
sh = m.merge(m.groupby(["cat", "month"]).rev.sum().rename("tot"), on=["cat", "month"])
sh["share"] = sh.rev / sh.tot
sh = sh.merge(sellers_per, on=["cat", "month"]); sh = sh[sh.k >= 5]
sh = sh.sort_values(["cat", "seller_id", "month"])
sh["next"] = sh.groupby(["cat", "seller_id"]).share.shift(-1)
sh["gap"] = (sh.groupby(["cat", "seller_id"]).month.shift(-1) - sh.month).apply(lambda x: x.n if pd.notna(x) else np.nan)
z = sh[sh.gap == 1].dropna(subset=["next"])
print("share rows with observed next month (cats with >=5 sellers):", len(z),
      "| persistence MAE (pct points): %.2f | mean share %.2f" % (100 * (z.next - z.share).abs().mean(), 100 * z.share.mean()))
