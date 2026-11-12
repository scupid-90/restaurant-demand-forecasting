import sys, json
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import demand_model as dm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE

df = dm.load_data(ROOT / "data" / "restaurant_sales_data.csv")
feats = dm.CAT_FEATURES + dm.BOOL_FEATURES
X, y = df[feats], df[dm.TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
tr = Xtr.assign(y=ytr)
res = {}
# 1 global mean
p1 = np.full(len(yte), ytr.mean())
# 2 per-item mean
m = tr.groupby("menu_item_name").y.mean(); p2 = Xte["menu_item_name"].map(m).fillna(ytr.mean()).values
# 3 item x meal x weekday mean
g = tr.groupby(["menu_item_name","meal_type","weekday"]).y.mean()
p3 = np.array([g.get((a,b,c), m.get(a, ytr.mean())) for a,b,c in zip(Xte.menu_item_name,Xte.meal_type,Xte.weekday)])
# 4 linear regression
pre = lambda: ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), dm.CAT_FEATURES)], remainder="passthrough")
lr = Pipeline([("p", pre()), ("m", LinearRegression())]).fit(Xtr, ytr); p4 = lr.predict(Xte)
# 5 gradient boosting
gb = Pipeline([("p", pre()), ("m", GradientBoostingRegressor(random_state=42))]).fit(Xtr, ytr); p5 = gb.predict(Xte)
# 6 RF (ours)
rf = dm.build_pipeline().fit(Xtr, ytr); p6 = rf.predict(Xte)
names = ["Global average","Per-item average","Item x meal x weekday average","Linear regression","Gradient boosting","Random forest (ours)"]
preds = [p1,p2,p3,p4,p5,p6]
for n,p in zip(names,preds):
    res[n] = {"mae": round(MAE(yte,p),1), "rmse": round(float(np.sqrt(np.mean((yte-p)**2))),1)}
    print(n, res[n])

# policy simulation on test set
econ = dm.item_economics(df)
resid = dm.per_item_residual_std(rf, df)
ic = Xte["menu_item_name"].map(lambda i: econ[i]["ingredient_cost"]).values
mg = Xte["menu_item_name"].map(lambda i: econ[i]["margin"]).values
def cost(prep):
    over = np.maximum(prep - yte.values, 0); under = np.maximum(yte.values - prep, 0)
    return over.sum(), under.sum(), (over*ic).sum(), (under*mg).sum()
# policies
cr = np.clip(mg/(mg+ic), .5, .97)
z = np.array([dm._norm_ppf(c) for c in cr]); sd = Xte["menu_item_name"].map(resid).values
pols = {
 "Prep the item's average (par level)": p2,
 "Prep the item's average + 10% cushion": p2*1.10,
 "Item x meal x weekday average": p3,
 "Our forecast, used as-is": p6,
 "Our forecast + newsvendor buffer": np.maximum(0, p6 + z*sd),
}
out = {}
for n,p in pols.items():
    ou, un, wc, lm = cost(p)
    out[n] = dict(waste_units=float(ou), short_units=float(un), waste_cost=float(wc), lost_margin=float(lm), total=float(wc+lm), fill=float(1-un/yte.sum()))
    print(f"{n:42s} waste_units={ou:8.0f} short={un:8.0f} wasteCost={wc:10.0f} lostMargin={lm:10.0f} total={wc+lm:10.0f} fill={1-un/yte.sum():.3f}")
print("test units", yte.sum(), "n", len(yte))
json.dump({"models":res,"policies":out,"n_test":len(yte),"test_units":float(yte.sum())}, open(ROOT / "results" / "week3_benchmark.json","w"), indent=1)
