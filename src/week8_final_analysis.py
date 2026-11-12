import json, sys, numpy as np, pandas as pd
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import demand_model as dm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
df = dm.load_data(ROOT / "data" / "restaurant_sales_data.csv")
feats = dm.CAT_FEATURES + dm.BOOL_FEATURES
print(df.columns.tolist()); print(df['date'].min(), df['date'].max())
econ = dm.item_economics(df)
def evalsplit(tr, te, label):
    ytr, yte = tr[dm.TARGET], te[dm.TARGET]
    rf = dm.build_pipeline().fit(tr[feats], ytr); p6 = rf.predict(te[feats])
    m = tr.groupby("menu_item_name")[dm.TARGET].mean()
    p2 = te["menu_item_name"].map(m).fillna(ytr.mean()).values
    resid = dm.per_item_residual_std(rf, tr)
    ic = te["menu_item_name"].map(lambda i: econ[i]["ingredient_cost"]).values
    mg = te["menu_item_name"].map(lambda i: econ[i]["margin"]).values
    cr = np.clip(mg/(mg+ic), .5, .97); z = np.array([dm._norm_ppf(c) for c in cr])
    sd = te["menu_item_name"].map(resid).fillna(np.mean(list(resid.values()))).values
    pn = np.maximum(0, p6 + z*sd)
    def sim(prep):
        o=np.maximum(prep-yte.values,0); u=np.maximum(yte.values-prep,0)
        return dict(waste=o.sum(), short=u.sum(), fill=1-u.sum()/yte.sum(), cost=(o*ic).sum()+(u*mg).sum())
    r = dict(label=label, n=len(te), mae_rf=MAE(yte,p6), mae_par=MAE(yte,p2), par=sim(p2), rf=sim(p6), buf=sim(pn))
    # scale par to match buffered fill
    lo,hi=0.5,3
    for _ in range(60):
        k=(lo+hi)/2
        if sim(p2*k)['fill']<r['buf']['fill']: lo=k
        else: hi=k
    r['par_eq']=sim(p2*hi); r['par_eq_k']=hi
    return r, rf, te.assign(p6=p6,p2=p2,ic=ic,mg=mg,pn=pn)
out={}
# random split (reproduces week 3)
Xtr,Xte = train_test_split(df, test_size=0.2, random_state=42)
out['random'],rf,tt = evalsplit(Xtr,Xte,'random')
# time split: first 80% of days train, last 20% test
d = df['date'].sort_values().unique(); cut = d[int(len(d)*0.8)]
out['time'],_,tt2 = evalsplit(df[df.date<cut], df[df.date>=cut], f'time cut {cut}')
# per restaurant check (random-split test): RF vs par MAE
g = tt.groupby('restaurant_id').apply(lambda x: pd.Series(dict(rf=MAE(x[dm.TARGET],x.p6), par=MAE(x[dm.TARGET],x.p2), n=len(x))))
out['rest_better']=int((g.rf<g.par).sum()); out['rest_total']=len(g); out['rest_g']=g.describe().to_dict()
gt = tt2.groupby('restaurant_id').apply(lambda x: pd.Series(dict(rf=MAE(x[dm.TARGET],x.p6), par=MAE(x[dm.TARGET],x.p2))))
out['rest_better_time']=int((gt.rf<gt.par).sum()); out['rest_total_time']=len(gt)
# per type
out['by_type']=tt.groupby('restaurant_type').apply(lambda x: dict(rf=MAE(x[dm.TARGET],x.p6), par=MAE(x[dm.TARGET],x.p2))).to_dict()
# feature importance (permutation-free: pipeline importances)
m=rf.named_steps[list(rf.named_steps)[-1]]; pre=rf.named_steps[list(rf.named_steps)[0]]
try:
    names=pre.get_feature_names_out(); imp=pd.Series(m.feature_importances_,index=names).sort_values(ascending=False)
    agg={}
    for n,v in imp.items():
        k=n.split('__')[-1]
        for c in feats:
            if k.startswith(c): k=c; break
        agg[k]=agg.get(k,0)+v
    out['imp']=dict(sorted(agg.items(),key=lambda x:-x[1]))
except Exception as e: print('imp err',e)
print(json.dumps(out,indent=1,default=float))
json.dump(out,open(ROOT / "results" / "week8_final.json","w"),indent=1,default=float)
