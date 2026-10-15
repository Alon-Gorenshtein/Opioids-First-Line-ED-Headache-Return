#!/usr/bin/env python3
"""eTable 3: per-covariate standardized mean differences (SMD) before and after overlap
weighting, primary contrast (opioid vs dopamine antagonist, discharged 72h population, MIMIC-IV-ED).
Reuses the design/overlap_weights/smd logic from 03_primary_analysis.py so numbers reconcile exactly."""
import os as _os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
df=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
CAT=["gender","race_grp","acuity","arrival_transport","shift","yeargrp"]; NUM=["age","pain_num","prior_ed"]
def design(d):
    d=d.copy(); d["pain_miss"]=d["pain_num"].isna().astype(int)
    d["pain_num"]=d["pain_num"].fillna(d["pain_num"].median()); d["acuity"]=d["acuity"].fillna("missing").astype(str)
    X=pd.get_dummies(d[CAT],drop_first=True,dummy_na=False)
    return pd.concat([d[NUM+["pain_miss"]].reset_index(drop=True),X.reset_index(drop=True)],axis=1).astype(float)
def overlap_weights(d):
    y=d["opioid"].values; Xs=StandardScaler().fit_transform(design(d).values)
    lr=LogisticRegression(max_iter=2000,C=1.0).fit(Xs,y); e=lr.predict_proba(Xs)[:,1]
    return np.where(y==1,1-e,e)
def smd(x,t,w=None):
    x=np.asarray(x,float); t=np.asarray(t); w=np.ones_like(x) if w is None else w
    def mv(m): ww=w[m]; xx=x[m]; mu=np.sum(ww*xx)/np.sum(ww); return mu,np.sum(ww*(xx-mu)**2)/np.sum(ww)
    m1,v1=mv(t==1); m0,v0=mv(t==0); s=np.sqrt((v1+v0)/2)
    return 0.0 if s==0 else (m1-m0)/s
LABEL={"age":"Age, y","pain_num":"Triage pain score","prior_ed":"Prior ED visits","pain_miss":"Triage pain missing",
 "gender_M":"Male sex","race_grp_Black":"Black race","race_grp_Hispanic":"Hispanic","race_grp_Other/Unknown":"Race other/unknown",
 "race_grp_White":"White race","acuity_2.0":"Triage acuity 2","acuity_3.0":"Triage acuity 3","acuity_4.0":"Triage acuity 4",
 "acuity_5.0":"Triage acuity 5","acuity_missing":"Triage acuity missing","arrival_transport_HELICOPTER":"Arrival by helicopter",
 "arrival_transport_OTHER":"Arrival other","arrival_transport_UNKNOWN":"Arrival unknown","arrival_transport_WALK IN":"Arrival walk-in",
 "shift_evening":"Evening arrival (15-23h)","shift_night":"Night arrival (23-7h)",
 "yeargrp_2014 - 2016":"Index period 2014-2016","yeargrp_2017 - 2019":"Index period 2017-2019"}
c=df[df["first_class"].isin(["opioid","dopamine_antagonist"])].copy()
c["opioid"]=(c["first_class"]=="opioid").astype(int)
d=c[c["discharged"]].reset_index(drop=True)
w=overlap_weights(d); t=d["opioid"].values; X=design(d)
rows=[]; yr_pre=[]; yr_post=[]
for col in X.columns:
    pre=smd(X[col].values,t); post=smd(X[col].values,t,w)
    if col.startswith("yeargrp_"):   # collapse MIMIC deidentified anchor-year dummies into one row
        yr_pre.append(abs(pre)); yr_post.append(abs(post)); continue
    rows.append((LABEL.get(col,col),pre,post))
rows.sort(key=lambda r:-abs(r[1]))
if yr_pre:
    rows.append((f"Index calendar period (max across {len(yr_pre)} 3-year terms)", max(yr_pre), max(yr_post)))
md="| Covariate | SMD before weighting | SMD after weighting |\n|---|---:|---:|\n"
for lab,pre,post in rows: md+=f"| {lab} | {pre:+.3f} | {post:+.3f} |\n"
allpre=[abs(r[1]) for r in rows]+yr_pre; allpost=[abs(r[2]) for r in rows]+yr_post
md+=f"| **Maximum absolute SMD (all terms)** | **{max(allpre):.3f}** | **{max(allpost):.3f}** |\n"
open(f"{OUT}/tables/etable3_balance.md","w").write(md)
print(md); print("wrote etable3_balance.md  (n=%d, opioid=%d)"%(len(d),int(t.sum())))
