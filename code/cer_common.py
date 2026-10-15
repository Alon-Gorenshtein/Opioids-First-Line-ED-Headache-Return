#!/usr/bin/env python3
"""Shared comparative-effectiveness estimator for the headache ED-revisit study.
One overlap-weighting/bootstrap/E-value implementation, imported by every sensitivity
script so all analyses are numerically consistent with the primary analysis."""
import zlib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MIMIC_CAT = ["gender","race_grp","acuity","arrival_transport","shift","yeargrp"]
POOL_CAT  = ["gender","race_grp","acuity","arrival_transport","shift","site"]
SITE_CAT  = ["gender","race_grp","acuity","arrival_transport","shift"]
NUM = ["age","pain_num","prior_ed"]

def design(d, cat, num=NUM):
    d = d.copy()
    d["pain_miss"] = d["pain_num"].isna().astype(int)
    med = d["pain_num"].median(); d["pain_num"] = d["pain_num"].fillna(med if pd.notna(med) else 5)
    d["acuity"] = d["acuity"].fillna("missing").astype(str)
    X = pd.get_dummies(d[cat], drop_first=True, dummy_na=False)
    return pd.concat([d[num+["pain_miss"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1).astype(float)

def overlap_weights(d, cat, num=NUM):
    y = d["opioid"].values
    Xs = StandardScaler().fit_transform(design(d, cat, num).values)
    e = LogisticRegression(max_iter=3000).fit(Xs, y).predict_proba(Xs)[:,1]
    return np.where(y==1, 1-e, e)

def wrisk(y, w): return np.sum(w*y)/np.sum(w)

def smd(x, t, w):
    x = np.asarray(x, float)
    def mv(mk): ww=w[mk]; xx=x[mk]; mm=np.sum(ww*xx)/np.sum(ww); return mm, np.sum(ww*(xx-mm)**2)/np.sum(ww)
    m1,v1 = mv(t==1); m0,v0 = mv(t==0); s = np.sqrt((v1+v0)/2)
    return 0.0 if s==0 else (m1-m0)/s

def effective_n(w, t):
    def ess(mk): ww=w[mk]; return float(ww.sum()**2/np.sum(ww**2)) if np.sum(ww**2)>0 else 0.0
    return round(ess(t==1),1), round(ess(t==0),1)

def analyze(pop, outcome, cat, num=NUM, seed=20260707, n_boot=1000, label=""):
    d = pop.reset_index(drop=True)
    rng = np.random.default_rng(seed + zlib.crc32((outcome + label).encode()))   # deterministic, distinct per outcome
    w = overlap_weights(d, cat, num); y = d[outcome].astype(float).values; t = d["opioid"].values
    r1 = wrisk(y[t==1], w[t==1]); r0 = wrisk(y[t==0], w[t==0])
    rr = r1/r0 if r0>0 else np.nan; rd = r1-r0
    rrs=[]; rds=[]
    for _ in range(n_boot):
        idx = rng.integers(0,len(d),len(d)); db = d.iloc[idx]
        try:
            wb = overlap_weights(db, cat, num); yb = db[outcome].astype(float).values; tb = db["opioid"].values
            b1 = wrisk(yb[tb==1],wb[tb==1]); b0 = wrisk(yb[tb==0],wb[tb==0])
            rrs.append(b1/b0 if b0>0 else np.nan); rds.append(b1-b0)
        except Exception: pass
    ci = np.nanpercentile(rrs,[2.5,97.5]); dci = np.nanpercentile(rds,[2.5,97.5])
    X = design(d, cat, num)
    pre  = max(abs(smd(X[c].values, t, np.ones_like(w))) for c in X.columns)
    post = max(abs(smd(X[c].values, t, w)) for c in X.columns)
    rr_ = rr if rr>=1 else 1/rr; ev = rr_+np.sqrt(rr_*(rr_-1))
    eo, ed_ = effective_n(w, t)
    return dict(label=label, outcome=outcome, n=int(len(d)), n_opioid=int(t.sum()),
                risk_opioid=round(100*r1,2), risk_dopa=round(100*r0,2),
                rd_pct=round(100*rd,2), rd_ci=[round(100*dci[0],2), round(100*dci[1],2)],
                rr=round(rr,3), rr_ci=[round(ci[0],3), round(ci[1],3)],
                smd_pre=round(pre,3), smd_post=round(post,3), evalue=round(ev,2),
                ess_opioid=eo, ess_dopa=ed_)

def stack_pooled(out_dir):
    m = pd.read_parquet(f"{out_dir}/intermediate/cohort_mimic.parquet"); m["site"]="MIMIC"; m["pid"]=m["subject_id"]
    mc = pd.read_parquet(f"{out_dir}/intermediate/cohort_mcmed.parquet"); mc["site"]="MC-MED"
    if "pid" not in mc.columns: mc["pid"]=range(len(mc))
    shared = [c for c in m.columns if c in mc.columns]
    return pd.concat([m[shared], mc[shared]], ignore_index=True)
