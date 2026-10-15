#!/usr/bin/env python3
"""Phase-5 primary analysis (MIMIC-IV-ED): opioid vs dopamine-antagonist first-line.
Overlap-weighted (Li 2018) risk of 72h/7d ED return (among discharged) and index admission.
Reports weighted risks, risk difference + risk ratio with patient-bootstrap 95% CIs,
covariate balance (SMD), and an E-value for the return RR. Descriptive competing-risks CIF by class.
Non-causal framing: adjusted associations.
"""
import os as _os
import json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
rng = np.random.default_rng(20260707)
df = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")

CAT = ["gender","race_grp","acuity","arrival_transport","shift","yeargrp"]
NUM = ["age","pain_num","prior_ed"]

def design(d):
    d = d.copy()
    d["pain_miss"] = d["pain_num"].isna().astype(int)
    d["pain_num"] = d["pain_num"].fillna(d["pain_num"].median())
    d["acuity"] = d["acuity"].fillna("missing").astype(str)
    X = pd.get_dummies(d[CAT], drop_first=True, dummy_na=False)
    X = pd.concat([d[NUM+["pain_miss"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return X.astype(float)

def overlap_weights(d):
    """Return overlap weights for opioid(1) vs dopamine(0) within population d."""
    y = d["opioid"].values
    X = design(d).values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xs, y)
    e = lr.predict_proba(Xs)[:,1]
    w = np.where(y==1, 1-e, e)   # overlap weight
    return w, e

def wrisk(y, w):
    return np.sum(w*y)/np.sum(w)

def smd(x, t, w=None):
    x=np.asarray(x,float); t=np.asarray(t)
    if w is None: w=np.ones_like(x)
    def mv(mask):
        ww=w[mask]; xx=x[mask]
        m=np.sum(ww*xx)/np.sum(ww)
        v=np.sum(ww*(xx-m)**2)/np.sum(ww)
        return m,v
    m1,v1=mv(t==1); m0,v0=mv(t==0)
    s=np.sqrt((v1+v0)/2)
    return 0.0 if s==0 else (m1-m0)/s

def analyze(pop, outcome, label):
    d = pop.reset_index(drop=True)
    w,e = overlap_weights(d)
    y = d[outcome].astype(float).values
    t = d["opioid"].values
    r1 = wrisk(y[t==1], w[t==1]); r0 = wrisk(y[t==0], w[t==0])
    rd = r1-r0; rr = r1/r0 if r0>0 else np.nan
    # patient bootstrap
    rds,rrs=[],[]
    for _ in range(1000):
        idx = rng.integers(0,len(d),len(d))
        db = d.iloc[idx]
        try:
            wb,_=overlap_weights(db); yb=db[outcome].astype(float).values; tb=db["opioid"].values
            b1=wrisk(yb[tb==1],wb[tb==1]); b0=wrisk(yb[tb==0],wb[tb==0])
            rds.append(b1-b0); rrs.append(b1/b0 if b0>0 else np.nan)
        except Exception: pass
    rd_ci=np.nanpercentile(rds,[2.5,97.5]); rr_ci=np.nanpercentile(rrs,[2.5,97.5])
    # balance (max |SMD| pre/post across design cols)
    X=design(d); pre=max(abs(smd(X[c].values,t)) for c in X.columns)
    post=max(abs(smd(X[c].values,t,w)) for c in X.columns)
    # E-value for RR (VanderWeele)
    rr_=rr if rr>=1 else 1/rr
    ev=rr_+np.sqrt(rr_*(rr_-1))
    res=dict(label=label, outcome=outcome, n=int(len(d)), n_opioid=int(t.sum()),
             risk_opioid=round(100*r1,2), risk_dopa=round(100*r0,2),
             rd_pct=round(100*rd,2), rd_ci=[round(100*rd_ci[0],2),round(100*rd_ci[1],2)],
             rr=round(rr,3), rr_ci=[round(rr_ci[0],3),round(rr_ci[1],3)],
             smd_pre=round(pre,3), smd_post=round(post,3), evalue=round(ev,2))
    print(f"\n[{label}] {outcome}  n={len(d)} (opioid {int(t.sum())})")
    print(f"  risk: opioid {100*r1:.2f}%  vs dopamine {100*r0:.2f}%")
    print(f"  RD {100*rd:+.2f}pp (95% CI {100*rd_ci[0]:+.2f}, {100*rd_ci[1]:+.2f}) | RR {rr:.2f} (95% CI {rr_ci[0]:.2f}, {rr_ci[1]:.2f})")
    print(f"  balance max|SMD| pre {pre:.3f} -> post {post:.3f} | E-value {ev:.2f}")
    return res

contrast = df[df["first_class"].isin(["opioid","dopamine_antagonist"])].copy()
contrast["opioid"] = (contrast["first_class"]=="opioid").astype(int)
disch = contrast[contrast["discharged"]].copy()

results=[]
results.append(analyze(disch, "ret_72h", "72h return | discharged, opioid vs dopamine"))
results.append(analyze(disch, "ret_7d",  "7d return | discharged, opioid vs dopamine"))
results.append(analyze(contrast, "admitted", "index admission | all, opioid vs dopamine"))

# descriptive crude return by class (all 6, discharged)
crude = df[df["discharged"]].groupby("first_class").agg(
    n=("stay_id","size"), ret72=("ret_72h","mean"), ret7d=("ret_7d","mean"), ret30=("ret_30d","mean"))
crude[["ret72","ret7d","ret30"]] = (crude[["ret72","ret7d","ret30"]]*100).round(2)
print("\nCrude return by class (discharged):\n", crude.sort_values("n",ascending=False).to_string())

json.dump({"primary": results,
           "crude_by_class": {k:{"n":int(r["n"]),"ret72":float(r["ret72"]),"ret7d":float(r["ret7d"]),"ret30":float(r["ret30"])}
                              for k,r in crude.iterrows()}},
          open(f"{OUT}/results_digest.json","w"), indent=2)
print("\nwrote results_digest.json")
