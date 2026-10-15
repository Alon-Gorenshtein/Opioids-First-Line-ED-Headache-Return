#!/usr/bin/env python3
"""Phase-5 pooled two-center analysis: opioid vs dopamine-antagonist, MIMIC-IV-ED + MC-MED stacked
with a site term. Overlap-weighted 72h/7d return (discharged) and index admission; per-site + pooled
RRs with bootstrap CIs and E-values. Feeds the forest figure.
"""
import os as _os
import json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
rng=np.random.default_rng(20260707)
COMMON=["pid","site","first_class","age","gender","race_grp","acuity","pain_num","arrival_transport","prior_ed","shift","admitted","discharged","ret_72h","ret_7d","ret_30d"]

m=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
m["site"]="MIMIC"; m["pid"]=m["subject_id"]
mc=pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet")
pool=pd.concat([m[COMMON], mc[COMMON]], ignore_index=True)
print(f"pooled cohort {len(pool):,} (MIMIC {len(m):,}, MC-MED {len(mc):,})")

CAT=["gender","race_grp","acuity","arrival_transport","shift","site"]; NUM=["age","pain_num","prior_ed"]
def design(d):
    d=d.copy(); d["pain_miss"]=d["pain_num"].isna().astype(int)
    med=d["pain_num"].median(); d["pain_num"]=d["pain_num"].fillna(med if pd.notna(med) else 5)
    d["acuity"]=d["acuity"].fillna("missing").astype(str)
    X=pd.get_dummies(d[CAT],drop_first=True,dummy_na=False)
    return pd.concat([d[NUM+["pain_miss"]].reset_index(drop=True),X.reset_index(drop=True)],axis=1).astype(float)
def ow(d):
    y=d["opioid"].values; X=StandardScaler().fit_transform(design(d).values)
    e=LogisticRegression(max_iter=3000).fit(X,y).predict_proba(X)[:,1]
    return np.where(y==1,1-e,e)
def wr(y,w): return np.sum(w*y)/np.sum(w)
def smd(x,t,w):
    x=np.asarray(x,float)
    def mv(mk): ww=w[mk]; xx=x[mk]; mm=np.sum(ww*xx)/np.sum(ww); return mm,np.sum(ww*(xx-mm)**2)/np.sum(ww)
    m1,v1=mv(t==1); m0,v0=mv(t==0); s=np.sqrt((v1+v0)/2); return 0 if s==0 else (m1-m0)/s
def analyze(pop,outcome,label,use_site=True):
    d=pop.reset_index(drop=True)
    global CAT
    CAT=["gender","race_grp","acuity","arrival_transport","shift"]+(["site"] if use_site else [])
    w=ow(d); y=d[outcome].astype(float).values; t=d["opioid"].values
    r1=wr(y[t==1],w[t==1]); r0=wr(y[t==0],w[t==0]); rr=r1/r0 if r0>0 else np.nan; rd=r1-r0
    rrs=[];rds=[]
    for _ in range(1000):
        idx=rng.integers(0,len(d),len(d)); db=d.iloc[idx]
        try:
            wb=ow(db); yb=db[outcome].astype(float).values; tb=db["opioid"].values
            b1=wr(yb[tb==1],wb[tb==1]); b0=wr(yb[tb==0],wb[tb==0]); rrs.append(b1/b0 if b0>0 else np.nan); rds.append(b1-b0)
        except Exception: pass
    ci=np.nanpercentile(rrs,[2.5,97.5]); dci=np.nanpercentile(rds,[2.5,97.5])
    X=design(d); post=max(abs(smd(X[c].values,t,w)) for c in X.columns)
    rr_=rr if rr>=1 else 1/rr; ev=rr_+np.sqrt(rr_*(rr_-1))
    print(f"  [{label}] {outcome}: opioid {100*r1:.2f}% vs dopa {100*r0:.2f}% | RR {rr:.2f} ({ci[0]:.2f},{ci[1]:.2f}) | RD {100*rd:+.2f}pp ({100*dci[0]:+.2f},{100*dci[1]:+.2f}) | postSMD {post:.3f} | E {ev:.2f}")
    return dict(label=label,outcome=outcome,n=int(len(d)),n_opioid=int(t.sum()),risk_opioid=round(100*r1,2),risk_dopa=round(100*r0,2),
                rd_pct=round(100*rd,2),rd_ci=[round(100*dci[0],2),round(100*dci[1],2)],rr=round(rr,3),rr_ci=[round(ci[0],3),round(ci[1],3)],
                smd_post=round(post,3),evalue=round(ev,2))

pool2=pool[pool["first_class"].isin(["opioid","dopamine_antagonist"])].copy()
pool2["opioid"]=(pool2["first_class"]=="opioid").astype(int)
dis=pool2[pool2["discharged"]].copy()
print("POOLED (two-center):")
res={"pooled":[analyze(dis,"ret_72h","pooled 72h"),analyze(dis,"ret_7d","pooled 7d"),analyze(pool2,"admitted","pooled admission")]}
# per-site for forest
res["by_site"]={}
for s in ["MIMIC","MC-MED"]:
    ds=dis[dis["site"]==s]; ca=pool2[pool2["site"]==s]
    print(f"SITE {s}:")
    res["by_site"][s]=[analyze(ds,"ret_72h",f"{s} 72h",use_site=False),
                        analyze(ds,"ret_7d",f"{s} 7d",use_site=False),
                        analyze(ca,"admitted",f"{s} admission",use_site=False)]
json.dump(res, open(f"{OUT}/results_digest_pooled.json","w"), indent=2)
print("wrote results_digest_pooled.json")
