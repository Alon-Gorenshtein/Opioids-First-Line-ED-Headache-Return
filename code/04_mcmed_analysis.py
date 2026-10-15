#!/usr/bin/env python3
"""Phase-5 external replication (MC-MED / Stanford): adjusted opioid-vs-dopamine contrast.
Builds the treated adult headache cohort with covariates parallel to MIMIC, streams first
pain from numerics.csv, then runs the same overlap-weighted 72h/7d return + admission analysis.
"""
import os as _os
import re, json, os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MC = _os.environ.get("MCMED_DIR", "mc-med-1.0.0")
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
rng = np.random.default_rng(20260707)

CC_RE = re.compile(r"(headache|head ache|migraine|cephalgia|cephalalgia)", re.I)
def headache_icd(code):
    c=str(code).replace(".","").upper().strip()
    return c.startswith(("346","30781","339","G43","G44","R51")) or c=="7840"
DRUG_CLASSES = {
 "opioid": r"morphine|hydromorphone|dilaudid|oxycodone|hydrocodone|fentanyl|codeine|tramadol|oxymorphone|meperidine|demerol|percocet|butorphanol|nalbuphine|tapentadol",
 "dopamine_antagonist": r"metoclopramide|reglan|prochlorperazine|compazine|chlorpromazine|droperidol|promethazine|phenergan",
 "nsaid": r"ketorolac|toradol|ibuprofen|naproxen|ketoprofen|diclofenac|indomethacin|aspirin",
 "triptan": r"sumatriptan|rizatriptan|zolmitriptan|eletriptan|naratriptan|frovatriptan|almotriptan|imitrex|maxalt|zomig",
 "dhe": r"dihydroergotamine|d\.h\.e|dhe.?45|migranal", "magnesium": r"magnesium",
 "acetaminophen": r"acetaminophen|tylenol|ofirmev|paracetamol"}
CLASS_ORDER=["opioid","dopamine_antagonist","nsaid","triptan","dhe","magnesium","acetaminophen"]
COMPILED={k:re.compile(v,re.I) for k,v in DRUG_CLASSES.items()}
def classify(name):
    n=str(name).lower()
    for k in CLASS_ORDER:
        if COMPILED[k].search(n): return k
    return None
def race_group(r):
    r=str(r).upper()
    if "WHITE" in r: return "White"
    if "BLACK" in r or "AFRICAN" in r: return "Black"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic"
    if "ASIAN" in r: return "Asian"
    return "Other/Unknown"
def shift(h):
    return "day" if 7<=h<15 else ("evening" if 15<=h<23 else "night")

print("visits...")
v=pd.read_csv(f"{MC}/visits.csv", on_bad_lines="skip", low_memory=False)
v.columns=[c.strip() for c in v.columns]
cc=v["CC"].fillna("").astype(str); dxn=v.get("Dx_name",pd.Series([""]*len(v))).fillna("").astype(str)
is_ha=cc.str.contains(CC_RE)|dxn.str.contains(CC_RE)
for col in ["Dx_ICD9","Dx_ICD10"]:
    if col in v:
        s=v[col].fillna("").astype(str)
        is_ha=is_ha|s.apply(lambda x:any(headache_icd(t) for t in re.split(r"[;,\s]+",x) if t))
v["is_ha"]=is_ha
ha_csn=set(v.loc[is_ha,"CSN"])
print(f"  headache visits: {len(ha_csn):,}")

print("orders (meds)...")
chunks=[]
for ch in pd.read_csv(f"{MC}/orders.csv",usecols=["CSN","Order_type","Procedure_name","Order_time","First_admin_time"],
                      on_bad_lines="skip",chunksize=200_000,low_memory=False):
    ch=ch[(ch["Order_type"]=="Medications")&(ch["CSN"].isin(ha_csn))]
    if len(ch): chunks.append(ch)
med=pd.concat(chunks,ignore_index=True)
med["cls"]=med["Procedure_name"].map(classify); med=med.dropna(subset=["cls"])
med["t"]=pd.to_datetime(med["First_admin_time"].fillna(med["Order_time"]),errors="coerce",utc=True)
med=med.dropna(subset=["t"]); med["ord"]=med["cls"].map({c:i for i,c in enumerate(CLASS_ORDER)})
med=med.sort_values(["CSN","t","ord"])
fl=med.groupby("CSN").first()[["cls"]].rename(columns={"cls":"first_class"})

print("pain (stream numerics)...")
pain={}
for ch in pd.read_csv(f"{MC}/numerics.csv",usecols=["CSN","Measure","Value","Time"],on_bad_lines="skip",chunksize=1_000_000,low_memory=False):
    ch=ch[(ch["Measure"]=="Pain")&(ch["CSN"].isin(ha_csn))]
    if not len(ch): continue
    ch["Value"]=pd.to_numeric(ch["Value"],errors="coerce"); ch=ch.dropna(subset=["Value"])
    ch["Time"]=pd.to_datetime(ch["Time"],errors="coerce",utc=True)
    for csn,val in ch.sort_values("Time").groupby("CSN").first()["Value"].items():
        pain.setdefault(csn,val)

v=v.merge(fl,left_on="CSN",right_index=True,how="left")
v["treated"]=v["first_class"].notna()
v["age"]=pd.to_numeric(v["Age"],errors="coerce")
v["gender"]=v["Gender"].astype(str)
v["race_grp"]=v["Race"].map(race_group)
v["acuity"]=v["Triage_acuity"].fillna("missing").astype(str)
v["arrival_transport"]=v.get("Means_of_arrival","").fillna("missing").astype(str)
v["prior_ed"]=pd.to_numeric(v.get("Visit_no"),errors="coerce").fillna(1)-1
at=pd.to_datetime(v.get("Arrival_time"),errors="coerce",utc=True)
v["shift"]=at.dt.hour.map(lambda h: shift(h) if pd.notna(h) else "day")
v["pain_num"]=v["CSN"].map(pain)
disp=v["ED_dispo"].astype(str)
v["discharged"]=disp.eq("Discharge")
v["admitted"]=disp.isin(["Inpatient","ICU","Observation"])
h2n=pd.to_numeric(v.get("Hours_to_next_visit"),errors="coerce")
v["ret_72h"]=(h2n>0)&(h2n<=72); v["ret_7d"]=(h2n>0)&(h2n<=168); v["ret_30d"]=(h2n>0)&(h2n<=720)

v["_arrival"] = at                                               # parsed arrival time, for earliest-visit selection
v_sorted = v.assign(_at=at).sort_values(["MRN", "_at"])          # separate copy; do NOT reorder v
v["next_is_ha"] = v_sorted.groupby("MRN")["is_ha"].shift(-1)      # pandas aligns back to v by index
for wcol in ["ret_72h", "ret_7d"]:
    v[f"{wcol}_ha"] = v[wcol] & v["next_is_ha"].fillna(False).astype(bool)

coh=v[v["is_ha"]&v["treated"]&(v["age"]>=18)].copy()
# EARLIEST treated visit per patient (matches MIMIC new-user rule).
# .nth(0), not .first(): GroupBy.first() returns the first NON-NULL value of each column
# independently, so a patient with more than one treated headache visit would inherit a
# later visit's value into any column that is null at the index visit. pain_num is the only
# nullable column saved below (it is absent whenever the visit has no Pain measurement in
# numerics.csv), and 33 patients were carrying a later visit's pain score. Same defect as
# 02_cohort.py; .nth(0) selects the actual first row of each group.
coh=coh.sort_values(["MRN","_arrival"]).groupby("MRN",as_index=False).nth(0)
print(f"MC-MED analytic cohort: {len(coh):,}; opioid {int((coh.first_class=='opioid').sum())}, dopa {int((coh.first_class=='dopamine_antagonist').sum())}")

# save harmonized cohort for pooled analysis
coh.assign(site="MC-MED", pid=coh["MRN"])[["pid","site","first_class","age","gender","race_grp","acuity",
    "pain_num","arrival_transport","prior_ed","shift","admitted","discharged","ret_72h","ret_7d","ret_30d",
    "ret_72h_ha","ret_7d_ha"]
    ].to_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet", index=False)

CAT=["gender","race_grp","acuity","arrival_transport","shift"]; NUM=["age","pain_num","prior_ed"]
def design(d):
    d=d.copy(); d["pain_miss"]=d["pain_num"].isna().astype(int); d["pain_num"]=d["pain_num"].fillna(d["pain_num"].median() if d["pain_num"].notna().any() else 5)
    X=pd.get_dummies(d[CAT],drop_first=True,dummy_na=False)
    return pd.concat([d[NUM+["pain_miss"]].reset_index(drop=True),X.reset_index(drop=True)],axis=1).astype(float)
def ow(d):
    y=d["opioid"].values; X=StandardScaler().fit_transform(design(d).values)
    e=LogisticRegression(max_iter=2000).fit(X,y).predict_proba(X)[:,1]
    return np.where(y==1,1-e,e)
def wr(y,w): return np.sum(w*y)/np.sum(w)
def smd(x,t,w):
    x=np.asarray(x,float)
    def mv(m): ww=w[m]; xx=x[m]; mm=np.sum(ww*xx)/np.sum(ww); return mm,np.sum(ww*(xx-mm)**2)/np.sum(ww)
    m1,v1=mv(t==1); m0,v0=mv(t==0); s=np.sqrt((v1+v0)/2); return 0 if s==0 else (m1-m0)/s
def analyze(pop,outcome,label):
    d=pop.reset_index(drop=True); w=ow(d); y=d[outcome].astype(float).values; t=d["opioid"].values
    r1=wr(y[t==1],w[t==1]); r0=wr(y[t==0],w[t==0]); rr=r1/r0 if r0>0 else np.nan
    rrs=[]
    for _ in range(1000):
        idx=rng.integers(0,len(d),len(d)); db=d.iloc[idx]
        try:
            wb=ow(db); yb=db[outcome].astype(float).values; tb=db["opioid"].values
            b1=wr(yb[tb==1],wb[tb==1]); b0=wr(yb[tb==0],wb[tb==0]); rrs.append(b1/b0 if b0>0 else np.nan)
        except Exception: pass
    ci=np.nanpercentile(rrs,[2.5,97.5])
    X=design(d); post=max(abs(smd(X[c].values,t,w)) for c in X.columns)
    print(f"  [{label}] {outcome}: opioid {100*r1:.2f}% vs dopa {100*r0:.2f}% | RR {rr:.2f} (95% CI {ci[0]:.2f},{ci[1]:.2f}) | postSMD {post:.3f}")
    return dict(label=label,outcome=outcome,n=int(len(d)),risk_opioid=round(100*r1,2),risk_dopa=round(100*r0,2),
                rr=round(rr,3),rr_ci=[round(ci[0],3),round(ci[1],3)],smd_post=round(post,3))

c=coh[coh["first_class"].isin(["opioid","dopamine_antagonist"])].copy()
c["opioid"]=(c["first_class"]=="opioid").astype(int)
dis=c[c["discharged"]].copy()
res=[analyze(dis,"ret_72h","72h return"),analyze(dis,"ret_7d","7d return"),analyze(c,"admitted","admission")]
crude=coh[coh["discharged"]].groupby("first_class").agg(n=("CSN","size"),ret72=("ret_72h","mean"),ret7d=("ret_7d","mean"))
crude[["ret72","ret7d"]]=(crude[["ret72","ret7d"]]*100).round(2)
print("crude by class (discharged):\n",crude.sort_values("n",ascending=False).to_string())
json.dump({"n_cohort":int(len(coh)),"adjusted":res,
           "crude_by_class":{k:{"n":int(r["n"]),"ret72":float(r["ret72"]),"ret7d":float(r["ret7d"])} for k,r in crude.iterrows()}},
          open(f"{OUT}/results_digest_mcmed.json","w"),indent=2)
print("wrote results_digest_mcmed.json")
