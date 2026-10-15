#!/usr/bin/env python3
"""Phase-1 reconnaissance for MC-MED (Stanford ED) — external-replication feasibility.
Same questions as MIMIC recon: headache cohort, first-line class, return-visit rate by class.
MC-MED gives Hours_to_next_visit + Dispo_class_next_visit directly in visits.csv.
"""
import os as _os
import re, json, os
import pandas as pd, numpy as np

MC = _os.environ.get("MCMED_DIR", "mc-med-1.0.0")
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

CC_RE = re.compile(r"(headache|head ache|migraine|cephalgia|cephalalgia)", re.I)
def headache_icd(code):
    c = str(code).replace(".", "").upper().strip()
    return c.startswith(("346","30781","339","G43","G44","R51")) or c == "7840"

DRUG_CLASSES = {
 "opioid": r"morphine|hydromorphone|dilaudid|oxycodone|hydrocodone|fentanyl|codeine|tramadol|oxymorphone|meperidine|demerol|percocet|butorphanol|nalbuphine|tapentadol",
 "dopamine_antagonist": r"metoclopramide|reglan|prochlorperazine|compazine|chlorpromazine|droperidol|promethazine|phenergan",
 "nsaid": r"ketorolac|toradol|ibuprofen|naproxen|ketoprofen|diclofenac|indomethacin|aspirin",
 "triptan": r"sumatriptan|rizatriptan|zolmitriptan|eletriptan|naratriptan|frovatriptan|almotriptan|imitrex|maxalt|zomig",
 "dhe": r"dihydroergotamine|d\.h\.e|dhe.?45|migranal",
 "magnesium": r"magnesium",
 "acetaminophen": r"acetaminophen|tylenol|ofirmev|paracetamol",
}
CLASS_ORDER = ["opioid","dopamine_antagonist","nsaid","triptan","dhe","magnesium","acetaminophen"]
COMPILED = {k: re.compile(v, re.I) for k, v in DRUG_CLASSES.items()}
def classify(name):
    n = str(name).lower()
    for k in CLASS_ORDER:
        if COMPILED[k].search(n):
            return k
    return None

print("loading visits.csv...")
v = pd.read_csv(f"{MC}/visits.csv", on_bad_lines="skip", low_memory=False)
v.columns = [c.strip() for c in v.columns]
cc = v["CC"].fillna("").astype(str)
dx9 = v.get("Dx_ICD9","").fillna("").astype(str) if "Dx_ICD9" in v else pd.Series([""]*len(v))
dx10 = v.get("Dx_ICD10","").fillna("").astype(str) if "Dx_ICD10" in v else pd.Series([""]*len(v))
dxn = v.get("Dx_name","").fillna("").astype(str) if "Dx_name" in v else pd.Series([""]*len(v))
is_ha = cc.str.contains(CC_RE) | dxn.str.contains(CC_RE)
# ICD fields can be lists; check any token
for col in [dx9, dx10]:
    is_ha = is_ha | col.apply(lambda s: any(headache_icd(t) for t in re.split(r"[;,\s]+", s) if t))
v["is_ha"] = is_ha
print(f"  total visits: {len(v):,}")
print(f"  headache visits: {int(v['is_ha'].sum()):,}")
ha_csn = set(v.loc[v["is_ha"], "CSN"])

print("loading medication orders (filtered)...")
chunks = []
for ch in pd.read_csv(f"{MC}/orders.csv", usecols=["CSN","Order_type","Procedure_name","Order_time","First_admin_time"],
                      on_bad_lines="skip", chunksize=200_000, low_memory=False):
    ch = ch[(ch["Order_type"]=="Medications") & (ch["CSN"].isin(ha_csn))]
    if len(ch): chunks.append(ch)
med = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["CSN","Procedure_name","Order_time","First_admin_time"])
med["cls"] = med["Procedure_name"].map(classify)
med = med.dropna(subset=["cls"])
# administration time preferred, else order time
med["t"] = pd.to_datetime(med["First_admin_time"].fillna(med["Order_time"]), errors="coerce", utc=True)
med = med.dropna(subset=["t"])
med["ord"] = med["cls"].map({c:i for i,c in enumerate(CLASS_ORDER)})
med = med.sort_values(["CSN","t","ord"])
first_line = med.groupby("CSN").first()[["cls"]].rename(columns={"cls":"first_class"})
v = v.merge(first_line, left_on="CSN", right_index=True, how="left")
v["treated"] = v["first_class"].notna()

ha = v[v["is_ha"]].copy()
treated = ha[ha["treated"]].copy()
print(f"  headache treated: {len(treated):,} / {len(ha):,} ({100*len(treated)/max(len(ha),1):.1f}%)")
print("  first-line class distribution (treated):")
print(treated["first_class"].value_counts().to_string())

# returns from Hours_to_next_visit
h = pd.to_numeric(treated.get("Hours_to_next_visit"), errors="coerce")
treated = treated.assign(h2n=h)
for w,lab in [(72,"72h"),(168,"7d"),(720,"30d")]:
    treated[f"ret_{lab}"] = (treated["h2n"]>0) & (treated["h2n"]<=w)
print("  RETURN-VISIT rates (treated):")
for lab in ["72h","7d","30d"]:
    print(f"    {lab}: {100*treated[f'ret_{lab}'].mean():.1f}%  (n={int(treated[f'ret_{lab}'].sum()):,})")
print("  72h/7d return by first-line class:")
tab = treated.groupby("first_class").agg(n=("CSN","size"), ret72=("ret_72h","mean"), ret7d=("ret_7d","mean"))
tab["ret72"]=(tab["ret72"]*100).round(1); tab["ret7d"]=(tab["ret7d"]*100).round(1)
print(tab.sort_values("n",ascending=False).to_string())

digest = {"n_visits": int(len(v)), "n_headache": int(v["is_ha"].sum()), "n_treated": int(len(treated)),
          "first_class_counts": treated["first_class"].value_counts().to_dict(),
          "ret_rates": {lab: float(treated[f"ret_{lab}"].mean()) for lab in ["72h","7d","30d"]},
          "by_class": {k: {"n": int(r["n"]), "ret72_pct": float(r["ret72"]), "ret7d_pct": float(r["ret7d"])} for k,r in tab.iterrows()}}
os.makedirs(f"{OUT}/intermediate", exist_ok=True)
json.dump(digest, open(f"{OUT}/intermediate/recon_mcmed.json","w"), indent=2)
print("\nwrote", f"{OUT}/intermediate/recon_mcmed.json")
