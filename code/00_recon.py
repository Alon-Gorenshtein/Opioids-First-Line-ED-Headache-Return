#!/usr/bin/env python3
"""Phase-1 reconnaissance for the ED headache treatment -> return-visit study (MIMIC-IV-ED).
Answers the feasibility questions before any modeling:
  - headache cohort size, treated fraction
  - FIRST-LINE treatment-class distribution (esp. opioid vs dopamine-antagonist counts)
  - index disposition (admit vs discharge)
  - 72h / 7d / 30d ED return-visit rates, overall and by first-line class
Memory-smart: ED tables are small; pyxis is 16 MB gz. Nothing streamed here.
"""
import os as _os
import re, gzip, json
import pandas as pd, numpy as np

ED = _os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

CC_RE = re.compile(r"\b(headache|head ache|migraine|cephalgia|cephalalgia)\b")
def headache_icd(code, ver):
    c = str(code).replace(".", "").upper().strip()
    if str(ver) == "9":  return c.startswith(("346", "30781", "339")) or c == "7840"
    if str(ver) == "10": return c.startswith(("G43", "G44", "R51"))
    return False

DRUG_CLASSES = {
 "opioid": r"morphine|hydromorphone|dilaudid|oxycodone|hydrocodone|fentanyl|codeine|tramadol|oxymorphone|meperidine|demerol|percocet|butorphanol|nalbuphine|tapentadol",
 "dopamine_antagonist": r"metoclopramide|reglan|prochlorperazine|compazine|chlorpromazine|droperidol|promethazine|phenergan",
 "nsaid": r"ketorolac|toradol|ibuprofen|naproxen|ketoprofen|diclofenac|indomethacin|aspirin",
 "triptan": r"sumatriptan|rizatriptan|zolmitriptan|eletriptan|naratriptan|frovatriptan|almotriptan|imitrex|maxalt|zomig",
 "dhe": r"dihydroergotamine|d\.h\.e|dhe 45|migranal",
 "magnesium": r"magnesium",
 "acetaminophen": r"acetaminophen|tylenol|ofirmev|paracetamol",
}
# priority order for first-line assignment when several drugs share the earliest charttime
CLASS_ORDER = ["opioid","dopamine_antagonist","nsaid","triptan","dhe","magnesium","acetaminophen"]
COMPILED = {k: re.compile(v, re.I) for k, v in DRUG_CLASSES.items()}
def classify(name):
    n = str(name).lower()
    for k in CLASS_ORDER:  # opioid first: a combo containing an opioid counts as opioid
        if COMPILED[k].search(n):
            return k
    return None

print("loading ED tables...")
edstays = pd.read_csv(f"{ED}/edstays.csv.gz", parse_dates=["intime","outtime"])
triage  = pd.read_csv(f"{ED}/triage.csv.gz", usecols=["stay_id","chiefcomplaint","acuity","pain"])
diag    = pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id","icd_code","icd_version"])

# ---- headache identification ----
triage["cc"] = triage["chiefcomplaint"].fillna("").str.lower()
cc_hits = set(triage.loc[triage["cc"].str.contains(CC_RE), "stay_id"])
diag["is_ha"] = [headache_icd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
icd_hits = set(diag.loc[diag["is_ha"], "stay_id"])
ha_stays = cc_hits | icd_hits
edstays["is_ha"] = edstays["stay_id"].isin(ha_stays)
print(f"  total ED stays: {len(edstays):,}")
print(f"  headache stays (CC or ICD): {edstays['is_ha'].sum():,}  (CC={len(cc_hits):,}, ICD={len(icd_hits):,})")

# ---- adults only (age via edstays? MIMIC-IV-ED has no age; all are adults in ED module? use anchor)
# MIMIC-IV-ED patients are adults; skip age gate here (documented in prior study).

# ---- first-line treatment class from pyxis ----
print("loading pyxis...")
pyx = pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id","charttime","name"], parse_dates=["charttime"])
pyx = pyx[pyx["stay_id"].isin(ha_stays)].copy()
pyx["cls"] = pyx["name"].map(classify)
pyx = pyx.dropna(subset=["cls"])
# earliest qualifying administration per stay; tie-break by CLASS_ORDER
pyx["ord"] = pyx["cls"].map({c:i for i,c in enumerate(CLASS_ORDER)})
pyx = pyx.sort_values(["stay_id","charttime","ord"])
first_line = pyx.groupby("stay_id").first()[["cls","charttime"]].rename(columns={"cls":"first_class","charttime":"tx_time"})
edstays = edstays.merge(first_line, left_on="stay_id", right_index=True, how="left")
edstays["treated"] = edstays["first_class"].notna()

ha = edstays[edstays["is_ha"]].copy()
treated = ha[ha["treated"]].copy()
print(f"  headache treated (>=1 qualifying agent): {len(treated):,} / {len(ha):,} ({100*len(treated)/len(ha):.1f}%)")
print("  first-line class distribution (treated):")
print(treated["first_class"].value_counts().to_string())

# ---- index disposition ----
print("  index disposition (treated):")
print(treated["disposition"].value_counts().to_string())

# ---- return visits: next edstay for same subject within windows ----
edstays = edstays.sort_values(["subject_id","intime"])
edstays["next_intime"] = edstays.groupby("subject_id")["intime"].shift(-1)
# gap from THIS stay's outtime to NEXT stay's intime
edstays["gap_h"] = (edstays["next_intime"] - edstays["outtime"]).dt.total_seconds()/3600.0
ha = edstays[edstays["is_ha"] & edstays["treated"]].copy()
for w,lab in [(72,"72h"),(168,"7d"),(720,"30d")]:
    ha[f"ret_{lab}"] = (ha["gap_h"] > 0) & (ha["gap_h"] <= w)
print("  RETURN-VISIT rates among treated headache index visits:")
for lab in ["72h","7d","30d"]:
    print(f"    {lab}: {100*ha[f'ret_{lab}'].mean():.1f}%  (n_ret={int(ha[f'ret_{lab}'].sum()):,})")

# ---- return by first-line class (the key contrast) ----
print("  72h return by first-line class:")
tab = ha.groupby("first_class").agg(n=("stay_id","size"), ret72=("ret_72h","mean"), ret7d=("ret_7d","mean"),
                                    admit=("disposition", lambda s:(s.astype(str).str.upper().str.contains("ADMIT")).mean()))
tab["ret72"]=(tab["ret72"]*100).round(1); tab["ret7d"]=(tab["ret7d"]*100).round(1); tab["admit"]=(tab["admit"]*100).round(1)
print(tab.sort_values("n",ascending=False).to_string())

digest = {
 "n_ed_stays": int(len(edstays)),
 "n_headache": int(edstays["is_ha"].sum()),
 "n_treated": int(len(treated)),
 "first_class_counts": treated["first_class"].value_counts().to_dict(),
 "ret_rates": {lab: float(ha[f"ret_{lab}"].mean()) for lab in ["72h","7d","30d"]},
 "by_class_72h": {k: {"n": int(v["n"]), "ret72_pct": float(v["ret72"]), "ret7d_pct": float(v["ret7d"]), "admit_pct": float(v["admit"])}
                  for k,v in tab.iterrows()},
}
import os; os.makedirs(f"{OUT}/intermediate", exist_ok=True)
json.dump(digest, open(f"{OUT}/intermediate/recon_mimic.json","w"), indent=2)
print("\nwrote", f"{OUT}/intermediate/recon_mimic.json")
