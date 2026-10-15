#!/usr/bin/env python3
"""Phase-4 analytic cohort (MIMIC-IV-ED primary).
Assembles treated adult headache index visits with covariates + outcomes, saves parquet.
Covariates (propensity): age, sex, race, triage acuity, triage pain, arrival mode,
prior ED visits, arrival shift, calendar-year group.
Outcomes: index disposition (admit vs discharge), 72h/7d/30d ED return (among discharged),
30-day death (competing risk).
"""
import os as _os
import re, json, os
import pandas as pd, numpy as np

ED   = _os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
HOSP = _os.environ.get("MIMIC_IV_HOSP_DIR", "mimic-iv-3.1/hosp")
OUT  = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

CC_RE = re.compile(r"(headache|head ache|migraine|cephalgia|cephalalgia)", re.I)
def headache_icd(code, ver):
    c = str(code).replace(".", "").upper().strip()
    if str(ver) == "9":  return c.startswith(("346","30781","339")) or c == "7840"
    if str(ver) == "10": return c.startswith(("G43","G44","R51"))
    return False
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
        if COMPILED[k].search(n): return k
    return None
def race_group(r):
    r = str(r).upper()
    if "WHITE" in r: return "White"
    if "BLACK" in r: return "Black"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic"
    if "ASIAN" in r: return "Asian"
    return "Other/Unknown"
def shift(h):
    if 7 <= h < 15: return "day"
    if 15 <= h < 23: return "evening"
    return "night"

print("loading...")
ed = pd.read_csv(f"{ED}/edstays.csv.gz", parse_dates=["intime","outtime"])
tri = pd.read_csv(f"{ED}/triage.csv.gz", usecols=["stay_id","chiefcomplaint","acuity","pain"])
dg  = pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id","icd_code","icd_version"])
pat = pd.read_csv(f"{HOSP}/patients.csv.gz", usecols=["subject_id","anchor_age","anchor_year","dod"], parse_dates=["dod"])  # gender comes from edstays

# headache flag
tri["cc"] = tri["chiefcomplaint"].fillna("").astype(str)
cc_hits = set(tri.loc[tri["cc"].str.contains(CC_RE), "stay_id"])
dg["ha"] = [headache_icd(c,v) for c,v in zip(dg["icd_code"], dg["icd_version"])]
ha_stays = cc_hits | set(dg.loc[dg["ha"],"stay_id"])
ed["is_ha"] = ed["stay_id"].isin(ha_stays)

# first-line class
pyx = pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id","charttime","name"], parse_dates=["charttime"])
pyx = pyx[pyx["stay_id"].isin(ha_stays)].copy()
pyx["cls"] = pyx["name"].map(classify); pyx = pyx.dropna(subset=["cls"])
pyx["ord"] = pyx["cls"].map({c:i for i,c in enumerate(CLASS_ORDER)})
pyx = pyx.sort_values(["stay_id","charttime","ord"])
fl = pyx.groupby("stay_id").first()[["cls","charttime"]].rename(columns={"cls":"first_class","charttime":"tx_time"})
ed = ed.merge(fl, left_on="stay_id", right_index=True, how="left")
ed["treated"] = ed["first_class"].notna()

# prior ED visits + returns (all stays, per subject)
ed = ed.sort_values(["subject_id","intime"])
ed["prior_ed"] = ed.groupby("subject_id").cumcount()
ed["next_intime"] = ed.groupby("subject_id")["intime"].shift(-1)
ed["gap_h"] = (ed["next_intime"] - ed["outtime"]).dt.total_seconds()/3600.0
ed["next_is_ha"] = ed.groupby("subject_id")["is_ha"].shift(-1)   # was the immediate next ED visit a headache visit?

# merge covariates
ed = ed.merge(tri[["stay_id","acuity","pain"]], on="stay_id", how="left")
ed = ed.merge(pat, on="subject_id", how="left")
ed["age"] = ed["anchor_age"] + (ed["intime"].dt.year - ed["anchor_year"])
ed["race_grp"] = ed["race"].map(race_group)
ed["shift"] = ed["intime"].dt.hour.map(shift)
ed["yeargrp"] = ((ed["intime"].dt.year//3)*3).astype(str)
ed["pain_num"] = pd.to_numeric(ed["pain"], errors="coerce").clip(0,10)
ed["admitted"] = ed["disposition"].astype(str).str.upper().str.contains("ADMIT")
ed["discharged"] = ed["disposition"].astype(str).str.upper().eq("HOME")
# 30-day death after index outtime
ed["died_30d"] = (ed["dod"].notna()) & ((ed["dod"] - ed["outtime"]).dt.total_seconds()/86400.0).between(0,30)
for w,lab in [(72,"72h"),(168,"7d"),(720,"30d")]:
    ed[f"ret_{lab}"] = (ed["gap_h"]>0) & (ed["gap_h"]<=w)
for w,lab in [(72,"72h"),(168,"7d")]:
    ed[f"ret_{lab}_ha"] = ed[f"ret_{lab}"] & ed["next_is_ha"].fillna(False).astype(bool)
# time to return (days) capped at 30 for competing-risks; event code: 1=return, 2=death, 0=censored
ed["t_ret_d"] = np.where(ed["ret_30d"], ed["gap_h"]/24.0, np.nan)

# analytic cohort: treated adult headache index visits
coh = ed[ed["is_ha"] & ed["treated"] & (ed["age"]>=18)].copy()
# index visit = first treated headache visit per subject (new-user).
# .nth(0), not .first(): GroupBy.first() returns the first NON-NULL value of each column
# independently, so a patient with more than one treated headache visit would inherit a
# later visit's value into any column that is null at the index visit (t_ret_d is null
# by construction whenever the index visit had no 30-day return, and acuity/pain_num are
# missing for some triage records). .nth(0) selects the actual first row of each group.
coh = coh.sort_values(["subject_id","intime"]).groupby("subject_id", as_index=False).nth(0)
print(f"analytic cohort (treated adult headache, first per pt): {len(coh):,}")
print("  class counts:\n", coh["first_class"].value_counts().to_string())
print(f"  discharged home: {coh['discharged'].sum():,}  admitted: {coh['admitted'].sum():,}")
print(f"  30d death: {int(coh['died_30d'].sum())}")

os.makedirs(f"{OUT}/intermediate", exist_ok=True)
keep = ["subject_id","stay_id","intime","outtime","first_class","tx_time","age","gender","race_grp",
        "acuity","pain_num","arrival_transport","prior_ed","shift","yeargrp","admitted","discharged",
        "died_30d","gap_h","t_ret_d","ret_72h","ret_7d","ret_30d","ret_72h_ha","ret_7d_ha"]
coh[keep].to_parquet(f"{OUT}/intermediate/cohort_mimic.parquet", index=False)
print("wrote cohort_mimic.parquet")
