#!/usr/bin/env python3
"""Table 1: characteristics of the treated headache cohort by first-line class (MC-MED),
the external-validation counterpart to code/07_table1.py (MIMIC-IV-ED). Row labels and
row order are kept identical to table1.csv so the two tables read as a pair.
Emits markdown + csv for the manuscript.

Two fields need site-specific handling, not a literal copy of the MIMIC logic, because
MC-MED encodes them differently:
  - arrival_transport: MC-MED's Means_of_arrival values are "Self"/"EMS"/"Other/Unknown",
    not MIMIC-IV-ED's "AMBULANCE" string, so the ambulance flag is matched on "EMS".
  - acuity: MC-MED stores it as a labeled string ("1-Resuscitation".."5-Non-Urgent",
    "missing"), not MIMIC-IV-ED's numeric 1-5, so the leading digit is extracted first.
No row is dropped: table1.csv's 14 rows never reference yeargrp, t_ret_d, or died_30d
(the three columns MC-MED's parquet lacks), so every row is constructible as-is.
"""
import os as _os
import numpy as np, pandas as pd, json
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
m=pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet")
CLASSES=["dopamine_antagonist","opioid","nsaid","acetaminophen","magnesium"]
LAB={"dopamine_antagonist":"Dopamine antagonist","opioid":"Opioid","nsaid":"NSAID/ketorolac",
     "acetaminophen":"Acetaminophen","magnesium":"IV magnesium"}
def med_iqr(s): s=s.dropna(); return f"{s.median():.0f} ({s.quantile(.25):.0f}-{s.quantile(.75):.0f})"
def pct(mask,denom): return f"{100*mask.sum()/denom:.1f}"
def acuity_num(s):
    return pd.to_numeric(s.astype(str).str.extract(r"^(\d)")[0], errors="coerce")
rows=[]
def add(label, fn):
    rows.append([label]+[fn(m[m.first_class==c]) for c in CLASSES])
add("No. of patients", lambda d: f"{len(d):,}")
add("Age, median (IQR), y", lambda d: med_iqr(d["age"]))
add("Female, %", lambda d: pct(d["gender"].astype(str).str.upper().eq("F"), len(d)))
add("White, %", lambda d: pct(d["race_grp"].eq("White"), len(d)))
add("Black, %", lambda d: pct(d["race_grp"].eq("Black"), len(d)))
add("Hispanic, %", lambda d: pct(d["race_grp"].eq("Hispanic"), len(d)))
add("Arrival by ambulance, %", lambda d: pct(d["arrival_transport"].astype(str).str.upper().str.contains("EMS"), len(d)))
add("Triage acuity 1-2 (higher), %", lambda d: pct(acuity_num(d["acuity"]).le(2), len(d)))
add("Triage pain, median (IQR)", lambda d: med_iqr(d["pain_num"]))
add("Prior ED visits, median (IQR)", lambda d: med_iqr(d["prior_ed"]))
add("Index admission, %", lambda d: pct(d["admitted"], len(d)))
add("72-hour ED return, %", lambda d: pct(d["ret_72h"], len(d)))
add("7-day ED return, %", lambda d: pct(d["ret_7d"], len(d)))
add("30-day ED return, %", lambda d: pct(d["ret_30d"], len(d)))
tab=pd.DataFrame(rows, columns=["Characteristic"]+[LAB[c] for c in CLASSES])
tab.to_csv(f"{OUT}/tables/table1_mcmed.csv", index=False)
# markdown
md="| "+" | ".join(tab.columns)+" |\n|"+"|".join(["---"]*len(tab.columns))+"|\n"
for _,r in tab.iterrows(): md+="| "+" | ".join(str(x) for x in r.values)+" |\n"
open(f"{OUT}/tables/table1_mcmed.md","w").write(md)
print(md)
print("wrote table1_mcmed.csv / table1_mcmed.md")
