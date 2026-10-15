#!/usr/bin/env python3
"""Table 1: characteristics of the treated headache cohort by first-line class (MIMIC-IV-ED),
plus a compact two-center summary row block. Emits markdown + csv for the manuscript.
"""
import os as _os
import numpy as np, pandas as pd, json
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
m=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
CLASSES=["dopamine_antagonist","opioid","nsaid","acetaminophen","magnesium"]
LAB={"dopamine_antagonist":"Dopamine antagonist","opioid":"Opioid","nsaid":"NSAID/ketorolac",
     "acetaminophen":"Acetaminophen","magnesium":"IV magnesium"}
def med_iqr(s): s=s.dropna(); return f"{s.median():.0f} ({s.quantile(.25):.0f}-{s.quantile(.75):.0f})"
def pct(mask,denom): return f"{100*mask.sum()/denom:.1f}"
rows=[]
def add(label, fn):
    rows.append([label]+[fn(m[m.first_class==c]) for c in CLASSES])
add("No. of patients", lambda d: f"{len(d):,}")
add("Age, median (IQR), y", lambda d: med_iqr(d["age"]))
add("Female, %", lambda d: pct(d["gender"].astype(str).str.upper().eq("F"), len(d)))
add("White, %", lambda d: pct(d["race_grp"].eq("White"), len(d)))
add("Black, %", lambda d: pct(d["race_grp"].eq("Black"), len(d)))
add("Hispanic, %", lambda d: pct(d["race_grp"].eq("Hispanic"), len(d)))
add("Arrival by ambulance, %", lambda d: pct(d["arrival_transport"].astype(str).str.upper().str.contains("AMBULANCE"), len(d)))
add("Triage acuity 1-2 (higher), %", lambda d: pct(pd.to_numeric(d["acuity"],errors="coerce").le(2), len(d)))
add("Triage pain, median (IQR)", lambda d: med_iqr(d["pain_num"]))
add("Prior ED visits, median (IQR)", lambda d: med_iqr(d["prior_ed"]))
add("Index admission, %", lambda d: pct(d["admitted"], len(d)))
add("72-hour ED return, %", lambda d: pct(d["ret_72h"], len(d)))
add("7-day ED return, %", lambda d: pct(d["ret_7d"], len(d)))
add("30-day ED return, %", lambda d: pct(d["ret_30d"], len(d)))
tab=pd.DataFrame(rows, columns=["Characteristic"]+[LAB[c] for c in CLASSES])
tab.to_csv(f"{OUT}/tables/table1.csv", index=False)
# markdown
md="| "+" | ".join(tab.columns)+" |\n|"+"|".join(["---"]*len(tab.columns))+"|\n"
for _,r in tab.iterrows(): md+="| "+" | ".join(str(x) for x in r.values)+" |\n"
open(f"{OUT}/tables/table1.md","w").write(md)
print(md)
print("wrote table1.csv / table1.md")
