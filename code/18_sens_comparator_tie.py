#!/usr/bin/env python3
"""Sensitivity: (a) restrict the dopamine-antagonist comparator to metoclopramide/prochlorperazine/
droperidol (drop promethazine, chlorpromazine) — recompute first-line among headache index visits;
(b) alternative tie rule: exclude index visits whose earliest administration was an opioid+other tie.
Both re-run the MIMIC primary 72h/7d return + admission contrast. Requires re-reading pyxis to
recover per-agent timing for the tie rule; the comparator restriction re-labels first_class."""
import os as _os
import json, re, pandas as pd, cer_common as C
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
ED=_os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
# --- (a) restricted comparator: relabel dopamine only if the first-line agent is one of the 3 core agents
CORE=re.compile(r"metoclopramide|reglan|prochlorperazine|compazine|droperidol", re.I)
OPIOID=re.compile(r"morphine|hydromorphone|dilaudid|oxycodone|hydrocodone|fentanyl|codeine|tramadol|oxymorphone|meperidine|demerol|percocet|butorphanol|nalbuphine|tapentadol", re.I)
m=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
pyx=pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id","charttime","name"], parse_dates=["charttime"])
pyx=pyx[pyx.stay_id.isin(set(m.stay_id))].copy().sort_values(["stay_id","charttime"])
# The first-line agent is the earliest *qualifying acute headache agent*, not the first
# dispense of any drug. Rebuild the same way code/23_constituent_agents.py does, so the
# restricted comparator reconciles with eTable 15.
QUALIFY = re.compile(
    r"metoclopramide|reglan|prochlorperazine|compazine|droperidol|chlorpromazine|"
    r"promethazine|phenergan|"
    r"morphine|hydromorphone|dilaudid|oxycodone|hydrocodone|fentanyl|codeine|tramadol|"
    r"oxymorphone|meperidine|demerol|percocet|butorphanol|nalbuphine|tapentadol",
    re.I)
q = pyx[pyx.name.astype(str).str.contains(QUALIFY)].sort_values(["stay_id", "charttime"])
first_qual = q.groupby("stay_id").first()["name"].astype(str)
core_dopa = set(first_qual[first_qual.str.contains(CORE)].index)
# (a) keep opioid rows as-is; keep dopamine rows only if earliest agent is a core antimigraine agent
a=m[m.first_class.isin(["opioid","dopamine_antagonist"])].copy()
a=a[(a.first_class=="opioid") | (a.stay_id.isin(core_dopa))]
a["opioid"]=(a.first_class=="opioid").astype(int); adis=a[a.discharged]
# (b) tie rule: flag stays whose earliest timestamp had an opioid co-administered with another class
t0=pyx.merge(pyx.groupby("stay_id").charttime.min().rename("t0"), on="stay_id")
tied=t0[t0.charttime==t0.t0]
n_per=tied.groupby("stay_id").name.size(); has_op=tied.assign(op=tied.name.str.contains(OPIOID)).groupby("stay_id").op.max()
tie_op=set(n_per[(n_per>1)].index) & set(has_op[has_op].index)
b=m[m.first_class.isin(["opioid","dopamine_antagonist"]) & ~m.stay_id.isin(tie_op)].copy()
b["opioid"]=(b.first_class=="opioid").astype(int); bdis=b[b.discharged]
res={"restricted_comparator":[C.analyze(adis,oc,C.MIMIC_CAT,label=f"restrict-comp {oc}") for oc in ["ret_72h","ret_7d"]]+[C.analyze(a,"admitted",C.MIMIC_CAT,label="restrict-comp admission")],
     "drop_opioid_ties":[C.analyze(bdis,oc,C.MIMIC_CAT,label=f"drop-ties {oc}") for oc in ["ret_72h","ret_7d"]]+[C.analyze(b,"admitted",C.MIMIC_CAT,label="drop-ties admission")],
     "n_restricted_dopa": int((a.first_class == "dopamine_antagonist").sum()),
     "n_restricted_dopa_discharged": int(((a.first_class == "dopamine_antagonist") & a.discharged).sum()),
     "n_core_dopa_any_stay": len(core_dopa),
     "n_tie_opioid_dropped":len(tie_op)}
json.dump(res, open(f"{OUT}/results_comparator_tie.json","w"), indent=2)
print("core-antimigraine dopamine index visits:",len(core_dopa),"| opioid-tie index visits dropped:",len(tie_op))
for k in ["restricted_comparator","drop_opioid_ties"]:
    for r in res[k]: print(f"  {k} {r['outcome']}: n {r['n']} | RR {r['rr']} {r['rr_ci']}")
print("wrote results_comparator_tie.json")
