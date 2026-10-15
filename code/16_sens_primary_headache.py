#!/usr/bin/env python3
"""Sensitivity: restrict to index visits with a MIGRAINE / PRIMARY-headache code
(ICD-9 346.x, 307.81, 339.x ; ICD-10 G43, G44) — excludes the nonspecific 784.0 / R51 codes and
chief-complaint-only cases. Re-run opioid-vs-dopamine on 72h/7d all-cause return (discharged)."""
import os as _os
import json, re, pandas as pd, cer_common as C
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
ED=_os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
def primary_ha(code, ver):
    c=str(code).replace(".","").upper().strip()
    if str(ver)=="9":  return c.startswith(("346","30781","339"))
    if str(ver)=="10": return c.startswith(("G43","G44"))
    return False
dg=pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id","icd_code","icd_version"])
dg["pri"]=[primary_ha(c,v) for c,v in zip(dg.icd_code,dg.icd_version)]
pri_stays=set(dg.loc[dg.pri,"stay_id"])
m=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
m=m[m.stay_id.isin(pri_stays) & m.first_class.isin(["opioid","dopamine_antagonist"])].copy()
m["opioid"]=(m.first_class=="opioid").astype(int); dis=m[m.discharged]
res={"MIMIC_primary_headache":[C.analyze(dis,oc,C.MIMIC_CAT,label=f"primary-ha {oc}") for oc in ["ret_72h","ret_7d"]]}
res["MIMIC_primary_headache"].append(C.analyze(m,"admitted",C.MIMIC_CAT,label="primary-ha admission"))
json.dump(res, open(f"{OUT}/results_primary_headache.json","w"), indent=2)
for r in res["MIMIC_primary_headache"]:
    print(f"{r['outcome']}: n {r['n']} (opioid {r['n_opioid']}) | RR {r['rr']} {r['rr_ci']}")
print("wrote results_primary_headache.json")
