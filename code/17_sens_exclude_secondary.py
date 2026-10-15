#!/usr/bin/env python3
"""Sensitivity: EXCLUDE index visits with any secondary-headache / red-flag diagnosis
(intracranial hemorrhage, ischemic stroke/TIA, meningitis/encephalitis, CNS neoplasm, head/face
trauma & intracranial injury, hydrocephalus/raised ICP, temporal arteritis, cerebral venous
thrombosis, eclampsia/pre-eclampsia). Re-run opioid-vs-dopamine on 72h/7d return + admission."""
import os as _os
import json, pandas as pd, cer_common as C
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
ED=_os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
ICD9=("430","431","432","433","434","435","436","437","800","801","802","803","804","850","851","852",
      "853","854","95901","320","321","322","323","191","192","1983","225","3313","3314","3482","3485",
      "4465","3252","4376","64240","64250","64260","64270","6426")
ICD10=("I60","I61","I62","I63","G45","S02","S06","S09","G00","G01","G02","G03","G04","C71","C793","D33",
       "D43","G91","G932","G936","M315","M316","I676","G08","O11","O14","O15")
def secondary(code, ver):
    c=str(code).replace(".","").upper().strip()
    if str(ver)=="9":  return c.startswith(ICD9)
    if str(ver)=="10": return c.startswith(ICD10)
    return False
dg=pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id","icd_code","icd_version"])
dg["sec"]=[secondary(c,v) for c,v in zip(dg.icd_code,dg.icd_version)]
sec_stays=set(dg.loc[dg.sec,"stay_id"])
m=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
m=m[~m.stay_id.isin(sec_stays) & m.first_class.isin(["opioid","dopamine_antagonist"])].copy()
m["opioid"]=(m.first_class=="opioid").astype(int); dis=m[m.discharged]
res={"n_excluded":int(len(sec_stays)),
     "MIMIC_no_secondary":[C.analyze(dis,oc,C.MIMIC_CAT,label=f"no-secondary {oc}") for oc in ["ret_72h","ret_7d"]]}
res["MIMIC_no_secondary"].append(C.analyze(m,"admitted",C.MIMIC_CAT,label="no-secondary admission"))
json.dump(res, open(f"{OUT}/results_exclude_secondary.json","w"), indent=2)
print("index stays carrying a red-flag code:", len(sec_stays))
for r in res["MIMIC_no_secondary"]:
    print(f"{r['outcome']}: n {r['n']} (opioid {r['n_opioid']}) | RR {r['rr']} {r['rr_ci']}")
print("wrote results_exclude_secondary.json")
