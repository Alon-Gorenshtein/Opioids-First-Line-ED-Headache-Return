#!/usr/bin/env python3
"""Sensitivity: restrict to index visits coded as MIGRAINE ONLY
(ICD-9 346.x; ICD-10 G43.x), excluding tension-type (307.81, G44) and the
nonspecific 784.0 / R51 codes. Run both the all-cause and the headache-specific
return outcome so the phenotype-dilution hypothesis can be answered directly."""
import os as _os
import json, pandas as pd, cer_common as C

OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

def is_migraine(code, ver):
    c = str(code).replace(".", "").upper().strip()
    if str(ver) == "9":
        return c.startswith("346")
    if str(ver) == "10":
        return c.startswith("G43")
    return False

# Load the ED diagnosis table exactly as code/16_sens_primary_headache.py does.
ED = _os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
dg = pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id", "icd_code", "icd_version"])
dg["mig"] = [is_migraine(c, v) for c, v in zip(dg.icd_code, dg.icd_version)]
mig_stays = set(dg.loc[dg.mig, "stay_id"])

m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
mig = m[m.first_class.isin(["opioid", "dopamine_antagonist"]) & m.stay_id.isin(mig_stays)].copy()
mig["opioid"] = (mig.first_class == "opioid").astype(int)
migdis = mig[mig.discharged]

res = {
    "n_migraine": int(len(mig)),
    "n_migraine_discharged": int(len(migdis)),
    "n_migraine_opioid": int(mig.opioid.sum()),
    "n_migraine_opioid_discharged": int(migdis.opioid.sum()),
    "results": (
        [C.analyze(migdis, oc, C.MIMIC_CAT, label=f"migraine-only {oc}")
         for oc in ["ret_72h", "ret_7d", "ret_72h_ha", "ret_7d_ha"]]
        + [C.analyze(mig, "admitted", C.MIMIC_CAT, label="migraine-only admission")]
    ),
}

# Raw unweighted event counts per arm, for outcomes with too few opioid-arm
# events for the overlap-weighted RR alone to convey precision.
res["raw_counts"] = {
    oc: {
        "opioid": {"num": int(pop[pop.opioid == 1][oc].sum()), "den": int((pop.opioid == 1).sum())},
        "dopamine": {"num": int(pop[pop.opioid == 0][oc].sum()), "den": int((pop.opioid == 0).sum())},
    }
    for oc, pop in [("ret_72h", migdis), ("ret_7d", migdis), ("ret_72h_ha", migdis),
                    ("ret_7d_ha", migdis), ("admitted", mig)]
}

json.dump(res, open(f"{OUT}/results_migraine_only.json", "w"), indent=2)
print(f"migraine-only cohort n={res['n_migraine']} (opioid {res['n_migraine_opioid']}), "
      f"discharged n={res['n_migraine_discharged']} (opioid {res['n_migraine_opioid_discharged']})")
for r in res["results"]:
    print(f"  {r['outcome']}: n {r['n']} | opioid n {r['n_opioid']} | RR {r['rr']} {r['rr_ci']} | ESS {r['ess_opioid']}/{r['ess_dopa']}")
for oc, c in res["raw_counts"].items():
    print(f"  raw {oc}: opioid {c['opioid']['num']}/{c['opioid']['den']} | dopamine {c['dopamine']['num']}/{c['dopamine']['den']}")
