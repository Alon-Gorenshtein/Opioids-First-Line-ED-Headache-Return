#!/usr/bin/env python3
"""Sensitivity: headache-specific short-term return (next ED visit was itself a headache visit),
opioid vs dopamine antagonist, among discharged patients. Pooled two-center + per site."""
import os as _os
import json, cer_common as C
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
p=C.stack_pooled(OUT); p=p[p.first_class.isin(["opioid","dopamine_antagonist"])].copy()
p["opioid"]=(p.first_class=="opioid").astype(int); dis=p[p.discharged].copy()
res={"pooled":[], "by_site":{}}
for oc in ["ret_72h_ha","ret_7d_ha"]:
    res["pooled"].append(C.analyze(dis, oc, C.POOL_CAT, label=f"pooled {oc}"))
for s in ["MIMIC","MC-MED"]:
    ds=dis[dis.site==s]
    res["by_site"][s]=[C.analyze(ds, oc, C.SITE_CAT, label=f"{s} {oc}") for oc in ["ret_72h_ha","ret_7d_ha"]]
json.dump(res, open(f"{OUT}/results_ha_return.json","w"), indent=2)
for r in res["pooled"]:
    print(f"{r['outcome']}: opioid {r['risk_opioid']}% vs dopa {r['risk_dopa']}% | RR {r['rr']} {r['rr_ci']} | E {r['evalue']}")
print("wrote results_ha_return.json")
