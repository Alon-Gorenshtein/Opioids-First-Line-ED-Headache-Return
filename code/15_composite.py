#!/usr/bin/env python3
"""Composite outcome = index admission OR 72-hour all-cause ED return, in the FULL treated
opioid-vs-dopamine contrast (no conditioning on discharge). Addresses selection/collider concern
from restricting return analyses to discharged patients."""
import os as _os
import json, cer_common as C
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
p=C.stack_pooled(OUT); p=p[p.first_class.isin(["opioid","dopamine_antagonist"])].copy()
p["opioid"]=(p.first_class=="opioid").astype(int)
p["admit_or_ret72"]=(p["admitted"].astype(bool) | p["ret_72h"].astype(bool)).astype(int)
res={"pooled":[C.analyze(p,"admit_or_ret72",C.POOL_CAT,label="pooled admit-or-72h")],
     "by_site":{s:[C.analyze(p[p.site==s],"admit_or_ret72",C.SITE_CAT,label=f"{s} admit-or-72h")] for s in ["MIMIC","MC-MED"]}}
json.dump(res, open(f"{OUT}/results_composite.json","w"), indent=2)
r=res["pooled"][0]
print(f"composite: opioid {r['risk_opioid']}% vs dopa {r['risk_dopa']}% | RR {r['rr']} {r['rr_ci']} | E {r['evalue']} | n {r['n']}")
print("wrote results_composite.json")
