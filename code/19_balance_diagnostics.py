#!/usr/bin/env python3
"""Balance diagnostics + Kish effective sample size for every analytic population, so the
manuscript's balance claim is scoped correctly (not presented as universal). Uses cer_common."""
import os as _os
import json, cer_common as C
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
p=C.stack_pooled(OUT); p=p[p.first_class.isin(["opioid","dopamine_antagonist"])].copy()
p["opioid"]=(p.first_class=="opioid").astype(int); dis=p[p.discharged]
POP=[("Pooled, discharged (72h return)", dis, "ret_72h", C.POOL_CAT),
     ("MIMIC-IV-ED, discharged (72h return)", dis[dis.site=="MIMIC"], "ret_72h", C.SITE_CAT),
     ("MC-MED, discharged (72h return)", dis[dis.site=="MC-MED"], "ret_72h", C.SITE_CAT),
     ("Pooled, full cohort (admission)", p, "admitted", C.POOL_CAT),
     ("MIMIC-IV-ED, full cohort (admission)", p[p.site=="MIMIC"], "admitted", C.SITE_CAT),
     ("MC-MED, full cohort (admission)", p[p.site=="MC-MED"], "admitted", C.SITE_CAT)]
rows=[]; digest=[]
for name,pop,oc,cat in POP:
    r=C.analyze(pop, oc, cat, label=name)
    rows.append((name, r["n"], r["n_opioid"], r["smd_pre"], r["smd_post"], r["ess_opioid"], r["ess_dopa"]))
    digest.append(r)
json.dump(digest, open(f"{OUT}/results_balance_all.json","w"), indent=2)
md="| Analytic population | n | Opioid n | Max SMD before | Max SMD after | Effective n, opioid | Effective n, dopamine |\n|---|---:|---:|---:|---:|---:|---:|\n"
for nm,n,no,pre,post,eo,ed_ in rows: md+=f"| {nm} | {n:,} | {no:,} | {pre:.3f} | {post:.3f} | {eo:,.0f} | {ed_:,.0f} |\n"
open(f"{OUT}/tables/etable_balance_all.md","w").write(md)
print(md); print("wrote results_balance_all.json + etable_balance_all.md")
