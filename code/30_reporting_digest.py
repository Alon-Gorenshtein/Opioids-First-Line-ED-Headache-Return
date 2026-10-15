#!/usr/bin/env python3
"""Single source of truth for every crude numerator/denominator quoted in the
manuscript, so no percentage in the text is hand-computed.

MC-MED's ret_30d column is fully populated (0 nulls of 2,486), so the by-class 30-day
keys are computed for both sites rather than skipped. This is data availability, not a
manuscript decision: the supplement's eTable 6 note states the by-class 30-day
breakdown "was computed only for MIMIC-IV-ED" and that only MC-MED's OVERALL 30-day rate
(16.4%, all treated classes combined) is reported in the Results. The mcmed_30d_* keys
below exist here for completeness because the data supports them, not because the
manuscript currently cites them - do not add a new by-class MC-MED 30-day figure to the
text from these keys without a deliberate revision of that eTable 6 note.
"""
import os as _os
import json, pandas as pd

OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))


def nd(df, mask_den, mask_num):
    den = int(mask_den.sum())
    num = int((mask_den & mask_num).sum())
    return {"num": num, "den": den, "pct": round(100 * num / den, 1) if den else None}


m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
s = pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet")

res = {}
for tag, df, has30 in (("mimic", m, True), ("mcmed", s, True)):
    for arm, cls in (("opioid", "opioid"), ("dopa", "dopamine_antagonist")):
        in_cls = df.first_class == cls
        disch = in_cls & df.discharged
        res[f"{tag}_72h_{arm}"] = nd(df, disch, df.ret_72h)
        res[f"{tag}_7d_{arm}"] = nd(df, disch, df.ret_7d)
        if has30:
            res[f"{tag}_30d_{arm}"] = nd(df, disch, df.ret_30d)
        res[f"{tag}_admit_{arm}"] = nd(df, in_cls, df.admitted)

json.dump(res, open(f"{OUT}/results_reporting.json", "w"), indent=2)
for k, v in res.items():
    print(f"{k}: {v['num']:,}/{v['den']:,} ({v['pct']}%)")
