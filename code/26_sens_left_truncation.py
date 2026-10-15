#!/usr/bin/env python3
"""Sensitivity: left truncation and the new-user/washout definition.

A patient's first *observed* treated headache visit need
not be their first *ever*: MIMIC-IV-ED opens at a fixed point, so a patient already
established on opioids before that point is misclassified as a new presenter, and the
chronic-migraine / medication-overuse phenotype most likely to be left-truncated is
also the phenotype most prone to both opioid receipt and return.

A calendar cutoff cannot answer this in MIMIC-IV. Every patient's dates are shifted
into the future by a random *per-patient* offset, so `intime` in this cohort runs from
2110-01-15 to 2210-12-23 (a 100.9-year span for care actually delivered 2011-2019) and
`yeargrp` spans 2109 to 2208. Calendar dates are comparable within a patient and
meaningless across patients; a fixed "database start + 365 days" rule would select
patients on their de-identification offset, not on their observable history.

Intervals *within* a patient survive the shift, so the lookback is built from the
patient's own record instead: lookback_days = index visit intime minus that same
patient's earliest ED intime anywhere in the full edstays table. Patients with
lookback_days >= 365 have a year of observed ED history in this system with no prior
treated headache visit (the cohort already takes each patient's first treated headache
visit), which makes them confirmed incident presenters rather than assumed ones.

Limitation, stated in the manuscript and not hidden here: this confirms incidence only
for patients who had some earlier ED contact in this system. A patient with no prior
encounter is indistinguishable between "genuinely their first headache" and "previously
treated somewhere we cannot see". Requiring a lookback therefore selects toward prior ED
users, who are a different population from the full cohort.

MIMIC-IV-ED only; MC-MED lacks the linked longitudinal history needed to define the
same window.
"""
import os as _os
import json
import pandas as pd
import cer_common as C

ED = _os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
THRESHOLDS = [365, 180]          # primary cut first, secondary second
REPORT_AT = [0, 90, 180, 365]

DESIGN_NOTE = (
    "A calendar-date washout is invalid in MIMIC-IV: dates are shifted into the future by a "
    "random per-patient offset, so index intime spans 2110-2210 for care delivered 2011-2019 "
    "and cross-patient calendar comparisons are meaningless. Because within-patient intervals "
    "are preserved, the lookback is defined per patient as (index visit intime) - (that "
    "patient's earliest ED intime in the full edstays table). Patients whose index visit is "
    "their first recorded ED encounter have lookback_days = 0. Restricting to a lookback "
    "confirms incident status only among patients with prior ED contact in this system, and "
    "therefore selects toward prior ED users."
)

m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
d = m[m.first_class.isin(["opioid", "dopamine_antagonist"])].copy()
d["opioid"] = (d.first_class == "opioid").astype(int)

# Earliest ED encounter per patient across ALL of MIMIC-IV-ED, not just the headache cohort.
es = pd.read_csv(f"{ED}/edstays.csv.gz", usecols=["subject_id", "intime"], parse_dates=["intime"])
first_ed = es.groupby("subject_id")["intime"].min()
d["lookback_days"] = (d.intime - d.subject_id.map(first_ed)).dt.total_seconds() / 86400.0
assert d.lookback_days.notna().all(), "cohort subject missing from edstays"
assert (d.lookback_days >= 0).all(), "negative lookback: index visit precedes the patient's first ED stay"

# Cross-check: prior_ed is a cumcount over all ED stays ordered by intime, so
# prior_ed == 0 and lookback_days == 0 must identify exactly the same patients.
zero_lb = d.lookback_days == 0
zero_pe = d.prior_ed == 0
n_disagree = int((zero_lb != zero_pe).sum())

lb = d.lookback_days
dist = {
    "median": round(float(lb.median()), 1),
    "q25": round(float(lb.quantile(0.25)), 1),
    "q75": round(float(lb.quantile(0.75)), 1),
    "max": round(float(lb.max()), 1),
    "n_zero": int(zero_lb.sum()),
    "pct_zero": round(100 * float(zero_lb.mean()), 1),
    "n_at_least": {str(t): int((lb >= t).sum()) for t in REPORT_AT},
    # Among patients who do have prior ED contact, what does the window look like?
    "median_among_nonzero": round(float(lb[lb > 0].median()), 1),
    "q25_among_nonzero": round(float(lb[lb > 0].quantile(0.25)), 1),
    "q75_among_nonzero": round(float(lb[lb > 0].quantile(0.75)), 1),
    # By arm: is the truncated group the opioid-prone one?
    "by_arm": {
        arm: {
            "n": int((d.opioid == v).sum()),
            "median": round(float(lb[d.opioid == v].median()), 1),
            "q25": round(float(lb[d.opioid == v].quantile(0.25)), 1),
            "q75": round(float(lb[d.opioid == v].quantile(0.75)), 1),
            "pct_zero": round(100 * float(zero_lb[d.opioid == v].mean()), 1),
            "pct_ge_365": round(100 * float((lb[d.opioid == v] >= 365).mean()), 1),
        }
        for arm, v in [("opioid", 1), ("dopamine_antagonist", 0)]
    },
}


def run(pop, label_prefix):
    """72h/7d among discharged, admission in the full population -- same estimator as primary."""
    dis = pop[pop.discharged]
    return ([C.analyze(dis, oc, C.MIMIC_CAT, label=f"{label_prefix} {oc}") for oc in ["ret_72h", "ret_7d"]]
            + [C.analyze(pop, "admitted", C.MIMIC_CAT, label=f"{label_prefix} admission")])


# Unrestricted MIMIC reference, recomputed here so the comparison is estimator-identical.
print("unrestricted MIMIC reference...")
unrestricted = run(d, "unrestricted MIMIC")

by_threshold = {}
for t in THRESHOLDS:
    kept = d[d.lookback_days >= t].copy()
    dis = kept[kept.discharged]
    print(f"lookback >= {t} d: n={len(kept)} (opioid {int(kept.opioid.sum())}), discharged {len(dis)}...")
    by_threshold[str(t)] = {
        "threshold_days": t,
        "n_before_restriction": int(len(d)),
        "n_dropped": int(len(d) - len(kept)),
        "n_retained": int(len(kept)),
        "n_retained_opioid": int(kept.opioid.sum()),
        "n_retained_dopamine": int((1 - kept.opioid).sum()),
        "n_retained_discharged": int(len(dis)),
        "n_retained_discharged_opioid": int(dis.opioid.sum()),
        "raw_counts": {
            oc: {
                "opioid": {"num": int(dis[dis.opioid == 1][oc].sum()), "den": int((dis.opioid == 1).sum())},
                "dopamine": {"num": int(dis[dis.opioid == 0][oc].sum()), "den": int((dis.opioid == 0).sum())},
            } for oc in ["ret_72h", "ret_7d"]
        } | {
            "admitted": {
                "opioid": {"num": int(kept[kept.opioid == 1].admitted.sum()), "den": int((kept.opioid == 1).sum())},
                "dopamine": {"num": int(kept[kept.opioid == 0].admitted.sum()), "den": int((kept.opioid == 0).sum())},
            }
        },
        "results": run(kept, f"lookback>={t}d"),
    }

# The 365-day cut leaves few events, so the bootstrap CI is the fragile part of this
# analysis. Re-run the primary outcome across seeds: the point estimate is deterministic,
# so any movement here is Monte Carlo noise in the interval, and the reader needs to know
# how much of it there is before treating "CI now includes 1" as a finding.
print("seed stability for the 365 d primary outcome...")
_k365 = d[d.lookback_days >= 365]
_dis365 = _k365[_k365.discharged]
_seeds = [20260707, 11, 202, 3003, 40404]
_stab = [C.analyze(_dis365, "ret_72h", C.MIMIC_CAT, seed=s, label="seedcheck") for s in _seeds]
_los = [r["rr_ci"][0] for r in _stab]
_his = [r["rr_ci"][1] for r in _stab]
seed_stability_365_72h = {
    "n_seeds": len(_seeds),
    "rr_point_values": sorted({r["rr"] for r in _stab}),
    "rr_point_deterministic": len({r["rr"] for r in _stab}) == 1,
    "ci_lower_range": [round(min(_los), 3), round(max(_los), 3)],
    "ci_upper_range": [round(min(_his), 3), round(max(_his), 3)],
    "all_seeds_ci_include_null": all(lo <= 1.0 <= hi for lo, hi in zip(_los, _his)),
    "note": ("The point estimate does not depend on the bootstrap seed; only the interval does. "
             "With 11 opioid events the interval is wide under every seed, so this cut cannot "
             "confirm or refute the primary estimate."),
}

res = {
    "design_note": DESIGN_NOTE,
    "date_shift_evidence": {
        "cohort_intime_min": str(d.intime.min()),
        "cohort_intime_max": str(d.intime.max()),
        "cohort_intime_span_years": round((d.intime.max() - d.intime.min()).total_seconds() / (86400 * 365.25), 1),
        "yeargrp_min": str(m.yeargrp.min()),
        "yeargrp_max": str(m.yeargrp.max()),
        "true_collection_window": "2011-2019",
        "note": "A >100-year apparent span for a 9-year collection window is the per-patient date shift.",
    },
    "n_contrast_cohort": int(len(d)),
    "lookback_distribution": dist,
    "prior_ed_crosscheck": {
        "n_lookback_zero": int(zero_lb.sum()),
        "n_prior_ed_zero": int(zero_pe.sum()),
        "n_disagreeing": n_disagree,
        "agree": n_disagree == 0,
        "note": ("prior_ed is a per-subject cumulative count of earlier ED stays over the same "
                 "edstays table, so exact agreement is the expected result and any disagreement "
                 "would mean one of the two derivations is wrong."),
    },
    "unrestricted_reference": unrestricted,
    "by_threshold": by_threshold,
    "primary_threshold_days": THRESHOLDS[0],
    "seed_stability_365_72h": seed_stability_365_72h,
}
json.dump(res, open(f"{OUT}/results_left_truncation.json", "w"), indent=2)

print(f"\ncontrast cohort n={len(d)} | lookback median {dist['median']} d "
      f"(IQR {dist['q25']}-{dist['q75']}) | zero lookback {dist['n_zero']} ({dist['pct_zero']}%)")
print(f"prior_ed cross-check: {dist['n_zero']} vs {int(zero_pe.sum())}, disagreeing {n_disagree}")
print("  retained: " + ", ".join(f">={t}d n={dist['n_at_least'][str(t)]}" for t in REPORT_AT))
for r in unrestricted:
    print(f"  UNRESTRICTED {r['outcome']}: n {r['n']} | RR {r['rr']} {r['rr_ci']} | ESS {r['ess_opioid']}/{r['ess_dopa']}")
for t in THRESHOLDS:
    b = by_threshold[str(t)]
    print(f"  --- lookback >= {t} d: retained {b['n_retained']} (opioid {b['n_retained_opioid']}), "
          f"dropped {b['n_dropped']}")
    for r in b["results"]:
        print(f"      {r['outcome']}: n {r['n']} | RR {r['rr']} {r['rr_ci']} | "
              f"SMD post {r['smd_post']} | ESS {r['ess_opioid']}/{r['ess_dopa']}")
s = seed_stability_365_72h
print(f"  seed check (365 d, 72h): point {s['rr_point_values']} deterministic={s['rr_point_deterministic']}, "
      f"CI lower {s['ci_lower_range']}, upper {s['ci_upper_range']}, "
      f"all include null={s['all_seeds_ci_include_null']}")
