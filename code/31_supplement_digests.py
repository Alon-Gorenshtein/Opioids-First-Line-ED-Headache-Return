#!/usr/bin/env python3
"""Supplement-support digest.

Two jobs, both of which exist so that no number in the Supplementary Appendix is
hand-typed or hand-rounded:

  1. E-values for the confidence limit closest to the null.
     Uses the same closed form as cer_common.analyze (cer_common.py:59), applied to
     the confidence limit instead of the point estimate.
  2. AMA-formatted strings for every cell the supplement prints, rounded HALF-UP.
     Python's built-in round() is banker's rounding and returns 2.58 for 2.585
     where AMA wants 2.59, so all rounding here goes through decimal.ROUND_HALF_UP.

Reads the existing result digests, recomputes nothing that they already contain,
and writes results/results_supplement_digest.json.
"""
import os as _os
import json
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))


def r(x, nd):
    """Half-up rounding to a fixed number of decimals, returned as a string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(num, den, nd=1):
    """Exact crude percentage from a numerator and denominator, half-up."""
    return r(100.0 * num / den, nd)


def evalue(rr):
    """E-value on the risk-ratio scale; identical closed form to cer_common.py:59."""
    rr_ = rr if rr >= 1 else 1.0 / rr
    return float(rr_ + np.sqrt(rr_ * (rr_ - 1.0)))


def evalue_at_limit(rr, lo, hi):
    """E-value for the confidence limit closest to the null.

    If the interval already contains the null the E-value for the limit is 1 by
    convention (no unmeasured confounding is needed to reach the null).
    """
    if lo <= 1.0 <= hi:
        return 1.0, "interval contains the null"
    lim = lo if rr > 1.0 else hi
    return evalue(lim), f"limit {lim}"


def fmt_row(d):
    """AMA-formatted view of a standard analyze() result row."""
    o = {
        "label": d.get("label"),
        "outcome": d.get("outcome"),
        "n": d.get("n"),
        "n_opioid": d.get("n_opioid"),
        "risk_opioid_pct": r(d["risk_opioid"], 1) if "risk_opioid" in d else None,
        "risk_dopa_pct": r(d["risk_dopa"], 1) if "risk_dopa" in d else None,
        "rr": r(d["rr"], 2),
        "rr_ci": f"{r(d['rr_ci'][0], 2)} to {r(d['rr_ci'][1], 2)}",
        "evalue": r(d["evalue"], 2) if "evalue" in d else None,
    }
    if "rd_pct" in d:
        o["rd_pp"] = ("+" if d["rd_pct"] >= 0 else "") + r(d["rd_pct"], 1)
        o["rd_ci"] = f"{r(d['rd_ci'][0], 1)} to {r(d['rd_ci'][1], 1)}"
    for k in ("smd_pre", "smd_post"):
        if k in d:
            o[k] = r(d[k], 3)
    for k in ("ess_opioid", "ess_dopa"):
        if k in d:
            o[k] = r(d[k], 0)
    ev, note = evalue_at_limit(d["rr"], d["rr_ci"][0], d["rr_ci"][1])
    o["evalue_ci_limit"] = r(ev, 2)
    o["evalue_ci_limit_raw"] = round(ev, 6)
    o["evalue_ci_limit_note"] = note
    return o


def load(name):
    with open(f"{OUT}/{name}") as f:
        return json.load(f)


res = {}

# ---------------------------------------------------------------- eTable 4 / 5
pooled = load("results_digest_pooled.json")
res["pooled"] = [fmt_row(d) for d in pooled["pooled"]]
res["by_site"] = {s: [fmt_row(d) for d in rows] for s, rows in pooled["by_site"].items()}

print("E-values at the confidence limit closest to the null (pooled):")
for src, f in zip(pooled["pooled"], res["pooled"]):
    print(f"  {src['label']:<18} RR {src['rr']} ({src['rr_ci'][0]}, {src['rr_ci'][1]})"
          f" | E point {src['evalue']} -> E at CI limit {f['evalue_ci_limit']}"
          f"  (raw {f['evalue_ci_limit_raw']})")

# --------------------------------------------------------- other result tables
res["ha_return"] = {
    "pooled": [fmt_row(d) for d in load("results_ha_return.json")["pooled"]],
    "by_site": {s: [fmt_row(d) for d in rows]
                for s, rows in load("results_ha_return.json")["by_site"].items()},
}
comp = load("results_composite.json")
res["composite"] = {
    "pooled": [fmt_row(d) for d in comp["pooled"]],
    "by_site": {s: [fmt_row(d) for d in rows] for s, rows in comp["by_site"].items()},
}
res["primary_headache"] = [fmt_row(d) for d in load("results_primary_headache.json")["MIMIC_primary_headache"]]
res["exclude_secondary"] = [fmt_row(d) for d in load("results_exclude_secondary.json")["MIMIC_no_secondary"]]

ct = load("results_comparator_tie.json")
res["comparator_tie"] = {
    "restricted_comparator": [fmt_row(d) for d in ct["restricted_comparator"]],
    "drop_opioid_ties": [fmt_row(d) for d in ct["drop_opioid_ties"]],
    "n_restricted_dopa": ct["n_restricted_dopa"],
    "n_restricted_dopa_discharged": ct["n_restricted_dopa_discharged"],
    "n_tie_opioid_dropped": ct["n_tie_opioid_dropped"],
}

mig = load("results_migraine_only.json")
res["migraine_only"] = {
    "n_migraine": mig["n_migraine"],
    "n_migraine_discharged": mig["n_migraine_discharged"],
    "n_migraine_opioid": mig["n_migraine_opioid"],
    "n_migraine_opioid_discharged": mig["n_migraine_opioid_discharged"],
    "results": [fmt_row(d) for d in mig["results"]],
    "raw_counts": {
        k: {arm: {"num": v[arm]["num"], "den": v[arm]["den"],
                  "pct": pct(v[arm]["num"], v[arm]["den"])}
            for arm in ("opioid", "dopamine")}
        for k, v in mig["raw_counts"].items()
    },
}

lt = load("results_left_truncation.json")
res["left_truncation"] = {
    "n_contrast_cohort": lt["n_contrast_cohort"],
    "pct_zero_lookback": r(lt["lookback_distribution"]["pct_zero"], 1),
    "pct_ge_365_opioid": r(lt["lookback_distribution"]["by_arm"]["opioid"]["pct_ge_365"], 1),
    "pct_ge_365_dopa": r(lt["lookback_distribution"]["by_arm"]["dopamine_antagonist"]["pct_ge_365"], 1),
    "unrestricted_reference": [fmt_row(d) for d in lt["unrestricted_reference"]],
    "by_threshold": {
        k: {
            "n_before_restriction": v["n_before_restriction"],
            "n_dropped": v["n_dropped"],
            "n_retained": v["n_retained"],
            "n_retained_opioid": v["n_retained_opioid"],
            "n_retained_dopamine": v["n_retained_dopamine"],
            "n_retained_discharged": v["n_retained_discharged"],
            "n_retained_discharged_opioid": v["n_retained_discharged_opioid"],
            "raw_counts": {
                o: {arm: {"num": c[arm]["num"], "den": c[arm]["den"],
                          "pct": pct(c[arm]["num"], c[arm]["den"])}
                    for arm in ("opioid", "dopamine")}
                for o, c in v["raw_counts"].items()
            },
            "results": [fmt_row(d) for d in v["results"]],
        }
        for k, v in lt["by_threshold"].items()
    },
}

dg = load("results_diagnostics.json")
res["diagnostics"] = {
    "ps_range_opioid": [r(dg["ps_range_opioid"][0], 3), r(dg["ps_range_opioid"][1], 3)],
    "ps_range_dopa": [r(dg["ps_range_dopa"][0], 3), r(dg["ps_range_dopa"][1], 3)],
    "common_support": [r(dg["common_support"][0], 3), r(dg["common_support"][1], 3)],
    "n_outside_common_support": dg["n_outside_common_support"],
    "pct_outside_common_support": r(dg["pct_outside_common_support"], 1),
    "max_vif": r(dg["max_vif"], 1),
    "max_vif_term": max(dg["vif_by_term"], key=dg["vif_by_term"].get),
    "cramers_v_arrival_acuity": r(dg["cramers_v_arrival_acuity"], 3),
    "cramers_v_treatment_arrival": r(dg["cramers_v_treatment_arrival"], 3),
    "cramers_v_treatment_acuity": r(dg["cramers_v_treatment_acuity"], 3),
    "trimmed_1pct": fmt_row(dg["trimmed_1pct"]),
    "refit_drop_arrival": fmt_row(dg["refit_drop_arrival"]),
    "refit_drop_acuity": fmt_row(dg["refit_drop_acuity"]),
}

bd = load("results_bounds.json")
res["bounds"] = {}
for key in ("pooled_72h", "pooled_7d"):
    b = bd[key]
    res["bounds"][key] = {
        "p_discharged_opioid": r(b["p_discharged_opioid"], 1),
        "p_discharged_dopa": r(b["p_discharged_dopa"], 1),
        "always_discharged_share": r(b["always_discharged_share"], 3),
        "risk_opioid_pct": r(b["risk_opioid"], 1),
        "risk_dopa_observed_pct": r(b["risk_dopa_observed"], 1),
        "risk_dopa_lower_pct": r(b["risk_dopa_lower"], 1),
        "risk_dopa_upper_pct": r(b["risk_dopa_upper"], 1),
        "rr_lower": r(b["rr_lower"], 2),
        "rr_upper": b["rr_upper"],
        "rr_upper_unbounded": b["rr_upper_unbounded"],
        "rr_lower_boot_ci": f"{r(b['rr_lower_boot_ci'][0], 2)} to {r(b['rr_lower_boot_ci'][1], 2)}",
        "rr_lower_boot_ci_contains_null": b["rr_lower_boot_ci_contains_null"],
        "n_boot": b["n_boot"],
        "onesided_limit_note": b["onesided_limit_note"],
    }
res["bounds"]["n_neither_discharged_nor_admitted"] = bd["cohort"]["n_neither_discharged_nor_admitted"]
res["bounds"]["pct_neither"] = r(bd["cohort"]["pct_neither"], 1)
res["bounds"]["monotonicity_tipping_72h_pct"] = r(bd["monotonicity_tipping_72h"]["f_at_null_pct"], 1)
res["bounds"]["tipping_share_at_null"] = r(bd["tipping_point_72h"]["share_at_null"], 2)

# --------------------------------------------------------------- eTable 6 / 16
res["crude_by_class"] = {}
for site, dig in (("MIMIC-IV-ED", "results_digest.json"), ("MC-MED", "results_digest_mcmed.json")):
    cbc = load(dig)["crude_by_class"]
    res["crude_by_class"][site] = {
        k: {"n": f"{v['n']:,}", **{kk: r(vv, 1) for kk, vv in v.items() if kk != "n"}}
        for k, v in cbc.items()
    }

rep = load("results_reporting.json")
res["raw_events"] = {}
for k, v in rep.items():
    res["raw_events"][k] = {"num": f"{v['num']:,}", "den": f"{v['den']:,}",
                            "pct": pct(v["num"], v["den"])}
pooled_raw = load("results_constituents.json")["raw_events"]["Pooled"]
for arm in ("opioid", "dopa"):
    for oc in ("ret72", "ret7d", "admit"):
        num, den = pooled_raw[arm][oc]
        res["raw_events"][f"pooled_{oc}_{arm}"] = {"num": f"{num:,}", "den": f"{den:,}",
                                                   "pct": pct(num, den)}

# ------------------------------------------------------------------- eTable 7
res["cohort_class_distribution"] = {}
for site, path in (("MIMIC-IV-ED", "cohort_mimic.parquet"), ("MC-MED", "cohort_mcmed.parquet")):
    d = pd.read_parquet(f"{OUT}/intermediate/{path}")
    vc = d["first_class"].value_counts()
    res["cohort_class_distribution"][site] = {
        "n_analytic": f"{len(d):,}",
        "classes": {k: {"n": f"{int(v):,}", "pct": pct(int(v), len(d))} for k, v in vc.items()},
    }

# ------------------------------------------------------------------ eTable 15
res["constituents"] = {}
for site, classes in load("results_constituents.json")["constituents"].items():
    res["constituents"][site] = {}
    for cls, agents in classes.items():
        tot = sum(n for _, n in agents)
        res["constituents"][site][cls] = {
            "total": f"{tot:,}",
            "agents": [{"agent": a, "n": f"{n:,}", "pct_of_class": pct(n, tot)} for a, n in agents],
        }

# ------------------------------------------------------------------- eTable 9
res["balance_all"] = [
    {
        "label": d["label"], "n": f"{d['n']:,}", "n_opioid": f"{d['n_opioid']:,}",
        "smd_pre": r(d["smd_pre"], 3), "smd_post": r(d["smd_post"], 3),
        "ess_opioid": r(d["ess_opioid"], 0), "ess_dopa": r(d["ess_dopa"], 0),
    }
    for d in load("results_balance_all.json")
]
res["balance_all_max_smd_post"] = r(max(d["smd_post"] for d in load("results_balance_all.json")), 3)

with open(f"{OUT}/results_supplement_digest.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"\nwrote {OUT}/results_supplement_digest.json")
