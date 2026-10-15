#!/usr/bin/env python3
"""Bound the discharge-conditioned return contrast.

Return is only defined among patients discharged at the index visit, and discharge
follows treatment. The overlap-weighted discharged-only contrast is therefore not the
effect of first-line class in any pre-treatment-definable group. We bound the contrast
in the ALWAYS-DISCHARGED principal stratum: patients who would be discharged under
either first-line class.

Let p1 = P(discharged | opioid) and p0 = P(discharged | dopamine), both overlap-weighted.
Under monotonicity (opioid selection never makes discharge MORE likely, which the data
support: p1 << p0), every discharged opioid patient is always-discharged, so the opioid
arm's risk in the stratum is identified. The dopamine discharged group is a mixture:
a fraction p1/p0 are always-discharged and the remaining 1 - p1/p0 are discharged only
under dopamine. Manski/Lee worst-case bounds trim that unidentified fraction from each
tail: the always-discharged dopamine risk lies between the p1/p0-trimmed lower and upper
tails of the weighted dopamine outcome distribution. Binary outcomes make the trimming
exact.

Four results are produced, in increasing order of how much they assume:

  1. bounds_no_monotonicity  - assumption-free Manski bounds. The always-discharged
     share is itself only bounded, [max(0, p1+p0-1), min(p1, p0)], and BOTH arms are
     mixtures, so both are trimmed. With a rare outcome these are uninformative by
     construction; they are reported so the reader can see what the data alone deliver.
  2. bounds_*                - Lee (2009) bounds under monotonicity. One-sided here,
     because trimming a rare binary outcome drives the dopamine lower bound to exactly
     zero and the upper risk ratio to infinity.
  3. monotonicity_tipping_*  - how badly monotonicity must fail before the Lee LOWER
     bound reaches the null. This is the honest counterweight to result 2: a one-sided
     bound that rests entirely on an untestable assumption is only as strong as that
     assumption.
  4. tipping_point_72h       - the differential-selection factor that would explain the
     observed discharged-only association away entirely.

Sampling uncertainty in the Lee lower bound is carried by a nonparametric bootstrap
(row-level resampling, matching 03_primary_analysis.py and 05_pooled.py).
"""
import os as _os
import json, zlib, numpy as np, pandas as pd, cer_common as C

OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
SEED = 20260707
N_BOOT = 10000          # replicates behind the reported one-sided limit
MC_SEEDS = 12           # seeds in the Monte-Carlo stability study
MC_N_BOOT = 1000        # replicates per seed in that study
MC_SEED_BASE = 1000

# The one-sided limit sits essentially exactly at 1.0, so whether it "excludes the null"
# is decided by the resample draw rather than by the data. This string travels with every
# estimand so no downstream task can read a verdict off it.
ONESIDED_NOTE = (
    "Not statistically distinguishable from the null. The Monte Carlo standard deviation "
    "of this limit across bootstrap seeds is comparable to its distance from 1.0; do not "
    "report this bound as excluding the null."
)


def trimmed_risk_bounds(y, w, keep_frac):
    """Weighted risk of a binary outcome when only `keep_frac` of the mass is
    retained, choosing the retained mass to minimise then maximise the risk.

    Exact for binary y: the risk-minimising retained set takes non-events first,
    the risk-maximising set takes events first.
    """
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    W = float(np.sum(w))
    keep = keep_frac * W
    if keep <= 0:
        raise ValueError("no mass retained")
    w1 = float(np.sum(w[y == 1]))
    w0 = float(np.sum(w[y == 0]))
    # Lower bound: keep the zero-outcome mass first.
    lo_events = max(0.0, keep - w0)
    # Upper bound: keep the one-outcome mass first.
    hi_events = min(w1, keep)
    return lo_events / keep, hi_events / keep


def _rr(r1, r0):
    """Risk ratio, or None when the denominator bound is exactly zero (RR unbounded)."""
    return None if r0 <= 0 else r1 / r0


def discharge_probs(d, cat):
    """Overlap-weighted P(discharged | arm) in the FULL analytic population."""
    w = C.overlap_weights(d, cat)
    t = d["opioid"].values
    disch = d["discharged"].values.astype(float)
    adm = d["admitted"].values.astype(float)
    return dict(
        p1=C.wrisk(disch[t == 1], w[t == 1]), p0=C.wrisk(disch[t == 0], w[t == 0]),
        a1=C.wrisk(adm[t == 1], w[t == 1]), a0=C.wrisk(adm[t == 0], w[t == 0]),
    )


def _lee_core(d, outcome, cat):
    """Shared Lee-bound arithmetic. Returns the raw (unrounded) pieces."""
    d = d.reset_index(drop=True)
    pr = discharge_probs(d, cat)
    p1, p0 = pr["p1"], pr["p0"]
    ratio = p1 / p0 if p0 > 0 else np.nan
    share = min(1.0, ratio)

    dd = d[d["discharged"].values.astype(bool)].reset_index(drop=True)
    wd = C.overlap_weights(dd, cat)
    td = dd["opioid"].values
    yd = dd[outcome].astype(float).values
    r1 = C.wrisk(yd[td == 1], wd[td == 1])            # identified under monotonicity
    r0_obs = C.wrisk(yd[td == 0], wd[td == 0])
    r0_lo, r0_hi = trimmed_risk_bounds(yd[td == 0], wd[td == 0], share)
    return dict(pr=pr, ratio=ratio, share=share, r1=r1, r0_obs=r0_obs,
                r0_lo=r0_lo, r0_hi=r0_hi,
                y0=yd[td == 0], w0=wd[td == 0], y1=yd[td == 1], w1=wd[td == 1])


def bounds_for(pop, outcome, cat, n_boot=N_BOOT, seed=SEED, label=""):
    """Lee (2009) bounds on the always-discharged-stratum risk ratio, with a
    nonparametric bootstrap distribution for the LOWER bound."""
    d = pop.reset_index(drop=True)
    k = _lee_core(d, outcome, cat)
    if k["ratio"] > 1.0:
        raise AssertionError(
            f"monotonicity violated for {label}/{outcome}: weighted P(discharged|opioid) "
            f"{k['ratio']:.4f}x P(discharged|dopamine) > 1. The trimming argument inverts; "
            "the arm being trimmed would have to change. Stopping rather than flipping it."
        )
    rr_lo = _rr(k["r1"], k["r0_hi"])
    rr_hi = _rr(k["r1"], k["r0_lo"])

    rng = np.random.default_rng(seed + zlib.crc32((outcome + label).encode()))  # deterministic, distinct per outcome
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        try:
            kb = _lee_core(d.iloc[idx], outcome, cat)
            if kb["ratio"] > 1.0 or kb["r0_hi"] <= 0:
                continue
            boots.append(kb["r1"] / kb["r0_hi"])
        except Exception:
            continue
    bl = np.percentile(boots, [2.5, 97.5]) if boots else [np.nan, np.nan]
    # Identified set is [rr_lower, inf), so the Imbens-Manski 95% CI for the RR is
    # one-sided: the 5th percentile of the bootstrap distribution of the lower bound.
    im = float(np.percentile(boots, 5)) if boots else np.nan

    return {
        "p_discharged_opioid": round(100 * k["pr"]["p1"], 1),
        "p_discharged_dopa": round(100 * k["pr"]["p0"], 1),
        "p_admitted_opioid": round(100 * k["pr"]["a1"], 1),
        "p_admitted_dopa": round(100 * k["pr"]["a0"], 1),
        "gap_pp_opioid": round(100 * (1 - k["pr"]["p1"] - k["pr"]["a1"]), 2),
        "gap_pp_dopa": round(100 * (1 - k["pr"]["p0"] - k["pr"]["a0"]), 2),
        "always_discharged_share": round(k["share"], 3),
        "share_definition": ("fraction of the overlap-weighted DISCHARGED dopamine-antagonist "
                             "group that belongs to the always-discharged stratum, = p1/p0"),
        "risk_opioid": round(100 * k["r1"], 2),
        # Point-identified under monotonicity, so the opioid bounds coincide.
        "risk_opioid_lower": round(100 * k["r1"], 2),
        "risk_opioid_upper": round(100 * k["r1"], 2),
        "risk_dopa_observed": round(100 * k["r0_obs"], 2),
        "risk_dopa_lower": round(100 * k["r0_lo"], 2),
        "risk_dopa_upper": round(100 * k["r0_hi"], 2),
        # RR is highest when the dopamine risk is at its lower bound, and vice versa.
        "rr_lower": round(rr_lo, 2) if rr_lo is not None else None,
        "rr_upper": round(rr_hi, 2) if rr_hi is not None else None,
        "rr_upper_unbounded": rr_hi is None,
        "rr_lower_boot_ci": [round(float(bl[0]), 2), round(float(bl[1]), 2)],
        # Canonical name for the inferential limit. `rr_ci95_onesided_lower` is a retained
        # alias (progress.md names it); a test pins the two equal so they cannot drift.
        "onesided_limit": round(im, 2) if np.isfinite(im) else None,
        "rr_ci95_onesided_lower": round(im, 2) if np.isfinite(im) else None,
        "onesided_limit_note": ONESIDED_NOTE,
        "n_boot": n_boot,
        "n_boot_ok": len(boots),
        # Describes the IDENTIFIED SET, not the inferential limit. Stable, and a different
        # claim: it asks whether the bounds themselves straddle 1.0.
        "bounds_contain_null": bool((rr_lo is None or rr_lo <= 1.0) and (rr_hi is None or rr_hi >= 1.0)),
        "rr_lower_boot_ci_contains_null": bool(np.isnan(bl[0]) or bl[0] <= 1.0),
    }


def bounds_no_monotonicity(pop, outcome, cat, n_grid=201):
    """Assumption-free Manski bounds: monotonicity is NOT imposed, so the
    always-discharged share is only set-identified and both arms are mixtures.

    For each feasible stratum mass pi, the opioid discharged group is trimmed to
    pi/p1 of its mass and the dopamine discharged group to pi/p0. The reported
    interval is the union over all feasible pi.
    """
    k = _lee_core(pop, outcome, cat)
    p1, p0 = k["pr"]["p1"], k["pr"]["p0"]
    pi_lo, pi_hi = max(0.0, p1 + p0 - 1.0), min(p1, p0)
    rr_lo, rr_hi = np.inf, 0.0
    unbounded = False
    for pi in np.linspace(max(pi_lo, 1e-6), pi_hi, n_grid):
        a_lo, a_hi = trimmed_risk_bounds(k["y1"], k["w1"], min(1.0, pi / p1))
        b_lo, b_hi = trimmed_risk_bounds(k["y0"], k["w0"], min(1.0, pi / p0))
        lo = _rr(a_lo, b_hi)
        hi = _rr(a_hi, b_lo)
        if lo is not None:
            rr_lo = min(rr_lo, lo)
        if hi is None:
            unbounded = True
        else:
            rr_hi = max(rr_hi, hi)
    return {
        "stratum_mass_lower": round(pi_lo, 3),
        "stratum_mass_upper": round(pi_hi, 3),
        "rr_lower": round(float(rr_lo), 2) if np.isfinite(rr_lo) else None,
        "rr_upper": None if unbounded else round(float(rr_hi), 2),
        "rr_upper_unbounded": unbounded,
        "bounds_contain_null": True if unbounded else bool(rr_lo <= 1.0 <= rr_hi),
        "note": ("Monotonicity not imposed. Both arms are trimmed, so neither arm's "
                 "stratum risk is identified and the interval is uninformative by "
                 "construction for a rare binary outcome."),
    }


def monotonicity_tipping(pop, outcome, cat, n_grid=20001):
    """How far must monotonicity fail before the Lee LOWER bound reaches 1.0?

    Let f be the share of the overlap-weighted DISCHARGED opioid group that would
    have been admitted under a dopamine antagonist (opioid-only-discharged; f = 0 is
    exact monotonicity). The always-discharged mass becomes p1(1 - f), so the opioid
    discharged group is trimmed to 1 - f of its mass and the dopamine discharged group
    to share*(1 - f). Report the smallest f at which the lower bound touches the null.
    """
    k = _lee_core(pop, outcome, cat)
    share = k["share"]
    grid = np.linspace(0.0, 1.0, n_grid)
    f_null = None
    for f in grid:
        keep1 = 1.0 - f
        if keep1 <= 0:
            break
        a_lo, _ = trimmed_risk_bounds(k["y1"], k["w1"], keep1)
        _, b_hi = trimmed_risk_bounds(k["y0"], k["w0"], min(1.0, share * keep1))
        lo = _rr(a_lo, b_hi)
        if lo is None or lo <= 1.0:
            f_null = float(f)
            break
    # Closed form while both trims stay in the interior: RR_lo(f) = share*(r1 - f)/r0.
    closed = k["r1"] - k["r0_obs"] / share if share > 0 else np.nan
    return {
        "f_at_null_pct": None if f_null is None else round(100 * f_null, 2),
        "f_at_null_closed_form_pct": round(100 * float(closed), 2) if np.isfinite(closed) else None,
        "description": (
            "Monotonicity would have to fail for "
            f"{'>100' if f_null is None else round(100 * f_null, 2)}% of the discharged "
            "opioid group (patients discharged after an opioid who would have been admitted "
            "after a dopamine antagonist) before the always-discharged lower bound reaches 1.0. "
            "The outcome is rare, so the one-sided bound is fragile to small departures from "
            "the assumption."
        ),
    }


def mc_stability(pop, outcome, cat, n_seeds=MC_SEEDS, n_boot=MC_N_BOOT, big_boot=None):
    """Monte Carlo stability of the one-sided limit across independent bootstrap seeds.

    The limit sits at the null, so a single seed's verdict is not a finding. Re-running the
    whole bootstrap under `n_seeds` independent seeds measures how much of the reported
    value is resample noise. Recorded in the digest as the evidence that this instability
    was found here rather than in review.
    """
    lims = []
    for k in range(n_seeds):
        r = bounds_for(pop, outcome, cat, n_boot=n_boot, seed=MC_SEED_BASE + k,
                       label=f"mc{k}")
        lims.append(r["onesided_limit"])
    a = np.array([x for x in lims if x is not None], float)
    out = {
        "n_seeds": n_seeds,
        "n_boot_per_seed": n_boot,
        "n_boot_reported": N_BOOT,
        "limits": [round(float(x), 3) for x in a],
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
        "mean": round(float(a.mean()), 3),
        "sd": round(float(a.std(ddof=1)), 3),
        "n_seeds_excluding_null": int((a > 1.0).sum()),
        "fraction_excluding_null": f"{int((a > 1.0).sum())}/{len(a)}",
    }
    if big_boot:
        b = bounds_for(pop, outcome, cat, n_boot=big_boot, seed=SEED, label="big")
        out["big_run_n_boot"] = big_boot
        out["big_run_onesided_limit"] = b["onesided_limit"]
    return out


def tipping_point(pop, outcome, cat):
    """How strong must differential selection into discharge be to move the observed
    RR to 1.0? Model it as a multiplicative shift s applied to the opioid arm's
    discharged-group risk: report the smallest s that brings the RR to 1."""
    share = min(1.0, (lambda pr: pr["p1"] / pr["p0"])(discharge_probs(pop, cat)))
    d = pop[pop.discharged].reset_index(drop=True)
    w = C.overlap_weights(d, cat)
    t = d["opioid"].values
    y = d[outcome].astype(float).values
    r1 = C.wrisk(y[t == 1], w[t == 1])
    r0 = C.wrisk(y[t == 0], w[t == 0])
    rr = r1 / r0
    # (r1 / s) / r0 = 1  =>  s = rr. Also: the Lee lower bound is rr*share, so the
    # always-discharged share would have to fall to 1/rr for that bound to reach 1.
    s = rr
    return {
        "observed_rr": round(rr, 2),
        "selection_rr_at_null": round(s, 2),
        "observed_share": round(share, 3),
        "share_at_null": round(1.0 / rr, 3),
        "description": (
            "The selected (discharged) opioid group would have to have a return risk "
            f"{round(s, 2)} times lower than the always-discharged opioid population, "
            "with no corresponding distortion in the dopamine-antagonist arm, for the "
            "observed association to be entirely selection. Equivalently, the "
            "always-discharged share of the discharged dopamine-antagonist group would "
            f"have to fall from its estimated {round(share, 3)} to {round(1.0 / rr, 3)} "
            "for the always-discharged lower bound to reach the null."
        ),
    }


m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
m = m[m.first_class.isin(["opioid", "dopamine_antagonist"])].copy()
m["opioid"] = (m.first_class == "opioid").astype(int)

s = pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet")
s = s[s.first_class.isin(["opioid", "dopamine_antagonist"])].copy()
s["opioid"] = (s.first_class == "opioid").astype(int)

# Restrict to shared columns before stacking so no column silently becomes object
# dtype with NaN on one side (MC-MED has no yeargrp; MIMIC has no pid).
shared = [c for c in m.columns if c in s.columns]
pooled = pd.concat([m.assign(site="MIMIC")[shared + ["site"]],
                    s.assign(site="MC-MED")[shared + ["site"]]], ignore_index=True)

# `discharged` is disposition exactly "HOME"; `admitted` is disposition containing
# "ADMIT". They are not exact complements (transfers, elopements, AMA, deaths), so
# record the size of the gap rather than inferring one from the other.
gap_n = int((~pooled.discharged & ~pooled.admitted).sum())

res = {
    "cohort": {
        "n_pooled": int(len(pooled)), "n_mimic": int(len(m)), "n_mcmed": int(len(s)),
        "n_neither_discharged_nor_admitted": gap_n,
        "pct_neither": round(100 * gap_n / len(pooled), 2),
    },
    "pooled_72h": bounds_for(pooled, "ret_72h", C.POOL_CAT, label="pooled"),
    "pooled_7d": bounds_for(pooled, "ret_7d", C.POOL_CAT, label="pooled"),
    "mimic_72h": bounds_for(m, "ret_72h", C.MIMIC_CAT, label="mimic"),
    "pooled_72h_no_monotonicity": bounds_no_monotonicity(pooled, "ret_72h", C.POOL_CAT),
    "monotonicity_tipping_72h": monotonicity_tipping(pooled, "ret_72h", C.POOL_CAT),
    "tipping_point_72h": tipping_point(pooled, "ret_72h", C.POOL_CAT),
}

# Monte Carlo stability of the one-sided limit, measured per estimand.
res["mc_stability"] = {
    "pooled_72h": mc_stability(pooled, "ret_72h", C.POOL_CAT, big_boot=20000),
    "pooled_7d": mc_stability(pooled, "ret_7d", C.POOL_CAT),
    "mimic_72h": mc_stability(m, "ret_72h", C.MIMIC_CAT),
}
for key, src in (("pooled_72h", "pooled_72h"), ("pooled_7d", "pooled_7d"), ("mimic_72h", "mimic_72h")):
    mc = res["mc_stability"][src]
    res[key]["onesided_limit_mc_sd"] = mc["sd"]
    res[key]["onesided_limit_seed_range"] = [mc["min"], mc["max"]]

# Top-level aliases matching the interface contract in the task brief.
res["pi_discharged_opioid"] = res["pooled_72h"]["p_discharged_opioid"]
res["pi_discharged_dopa"] = res["pooled_72h"]["p_discharged_dopa"]
res["always_discharged_share"] = res["pooled_72h"]["always_discharged_share"]
# The bare alias travels without context, so carry the definition with it. 0.731 is the
# share of the DISCHARGED dopamine-antagonist group, not of the whole population (that is
# p1 = 59.8%). Quoting it as "73% of patients would have been discharged either way" is wrong.
res["always_discharged_share_definition"] = res["pooled_72h"]["share_definition"]
res["bounds_72h"] = res["pooled_72h"]
res["bounds_7d"] = res["pooled_7d"]

json.dump(res, open(f"{OUT}/results_bounds.json", "w"), indent=2)
for k in ("cohort", "pooled_72h", "pooled_7d", "mimic_72h",
          "pooled_72h_no_monotonicity", "monotonicity_tipping_72h", "tipping_point_72h",
          "mc_stability"):
    print(k, json.dumps(res[k]))
