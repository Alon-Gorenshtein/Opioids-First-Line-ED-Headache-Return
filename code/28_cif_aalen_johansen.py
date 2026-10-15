#!/usr/bin/env python3
"""Aalen-Johansen cumulative incidence of ED return, with death as a competing event.

The figure
(`06_figures.py:36`) plots `100*mean((t_ret_d > 0) & (t_ret_d <= x))`, an unnamed
cumulative proportion. This script computes the Aalen-Johansen nonparametric CIF
explicitly, so the Methods can name it, and confirms how far the plotted curve sits
from it.

Three things had to be established before the estimator could be trusted, and all three
are recorded in the digest rather than assumed.

1. THE TIME VARIABLE IN THE PARQUET IS SOUND. This was not always true, and the check
   is retained as a regression guard.

   `02_cohort.py` originally collapsed each patient to their index visit with
   `groupby("subject_id").first()`. Pandas' `.first()` returns the first NON-NULL value
   of each column independently, so for a patient with more than one treated adult
   headache visit any column that was null at the index visit silently inherited a later
   visit's value. `t_ret_d` is null by construction whenever the index visit had no
   return within 30 days, so 193 cohort rows (151 discharged) carried a `t_ret_d` copied
   forward from a subsequent visit while their own `ret_30d` was False, adding spurious
   early returns to the plotted curve.

   That defect has been REPAIRED AT SOURCE. `02_cohort.py` now collapses with `.nth(0)`,
   a true first-row selection, so the parquet stores the index visit's own return time.
   This script recomputes the return time from `gap_h` independently and asserts both
   that the two agree wherever a return occurred and that no row carries a return time
   without a 30-day return. `return_time_defect` in the digest reports the residual
   contamination count, which must now be zero.

   The boolean outcomes were never affected: booleans are never null, so `.first()` could
   not skip them. `ret_72h`, `ret_7d`, `ret_30d` and `gap_h` were each verified against a
   fresh derivation from the full `edstays` table and reproduce it exactly, and the
   headline estimates (72h RR 1.79, 7d 1.62, admission 2.32) were unchanged by the repair.

2. DEATH TIMES ARE RECOVERABLE. `died_30d` is a bare boolean in the parquet, but the
   date of death is in the MIMIC-IV `patients` table and is what `02_cohort.py:91`
   derived the flag from in the first place. Re-merging `dod` and recomputing
   `(dod - outtime)` reproduces `died_30d` for all 10,799 cohort rows, so deaths enter
   the estimator at their observed time rather than at an assumed one. `dod` is
   date-granular (all timestamps are midnight), so death times are resolved to the day.

3. ADMINISTRATIVE CENSORING CANNOT BE READ OFF THE CALENDAR, BUT IT CAN BE BOUNDED.
   Every patient's dates are shifted into the future by a random per-patient offset, so
   cohort `intime` spans 2110 to 2210 for care delivered in the 2010s and no calendar
   rule identifies the end of follow-up across patients. Two shift-invariant checks are
   combined instead, and a patient counts as having verified 30-day coverage if either
   one clears 30 days:

   (a) Observed record extent. The patient's own latest ED discharge, hospital
       discharge, or death anywhere in MIMIC-IV, minus their index ED discharge. A
       record at day 45 proves the patient was still observable through day 30.
   (b) Anchor-window bound. `outtime - Jan 1 of anchor_year` is shift-invariant because
       both dates carry the same offset. Calibrating that quantity against
       `anchor_year_group` puts the end of ED data collection at real-time 2020.84 to
       2020.92 from all four groups independently, which both cross-validates the shift
       model and yields a guaranteed remaining window per patient under the worst case
       that their true anchor year is their group's upper bound.

   The result is that 8,165 of 8,346 discharged patients (97.8%) have demonstrably
   complete 30-day follow-up. For the remaining 181 (2.2%) the data cannot settle it
   either way, so `n_censored_before_30d` is reported as null, not as zero. The primary
   estimate censors everyone at 30 days; a worst-case variant censoring those 181 at
   their verified window is reported alongside it.

A note on what the estimator change can and cannot do here. With no censoring before
30 days, the number at risk at time t is n minus the events already observed, so
S(t-) = Y(t)/n and the Aalen-Johansen increment collapses to d1(t)/n. The estimator is
then algebraically identical to the cumulative proportion of returns; the competing
event enters only by keeping deaths in the denominator, which the proportion already
did. So under the primary censoring assumption naming the estimator does not move the
curve at all, and on the repaired data that is now literally true of the shipped digest:
`max_abs_diff_vs_naive_pp` is 0.0. The honest Methods sentence is therefore that
cumulative incidence is computed with the Aalen-Johansen estimator and, under complete
follow-up to the horizon, coincides with the cumulative proportion. No caveat is needed.
"""
import os as _os
import json
import numpy as np
import pandas as pd

ED = _os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
HOSP = _os.environ.get("MIMIC_IV_HOSP_DIR", "mimic-iv-3.1/hosp")
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

CLASSES = ["opioid", "dopamine_antagonist", "nsaid", "acetaminophen"]
HORIZON = 30.0                     # days
GRID = np.arange(0, 30.25, 0.25)   # 0.25-day resolution, matching the figure's step plot
MIN_N = 150                        # magnesium (128) and triptan (16) stay off the figure

# anchor_year_group upper bounds, and the shift-invariant calibration of the end of ED
# data collection (see docstring point 3). Both are recomputed and re-checked below.
GROUP_HI = {"2008 - 2010": 2010, "2011 - 2013": 2013, "2014 - 2016": 2016, "2017 - 2019": 2019}
GROUP_LO = {"2008 - 2010": 2008, "2011 - 2013": 2011, "2014 - 2016": 2014, "2017 - 2019": 2017}


def aalen_johansen(time, event, grid, horizon=HORIZON):
    """Aalen-Johansen CIF for cause 1 with cause 2 competing, evaluated on `grid`.

    event: 1 = event of interest, 2 = competing event, 0 = censored.
    Ties are resolved with the usual convention that censoring at time t occurs after
    any events at t, so subjects censored at t are still at risk for events at t.
    The returned curve is right-continuous: grid point g carries CIF(g), including all
    events at exactly g.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if time.size == 0:
        return np.zeros(grid.size)
    if not np.isfinite(time).all():
        raise ValueError("non-finite event times")
    if time.min() < 0 or time.max() > horizon + 1e-9:
        raise ValueError(f"event times outside [0, {horizon}]")
    if not np.isin(event, (0, 1, 2)).all():
        raise ValueError("event codes must be 0, 1 or 2")

    at_risk = time.size
    surv = 1.0   # S(t-): probability of being free of BOTH events just before t
    cif = 0.0
    out = np.empty(grid.size, dtype=float)
    gi = 0
    for tt in np.unique(time):
        here = time == tt
        d1 = int(np.count_nonzero(here & (event == 1)))
        d2 = int(np.count_nonzero(here & (event == 2)))
        d0 = int(np.count_nonzero(here & (event == 0)))
        while gi < grid.size and grid[gi] < tt:
            out[gi] = cif
            gi += 1
        if at_risk > 0 and (d1 or d2):
            cif += surv * d1 / at_risk
            surv *= 1.0 - (d1 + d2) / at_risk
        at_risk -= d1 + d2 + d0
    while gi < grid.size:
        out[gi] = cif
        gi += 1
    return out


def naive_proportion(t_ret, grid):
    """The estimator the published figure actually used: cumulative share of the whole
    subgroup with a recorded return time in (0, x]. NaN compares False, so non-returners
    stay in the denominator."""
    t_ret = np.asarray(t_ret, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.array([np.mean((t_ret > 0) & (t_ret <= x)) for x in grid])


# --------------------------------------------------------------------------- load
m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
pat = pd.read_csv(f"{HOSP}/patients.csv.gz",
                  usecols=["subject_id", "anchor_year", "anchor_year_group", "dod"],
                  parse_dates=["dod"])
es = pd.read_csv(f"{ED}/edstays.csv.gz", usecols=["subject_id", "stay_id", "intime", "outtime"],
                 parse_dates=["intime", "outtime"])
adm = pd.read_csv(f"{HOSP}/admissions.csv.gz", usecols=["subject_id", "dischtime"],
                  parse_dates=["dischtime"])

assert set(CLASSES) <= set(m.first_class.dropna().unique()), "first_class labels changed"

# ------------------------------------------------- 1. rebuild and verify the timeline
# gap_h re-derived from the full edstays table, independently of the cohort build.
es = es.sort_values(["subject_id", "intime"])
es["gap_h_src"] = ((es.groupby("subject_id")["intime"].shift(-1) - es["outtime"])
                   .dt.total_seconds() / 3600.0)
m = m.merge(es[["stay_id", "gap_h_src"]], on="stay_id", how="left", validate="one_to_one")
assert np.allclose(m.gap_h.fillna(-1), m.gap_h_src.fillna(-1)), \
    "gap_h in the parquet does not reproduce a fresh derivation from edstays"
assert (m.ret_30d == ((m.gap_h > 0) & (m.gap_h <= 720))).all(), \
    "ret_30d is not the (0, 720] hour window on gap_h; the return timeline cannot be rebuilt"

m["t_ret"] = np.where(m.ret_30d, m.gap_h / 24.0, np.nan)
# Two-sided check on the stored column. The rebuild must agree wherever a return really
# occurred (it always did, even before the repair), and no row may carry a return time
# without a 30-day return (this was the defect; it must now read zero). The count is
# reported rather than asserted here so the digest records the verification and the test
# suite, not an import-time crash, is what fails if the defect ever returns.
_hit = m.ret_30d.values
assert np.allclose(m.t_ret.values[_hit], m.t_ret_d.values[_hit]), \
    "stored t_ret_d disagrees with the rebuild on rows that did have a 30-day return"
n_ret_time_from_later_visit = int((m.t_ret_d.notna() & ~m.ret_30d).sum())

# ------------------------------------------------------------ 2. recover death times
m = m.merge(pat, on="subject_id", how="left", validate="one_to_one")
assert m.anchor_year.notna().all(), "cohort subject missing from the patients table"
m["t_death"] = (m.dod - m.outtime).dt.total_seconds() / 86400.0
death_times_recovered = bool((m.died_30d == m.t_death.between(0, HORIZON)).all())
assert death_times_recovered, \
    "recomputed dod - outtime does not reproduce died_30d; death times are not trustworthy"
dod_day_granular = bool((m.dod.dropna().dt.normalize() == m.dod.dropna()).all())

# ------------------------------------------- 3. what can be established about censoring
last_ed = es.groupby("subject_id")["outtime"].max().rename("last_ed")
last_adm = adm.groupby("subject_id")["dischtime"].max().rename("last_adm")
m = m.merge(last_ed, on="subject_id", how="left").merge(last_adm, on="subject_id", how="left")
m["fu_observed_d"] = ((m[["last_ed", "last_adm", "dod"]].max(axis=1) - m.outtime)
                      .dt.total_seconds() / 86400.0)

# Shift-invariant calibration of the end of data collection, recomputed here rather than
# hardcoded: rel_y = outtime - Jan 1 of that patient's (shifted) anchor_year carries no
# offset, so max(rel_y) within an anchor_year_group pins the collection end relative to
# that group's lower bound. All four groups must agree, or the shift model is wrong.
es_cal = es.merge(pat[["subject_id", "anchor_year", "anchor_year_group"]], on="subject_id")
es_cal = es_cal[es_cal.anchor_year_group.isin(GROUP_LO)]
_jan1 = pd.to_datetime(es_cal.anchor_year.astype(int).astype(str) + "-01-01")
es_cal["rel_y"] = (es_cal.outtime - _jan1).dt.total_seconds() / (86400 * 365.25)
end_est = {g: round(GROUP_LO[g] + v, 3) for g, v in es_cal.groupby("anchor_year_group").rel_y.max().items()}
assert max(end_est.values()) - min(end_est.values()) < 0.25, (
    f"anchor-group estimates of the collection end disagree by more than a quarter year "
    f"({end_est}); the per-patient date-shift model behind this bound no longer holds"
)
collection_end = min(end_est.values())   # conservative: shortens the guaranteed window

_jan1_coh = pd.to_datetime(m.anchor_year.astype(int).astype(str) + "-01-01")
m["rel_y"] = (m.outtime - _jan1_coh).dt.total_seconds() / (86400 * 365.25)
m["guaranteed_fu_d"] = (m.anchor_year_group.map(GROUP_HI).rsub(collection_end) - m.rel_y) * 365.25
m["verified_fu_d"] = m[["guaranteed_fu_d", "fu_observed_d"]].max(axis=1)

d = m[m.discharged].copy()
n_verified = int((d.verified_fu_d >= HORIZON).sum())
n_indeterminate = int(len(d) - n_verified)

CENSORING_NOTE = (
    "Not determinable as a count. MIMIC-IV shifts every patient's dates by a random "
    "per-patient offset (cohort index intime spans 2110 to 2210 for care delivered in "
    "the 2010s), so end-of-window truncation cannot be identified by calendar date "
    f"across patients. Two shift-invariant checks were combined instead. {n_verified} of "
    f"{len(d)} discharged patients ({100 * n_verified / len(d):.1f}%) have demonstrably "
    "complete 30-day follow-up, either because their own record (ED, hospital, or death) "
    "extends at least 30 days past the index discharge or because the interval from their "
    "anchor year to the end of ED data collection guarantees it under the worst-case "
    f"assumption about their true anchor year. For the remaining {n_indeterminate} "
    f"({100 * n_indeterminate / len(d):.1f}%) the data cannot establish whether follow-up "
    "was truncated, so no censoring count is reported. The primary estimate censors all "
    "patients at 30 days; worst_case_admin_censoring censors these patients at their "
    "verified window instead and is reported as a bound."
)

# ------------------------------------------------------------------- 4. event coding
# 1 = ED return, 2 = death without a prior return, 0 = censored at the horizon.
# A patient who returned and then died contributes the return: the return was observed
# first and is the event of interest. Ties go to the return, because dod is day-granular
# and cannot resolve a same-day ordering that the ED timestamp already settles.
d = m[m.discharged].copy()
ret = d.ret_30d.values
death = d.died_30d.values
t_ret = d.t_ret.values
t_death = d.t_death.values
return_first = ret & (~death | (t_ret <= t_death))
death_first = death & ~return_first

d["event"] = np.where(return_first, 1, np.where(death_first, 2, 0)).astype(int)
d["time"] = np.where(return_first, t_ret, np.where(death_first, t_death, HORIZON))
d["time"] = np.minimum(d["time"], HORIZON)
assert d.time.notna().all() and (d.time >= 0).all(), "bad event times"

n_return = int((d.event == 1).sum())
n_death_no_return = int((d.event == 2).sum())
n_death_after_return = int((death & return_first).sum())

# Worst-case administrative censoring: patients whose 30-day coverage could not be
# verified are censored at their verified window. An event already observed proves
# follow-up reached it, so observed events are never censored away.
cap = np.where(d.verified_fu_d.values >= HORIZON, HORIZON,
               np.clip(d.verified_fu_d.values, 0.0, HORIZON))
wc_event = np.where(d.time.values <= cap, d.event.values, 0)
wc_time = np.where(d.time.values <= cap, d.time.values, cap)
n_wc_censored_early = int(((wc_event == 0) & (wc_time < HORIZON)).sum())

# ------------------------------------------------------------------ 5. per-class CIFs
by_class, diffs_stored, diffs_corrected, diffs_worstcase = {}, [], [], []
for cls in CLASSES:
    sub = d[d.first_class == cls]
    if len(sub) < MIN_N:
        continue
    mk = (d.first_class == cls).values
    aj = 100 * aalen_johansen(sub.time.values, sub.event.values, GRID)
    naive_stored = 100 * naive_proportion(sub.t_ret_d.values, GRID)  # from the parquet as stored
    naive_cor = 100 * naive_proportion(sub.t_ret.values, GRID)       # same estimator, independent rebuild
    aj_wc = 100 * aalen_johansen(wc_time[mk], wc_event[mk], GRID)

    diffs_stored.append(float(np.max(np.abs(aj - naive_stored))))
    diffs_corrected.append(float(np.max(np.abs(aj - naive_cor))))
    diffs_worstcase.append(float(np.max(np.abs(aj_wc - aj))))
    by_class[cls] = {
        "days": GRID.round(2).tolist(),
        "cif": aj.round(4).tolist(),
        "cif_worst_case_admin_censoring": aj_wc.round(4).tolist(),
        # Cumulative proportion computed from t_ret_d exactly as the parquet stores it. On the
        # repaired data this coincides with `cif` at every grid point (see estimator_identity_note),
        # so plot `cif`; this series exists only to demonstrate that the two agree.
        "naive_proportion_same_data": naive_stored.round(4).tolist(),
        "n": int(len(sub)),
        "n_return": int((sub.event == 1).sum()),
        "n_death_competing": int((sub.event == 2).sum()),
        "cif_30d": round(float(aj[-1]), 3),
        "naive_proportion_same_data_30d": round(float(naive_stored[-1]), 3),
        "max_abs_diff_vs_naive_proportion_pp": round(diffs_stored[-1], 3),
        "max_abs_diff_vs_naive_corrected_pp": round(diffs_corrected[-1], 6),
        "n_return_time_from_later_visit": int((sub.t_ret_d.notna() & ~sub.ret_30d).sum()),
    }

res = {
    "estimator": (
        "Aalen-Johansen nonparametric cumulative incidence function for ED return, with "
        "death within 30 days of index discharge treated as a competing event; patients "
        "without either event censored at 30 days"
    ),
    "population": "MIMIC-IV-ED, treated adult headache index visits discharged home",
    "n_discharged": int(len(d)),
    "horizon_days": HORIZON,
    "grid_step_days": 0.25,
    "n_return": n_return,
    "n_death_without_return": n_death_no_return,
    "n_death_after_return": n_death_after_return,
    "death_times_recovered": death_times_recovered,
    "death_time_source": (
        "MIMIC-IV hosp/patients.csv.gz dod, re-merged on subject_id; t_death = dod - index "
        "outtime. Reproduces the parquet's died_30d flag for all cohort rows. dod is "
        "date-granular" + (" (all timestamps midnight), so deaths are resolved to the day."
                           if dod_day_granular else ".")
    ),
    "n_censored_before_30d": None,
    "censoring_note": CENSORING_NOTE,
    "followup_verification": {
        "n_discharged": int(len(d)),
        "n_verified_30d_followup": n_verified,
        "n_indeterminate": n_indeterminate,
        "pct_verified": round(100 * n_verified / len(d), 2),
        "collection_end_estimates_by_anchor_group": end_est,
        "collection_end_used": collection_end,
        "method": (
            "verified if max(own last ED/hospital/death record - index outtime, "
            "guaranteed window from anchor_year to the calibrated end of ED data "
            "collection under the worst-case anchor year) >= 30 days"
        ),
    },
    "return_time_defect": {
        "status": (
            "Repaired at source. Counts below are the residual contamination in the parquet as "
            "read by this run and must be zero."
        ),
        "n_rows_cohort": int((m.t_ret_d.notna() & ~m.ret_30d).sum()),
        "n_rows_discharged": int((d.t_ret_d.notna() & ~d.ret_30d).sum()),
        "cause": (
            "02_cohort.py originally collapsed to the index visit with "
            "groupby('subject_id').first(), which returns the first NON-NULL value per column. "
            "t_ret_d is null whenever the index visit had no 30-day return, so 193 cohort rows "
            "(151 discharged) inherited a later visit's return time and added spurious early "
            "returns to the cumulative-incidence curve. Boolean outcomes could not be affected; "
            "gap_h, ret_72h, ret_7d and ret_30d were each re-derived from edstays and reproduce "
            "exactly, and the headline estimates were unchanged by the repair."
        ),
        "remedy": (
            "02_cohort.py now collapses with groupby('subject_id').nth(0), a true first-row "
            "selection, so the parquet stores each patient's own index-visit return time. This "
            "script re-derives the return time from the independently verified gap_h and checks "
            "it against the stored column in both directions; the counts above are that check."
        ),
    },
    "max_abs_diff_vs_naive_pp": round(max(diffs_stored), 3) if diffs_stored else 0.0,
    "max_abs_diff_vs_naive_corrected_pp": round(max(diffs_corrected), 6) if diffs_corrected else 0.0,
    "max_abs_diff_worst_case_censoring_pp": round(max(diffs_worstcase), 3) if diffs_worstcase else 0.0,
    "n_worst_case_censored_before_30d": n_wc_censored_early,
    "estimator_identity_note": (
        "With no censoring before the horizon the number at risk at time t is n minus the "
        "events already observed, so S(t-) = Y(t)/n and the Aalen-Johansen increment reduces "
        "to d1(t)/n. Under the primary censoring assumption the estimator is therefore "
        "algebraically identical to the cumulative proportion of returns, which is why both "
        "max_abs_diff_vs_naive_corrected_pp and max_abs_diff_vs_naive_pp are zero. The first "
        "compares the estimator against a cumulative proportion rebuilt here from the verified "
        "gap_h; the second compares it against a cumulative proportion computed from t_ret_d as "
        "the parquet stores it, and is zero only because the inherited-return-time defect has "
        "been repaired at source. Naming the estimator does not move the curve."
    ),
    "by_class": by_class,
}

with open(f"{OUT}/results_cif.json", "w") as fh:
    json.dump(res, fh, indent=2)

print(f"classes: {list(by_class)}")
print(f"discharged n={len(d):,} | returns {n_return} | competing deaths {n_death_no_return} "
      f"| deaths after a return {n_death_after_return}")
print(f"follow-up verified >=30 d for {n_verified:,}/{len(d):,} ({100 * n_verified / len(d):.1f}%); "
      f"{n_indeterminate} indeterminate; collection end {collection_end} {end_est}")
print(f"inherited return times remaining: {n_ret_time_from_later_visit} cohort rows "
      f"({int((d.t_ret_d.notna() & ~d.ret_30d).sum())} discharged) -- must be 0")
for cls, v in by_class.items():
    print(f"  {cls:20s} n={v['n']:5,d}  AJ 30d {v['cif_30d']:6.2f}%  "
          f"naive proportion {v['naive_proportion_same_data_30d']:6.2f}%  "
          f"max|AJ-naive| {v['max_abs_diff_vs_naive_proportion_pp']:.3f} pp")
print(f"max |AJ - naive proportion, same data| = {res['max_abs_diff_vs_naive_pp']} pp | "
      f"max |AJ - naive proportion, rebuilt time| = {res['max_abs_diff_vs_naive_corrected_pp']} pp | "
      f"worst-case censoring shifts AJ by <= {res['max_abs_diff_worst_case_censoring_pp']} pp")
print("wrote results_cif.json")
