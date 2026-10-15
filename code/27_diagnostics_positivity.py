#!/usr/bin/env python3
"""Assumption diagnostics.

1. Positivity / common support: propensity-score range by arm, and the number of
   patients outside the region where both arms are represented.
2. Trimming: re-estimate after dropping the top and bottom 1% of the propensity score.
3. Collinearity: variance inflation factors on the propensity design matrix, Cramer's V
   between arrival mode and triage acuity, and re-estimates dropping each in turn.
4. Confounding, which is the separate question: Cramer's V of treatment against arrival mode
   and against triage acuity, on the same population, so the two can be compared directly.

Overlap weighting is by construction robust to near-violations of positivity because it
down-weights the extremes, so these are reported as confirmatory rather than corrective.
"""
import os as _os
import json, numpy as np, pandas as pd, cer_common as C
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
d = m[m.first_class.isin(["opioid", "dopamine_antagonist"])].copy()
d["opioid"] = (d.first_class == "opioid").astype(int)
dis = d[d.discharged].reset_index(drop=True)

X = C.design(dis, C.MIMIC_CAT)
Xs = StandardScaler().fit_transform(X.values)
e = LogisticRegression(max_iter=3000).fit(Xs, dis["opioid"].values).predict_proba(Xs)[:, 1]
t = dis["opioid"].values

lo = max(e[t == 1].min(), e[t == 0].min())
hi = min(e[t == 1].max(), e[t == 0].max())
outside = int(((e < lo) | (e > hi)).sum())


def vif(Xarr, cols):
    out = {}
    for j, c in enumerate(cols):
        y = Xarr[:, j]
        others = np.delete(Xarr, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
        out[c] = round(float(1 / (1 - r2)) if r2 < 0.9999 else float("inf"), 2)
    return out


vifs = vif(Xs, list(X.columns))


def cramers_v(a, b):
    ct = pd.crosstab(a, b).values
    chi2 = ((ct - ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()) ** 2
            / (ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum())).sum()
    n = ct.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


v = cramers_v(dis["arrival_transport"].astype(str), dis["acuity"].astype(str))

# The collinearity question (how much do arrival mode and acuity overlap with each other)
# is not the confounding question (how strongly is each associated with treatment). Both
# are computed on the same population so the two can be compared directly.
v_treat_arrival = cramers_v(dis["opioid"], dis["arrival_transport"].astype(str))
v_treat_acuity = cramers_v(dis["opioid"], dis["acuity"].astype(str))

keep = (e >= np.quantile(e, 0.01)) & (e <= np.quantile(e, 0.99))
res = {
    "ps_range_opioid": [round(float(e[t == 1].min()), 4), round(float(e[t == 1].max()), 4)],
    "ps_range_dopa": [round(float(e[t == 0].min()), 4), round(float(e[t == 0].max()), 4)],
    "common_support": [round(float(lo), 4), round(float(hi), 4)],
    "n_outside_common_support": outside,
    "pct_outside_common_support": round(100 * outside / len(dis), 2),
    "max_vif": max(x for x in vifs.values() if np.isfinite(x)),
    "vif_by_term": vifs,
    "cramers_v_arrival_acuity": round(v, 3),
    "cramers_v_treatment_arrival": round(v_treat_arrival, 3),
    "cramers_v_treatment_acuity": round(v_treat_acuity, 3),
    "trimmed_1pct": C.analyze(dis[keep].reset_index(drop=True), "ret_72h", C.MIMIC_CAT, label="trim1pct"),
    "refit_drop_arrival": C.analyze(dis, "ret_72h",
                                    [c for c in C.MIMIC_CAT if c != "arrival_transport"],
                                    label="drop-arrival"),
    "refit_drop_acuity": C.analyze(dis, "ret_72h",
                                   [c for c in C.MIMIC_CAT if c != "acuity"],
                                   label="drop-acuity"),
}
json.dump(res, open(f"{OUT}/results_diagnostics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "vif_by_term"}, indent=2))
