#!/usr/bin/env python3
"""Supplementary figures (house style: Helvetica, Okabe-Ito, 600 dpi PDF+PNG, no in-figure titles).
eFigure 1: propensity-score distribution, opioid vs dopamine antagonist (region of overlap / equipoise).
eFigure 2: Love plot of absolute standardized mean differences before vs after overlap weighting.
Primary contrast, discharged 72-hour population, MIMIC-IV-ED. Reuses 03/09 logic."""
import os as _os
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results")); FIG=f"{OUT}/figures"; os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
    "font.size":8,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.direction":"out","ytick.direction":"out","legend.frameon":False,"figure.dpi":150,
    "pdf.fonttype":42,"ps.fonttype":42})  # TrueType embedding required by journals (Nature/Lancet/Wiley)
CAT=["gender","race_grp","acuity","arrival_transport","shift","yeargrp"]; NUM=["age","pain_num","prior_ed"]
def design(d):
    d=d.copy(); d["pain_miss"]=d["pain_num"].isna().astype(int)
    d["pain_num"]=d["pain_num"].fillna(d["pain_num"].median()); d["acuity"]=d["acuity"].fillna("missing").astype(str)
    X=pd.get_dummies(d[CAT],drop_first=True,dummy_na=False)
    return pd.concat([d[NUM+["pain_miss"]].reset_index(drop=True),X.reset_index(drop=True)],axis=1).astype(float)
def fit(d):
    y=d["opioid"].values; Xs=StandardScaler().fit_transform(design(d).values)
    lr=LogisticRegression(max_iter=2000,C=1.0).fit(Xs,y); e=lr.predict_proba(Xs)[:,1]
    return e, np.where(y==1,1-e,e)
def smd(x,t,w=None):
    x=np.asarray(x,float); t=np.asarray(t); w=np.ones_like(x) if w is None else w
    def mv(m): ww=w[m]; xx=x[m]; mu=np.sum(ww*xx)/np.sum(ww); return mu,np.sum(ww*(xx-mu)**2)/np.sum(ww)
    m1,v1=mv(t==1); m0,v0=mv(t==0); s=np.sqrt((v1+v0)/2)
    return 0.0 if s==0 else (m1-m0)/s
def save(fig,name):
    fig.savefig(f"{FIG}/{name}.pdf",dpi=600,bbox_inches="tight",pad_inches=0.15); fig.savefig(f"{FIG}/{name}.png",dpi=600,bbox_inches="tight",pad_inches=0.15)
    plt.close(fig); print("wrote",name)

df=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
c=df[df["first_class"].isin(["opioid","dopamine_antagonist"])].copy()
c["opioid"]=(c["first_class"]=="opioid").astype(int)
d=c[c["discharged"]].reset_index(drop=True)
e,w=fit(d); t=d["opioid"].values

# ---- eFigure 1: propensity-score overlap ----
fig,ax=plt.subplots(figsize=(4.4,3.2))
bins=np.linspace(0,1,31)
ax.hist(e[t==0],bins=bins,color="#0072B2",alpha=0.55,label=f"Dopamine antagonist (n={int((t==0).sum()):,})",density=True)
ax.hist(e[t==1],bins=bins,color="#D55E00",alpha=0.55,label=f"Opioid (n={int((t==1).sum()):,})",density=True)
ax.set_xlabel("Estimated probability of first-line opioid treatment"); ax.set_ylabel("Density")
ax.set_xlim(0,1); ax.legend(loc="upper right",fontsize=6.5)
save(fig,"eFigure1_ps_overlap")

# ---- eFigure 2: Love plot ----
LABEL={"age":"Age","pain_num":"Triage pain score","prior_ed":"Prior ED visits","pain_miss":"Triage pain missing",
 "gender_M":"Male sex","race_grp_Black":"Black race","race_grp_Hispanic":"Hispanic","race_grp_Other/Unknown":"Race other/unknown",
 "race_grp_White":"White race","acuity_2.0":"Triage acuity 2","acuity_3.0":"Triage acuity 3","acuity_4.0":"Triage acuity 4",
 "acuity_5.0":"Triage acuity 5","acuity_missing":"Triage acuity missing","arrival_transport_WALK IN":"Arrival walk-in",
 "arrival_transport_UNKNOWN":"Arrival unknown","arrival_transport_OTHER":"Arrival other","arrival_transport_HELICOPTER":"Arrival helicopter",
 "shift_evening":"Evening arrival","shift_night":"Night arrival"}
X=design(d); rows=[]
for col in X.columns:
    if col.startswith("yeargrp_"): continue
    rows.append((LABEL.get(col,col),abs(smd(X[col].values,t)),abs(smd(X[col].values,t,w))))
rows.sort(key=lambda r:r[1])
fig,ax=plt.subplots(figsize=(4.8,4.6)); y=np.arange(len(rows))
ax.scatter([r[1] for r in rows],y,s=22,color="#999999",label="Before weighting",zorder=3)
ax.scatter([r[2] for r in rows],y,s=22,color="#D55E00",label="After overlap weighting",zorder=3)
for yi,r in zip(y,rows): ax.plot([r[2],r[1]],[yi,yi],color="#cccccc",lw=0.8,zorder=1)
ax.axvline(0.1,color="#888888",lw=0.8,ls="--")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=6.8)
ax.set_xlabel("Absolute standardized mean difference"); ax.set_xlim(-0.01,0.4); ax.legend(loc="lower right",fontsize=6.5)
save(fig,"eFigure2_loveplot")
print("efigures done")
