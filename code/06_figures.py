#!/usr/bin/env python3
"""Phase-6 figures (house style: Helvetica, Okabe-Ito, 600 dpi PDF+PNG, no in-figure titles).
Fig 1: unadjusted Aalen-Johansen cumulative incidence of ED return over 30 days by
       first-line class (MIMIC, discharged; competing risk of in-hospital death).
Fig 2: forest of adjusted opioid-vs-dopamine RR (72h/7d return only; index admission
       dropped, since it follows the treatment decision and shares the triage-acuity
       confound) pooled + per site.
Fig 3: two-center crude replication of the primary contrast (opioid vs dopamine
       antagonist), 72h/7d return, MIMIC vs MC-MED, with Wilson 95% binomial CIs.
"""
import os as _os
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=_os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
FIG=f"{OUT}/figures"; import os; os.makedirs(FIG,exist_ok=True)

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
    "font.size":8,"axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.direction":"out","ytick.direction":"out","legend.frameon":False,"figure.dpi":150,
    "pdf.fonttype":42,"ps.fonttype":42})  # TrueType embedding required by journals (Nature/Lancet/Wiley)
OI={"dopamine_antagonist":"#0072B2","opioid":"#D55E00","nsaid":"#009E73","acetaminophen":"#CC79A7",
    "magnesium":"#E69F00","triptan":"#56B4E9"}
LAB={"dopamine_antagonist":"Dopamine antagonist","opioid":"Opioid","nsaid":"NSAID/ketorolac",
     "acetaminophen":"Acetaminophen","magnesium":"IV magnesium","triptan":"Triptan"}
def save(fig,name):
    fig.savefig(f"{FIG}/{name}.pdf",dpi=600,bbox_inches="tight",pad_inches=0.3); fig.savefig(f"{FIG}/{name}.png",dpi=600,bbox_inches="tight",pad_inches=0.3)
    plt.close(fig); print("wrote",name)

# ---------- Fig 1: unadjusted Aalen-Johansen cumulative incidence of return by class (MIMIC discharged) ----------
CIFJ=json.load(open(f"{OUT}/results_cif.json"))
fig,ax=plt.subplots(figsize=(4.2,3.4))
for cls in ["opioid","nsaid","acetaminophen","dopamine_antagonist"]:
    v=CIFJ["by_class"][cls]
    ax.step(v["days"],v["cif"],where="post",color=OI[cls],lw=1.8,label=f"{LAB[cls]} (n={v['n']:,})")
ax.set_xlabel("Days after index emergency department visit")
ax.set_ylabel("Unadjusted cumulative incidence of ED return (%)")
ax.set_xlim(0,30); ax.set_ylim(0,20)
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.16), ncol=2, frameon=False)
fig.subplots_adjust(bottom=0.30)
save(fig,"Figure1_cif_by_class")

# ---------- Fig 2: forest of adjusted RRs ----------
P=json.load(open(f"{OUT}/results_digest_pooled.json"))
rows=[]  # (ylabel, rr, lo, hi, group)
def grab(block,outcome):
    for r in block:
        if r["outcome"]==outcome: return r
labels=[("ret_72h","72-hour ED return"),("ret_7d","7-day ED return")]
order=[]
for oc,nm in labels:
    for site,tag in [("Pooled","pooled"),("MIMIC","MIMIC-IV-ED"),("MC-MED","MC-MED")]:
        blk=P["pooled"] if site=="Pooled" else P["by_site"][site]
        r=grab(blk,oc); rows.append((f"{nm}, {tag}",r["rr"],r["rr_ci"][0],r["rr_ci"][1],oc)); order.append(site)
fig,ax=plt.subplots(figsize=(5.0,3.4))
y=np.arange(len(rows))[::-1]
for yi,(lab,rr,lo,hi,oc),grp in zip(y,rows,order):
    col="#D55E00" if grp=="Pooled" else "#333333"
    ax.plot([lo,hi],[yi,yi],color=col,lw=1.3); ax.plot(rr,yi,"o",color=col,ms=5 if grp=="Pooled" else 4)
ax.axvline(1,color="#888888",lw=0.8,ls="--")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=6.8)
ax.set_xscale("log"); ax.set_xticks([0.5,1,2,3,4,5]); ax.set_xticklabels(["0.5","1","2","3","4","5"])
ax.set_xlabel("Adjusted risk ratio, opioid vs dopamine antagonist (log scale)"); ax.set_xlim(0.45,5.0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker="o",color="#D55E00",lw=1.3,label="Pooled two-center"),
                   Line2D([0],[0],marker="o",color="#333333",lw=1.3,label="Site-specific")],
          loc="upper center", bbox_to_anchor=(0.5,-0.13), ncol=2, fontsize=6.5)
save(fig,"Figure2_forest_adjusted")

# ---------- Fig 3: two-center crude replication by first-line class, with Wilson 95% CIs ----------
# Counts computed directly from the discharged cohort (matches eTable 6 / the existing
# figure's population), not from results_reporting.json, which is scoped to the primary
# opioid/dopamine-antagonist contrast only and does not carry nsaid/acetaminophen.
Cm=pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
Cs=pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet")

def wilson(num,den,z=1.96):
    if not den: return 0.0,0.0
    p=num/den
    c=z*np.sqrt(p*(1-p)/den + z*z/(4*den*den))
    lo=(p + z*z/(2*den) - c)/(1+z*z/den)
    hi=(p + z*z/(2*den) + c)/(1+z*z/den)
    return 100*p-100*lo, 100*hi-100*p

def crude(df,cls,window_col):
    mask=(df["first_class"]==cls)&(df["discharged"])
    den=int(mask.sum()); num=int((mask&df[window_col]).sum())
    return num,den

classes=["dopamine_antagonist","nsaid","acetaminophen","opioid"]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),sharey=True)
for ax,(df,site) in zip(axes,[(Cm,"MIMIC-IV-ED (BIDMC)"),(Cs,"MC-MED (Stanford)")]):
    x=np.arange(len(classes)); tops={c:0.0 for c in classes}; dens={c:0 for c in classes}
    for off,window_col,col,lab in [(-0.19,"ret_72h","#0072B2","72-hour"),(0.19,"ret_7d","#D55E00","7-day")]:
        nums,dens_,vals,los,his=[],[],[],[],[]
        for c in classes:
            num,den=crude(df,c,window_col)
            lo,hi=wilson(num,den)
            nums.append(num); dens_.append(den); vals.append(100*num/den if den else 0.0); los.append(lo); his.append(hi)
        ax.bar(x+off,vals,0.36,color=col,label=lab)
        ax.errorbar(x+off,vals,yerr=[los,his],fmt="none",ecolor="#333333",elinewidth=1.0,capsize=3)
        for c,v,hi,den in zip(classes,vals,his,dens_): tops[c]=max(tops[c],v+hi); dens[c]=den
    for xi,cls in zip(x,classes):
        ax.text(xi,tops[cls]+1.0,f"n={dens[cls]:,}",ha="center",va="bottom",fontsize=5.0,color="#444444")
    ax.set_xticks(x); ax.set_xticklabels([LAB[c].replace("/ketorolac","") for c in classes],rotation=30,ha="right",fontsize=6.5)
    ax.set_title(site,fontsize=7.5); ax.set_ylim(0,24)
axes[0].set_ylabel("Unadjusted crude return-visit rate (%)"); axes[0].legend(fontsize=6.5,loc="upper left")
save(fig,"Figure3_twocenter_replication")
print("figures done")
