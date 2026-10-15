#!/usr/bin/env python3
"""Supplement additions: eTable 15 constituent first-line agents within the
opioid and dopamine-antagonist classes, and eTable 16 raw (unweighted) event counts underlying
the primary associations. No new modeling: agent counts re-derive the specific first-line agent
(MIMIC pyxis 'name'; MC-MED orders 'Procedure_name') for the already-locked cohort, and raw event
counts are pure group sums of the cohort outcome flags. Every count is reconciled against the
locked class totals and crude rates before it is written."""
import os as _os
import re, json, os, pandas as pd, numpy as np

ED  = _os.environ.get("MIMIC_IV_ED_DIR", "mimic-iv-ed-2.2/ed")
MC  = _os.environ.get("MCMED_DIR", "mc-med-1.0.0")
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))

# ---- class + agent vocabularies (identical match terms to 02_cohort.py / 04_mcmed_analysis.py) ----
DRUG_CLASSES = {
 "opioid": r"morphine|hydromorphone|dilaudid|oxycodone|hydrocodone|fentanyl|codeine|tramadol|oxymorphone|meperidine|demerol|percocet|butorphanol|nalbuphine|tapentadol",
 "dopamine_antagonist": r"metoclopramide|reglan|prochlorperazine|compazine|chlorpromazine|droperidol|promethazine|phenergan",
 "nsaid": r"ketorolac|toradol|ibuprofen|naproxen|ketoprofen|diclofenac|indomethacin|aspirin",
 "triptan": r"sumatriptan|rizatriptan|zolmitriptan|eletriptan|naratriptan|frovatriptan|almotriptan|imitrex|maxalt|zomig",
 "dhe": r"dihydroergotamine|d\.h\.e|dhe.?45|migranal", "magnesium": r"magnesium",
 "acetaminophen": r"acetaminophen|tylenol|ofirmev|paracetamol"}
CLASS_ORDER = ["opioid","dopamine_antagonist","nsaid","triptan","dhe","magnesium","acetaminophen"]
COMPILED = {k: re.compile(v, re.I) for k, v in DRUG_CLASSES.items()}
def classify(name):
    n = str(name).lower()
    for k in CLASS_ORDER:
        if COMPILED[k].search(n): return k
    return None

# ordered (match term -> display agent); first hit wins, so combos resolve deterministically
OPI_AGENTS = [("morphine","Morphine"),("hydromorphone","Hydromorphone"),("dilaudid","Hydromorphone"),
    ("fentanyl","Fentanyl"),("oxycodone","Oxycodone"),("percocet","Oxycodone"),("hydrocodone","Hydrocodone"),
    ("codeine","Codeine"),("tramadol","Tramadol"),("oxymorphone","Oxymorphone"),("meperidine","Meperidine"),
    ("demerol","Meperidine"),("butorphanol","Butorphanol"),("nalbuphine","Nalbuphine"),("tapentadol","Tapentadol")]
DOP_AGENTS = [("metoclopramide","Metoclopramide"),("reglan","Metoclopramide"),("prochlorperazine","Prochlorperazine"),
    ("compazine","Prochlorperazine"),("droperidol","Droperidol"),("chlorpromazine","Chlorpromazine"),
    ("promethazine","Promethazine"),("phenergan","Promethazine")]
AGENT_MAP = {"opioid": OPI_AGENTS, "dopamine_antagonist": DOP_AGENTS}
def agent_of(name, cls):
    n = str(name).lower()
    for term, disp in AGENT_MAP[cls]:
        if term in n: return disp
    return None  # unmapped -> caught by assert

def constituents(names, cls):
    """names: pd.Series of raw first-line agent strings for one class. Returns ordered [(agent,n)]."""
    mapped = names.map(lambda x: agent_of(x, cls))
    unmapped = names[mapped.isna()]
    assert unmapped.empty, f"UNMAPPED {cls} agents:\n{unmapped.value_counts().to_string()}"
    vc = mapped.value_counts()
    return [(a, int(c)) for a, c in vc.items()]

# ===================== MIMIC: re-derive first-line agent name per cohort stay =====================
print("MIMIC: re-deriving first-line agent name ...")
coh_m = pd.read_parquet(f"{OUT}/intermediate/cohort_mimic.parquet")
pyx = pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id","charttime","name"], parse_dates=["charttime"])
pyx = pyx[pyx["stay_id"].isin(set(coh_m["stay_id"]))].copy()
pyx["cls"] = pyx["name"].map(classify); pyx = pyx.dropna(subset=["cls"])
pyx["ord"] = pyx["cls"].map({c: i for i, c in enumerate(CLASS_ORDER)})
pyx = pyx.sort_values(["stay_id","charttime","ord"])
fl_m = pyx.groupby("stay_id").first()[["cls","name"]].rename(columns={"cls":"fl_cls","name":"agent_name"})
coh_m = coh_m.merge(fl_m, left_on="stay_id", right_index=True, how="left")
# reconcile: recomputed class must equal the locked first_class for every row
mism = coh_m[coh_m["fl_cls"] != coh_m["first_class"]]
assert mism.empty, f"MIMIC class reconciliation failed for {len(mism)} rows"
print(f"  class reconciliation OK for all {len(coh_m):,} cohort stays")

mimic_const = {}
for cls in ["opioid","dopamine_antagonist"]:
    names = coh_m.loc[coh_m["first_class"] == cls, "agent_name"]
    rows = constituents(names, cls)
    tot = sum(n for _, n in rows)
    assert tot == len(names), f"MIMIC {cls} agent counts {tot} != class total {len(names)}"
    mimic_const[cls] = rows
    print(f"  MIMIC {cls}: total {tot} -> " + ", ".join(f"{a} {n}" for a, n in rows))

# ===================== MC-MED: reproduce index-visit selection + first-line Procedure_name =========
print("MC-MED: reproducing index-visit selection ...")
CC_RE = re.compile(r"(headache|head ache|migraine|cephalgia|cephalalgia)", re.I)
def headache_icd(code):
    c = str(code).replace(".","").upper().strip()
    return c.startswith(("346","30781","339","G43","G44","R51")) or c == "7840"
v = pd.read_csv(f"{MC}/visits.csv", on_bad_lines="skip", low_memory=False)
v.columns = [c.strip() for c in v.columns]
cc = v["CC"].fillna("").astype(str); dxn = v.get("Dx_name", pd.Series([""]*len(v))).fillna("").astype(str)
is_ha = cc.str.contains(CC_RE) | dxn.str.contains(CC_RE)
for col in ["Dx_ICD9","Dx_ICD10"]:
    if col in v:
        s = v[col].fillna("").astype(str)
        is_ha = is_ha | s.apply(lambda x: any(headache_icd(t) for t in re.split(r"[;,\s]+", x) if t))
v["is_ha"] = is_ha
ha_csn = set(v.loc[is_ha, "CSN"])

chunks = []
for ch in pd.read_csv(f"{MC}/orders.csv", usecols=["CSN","Order_type","Procedure_name","Order_time","First_admin_time"],
                      on_bad_lines="skip", chunksize=200_000, low_memory=False):
    ch = ch[(ch["Order_type"] == "Medications") & (ch["CSN"].isin(ha_csn))]
    if len(ch): chunks.append(ch)
med = pd.concat(chunks, ignore_index=True)
med["cls"] = med["Procedure_name"].map(classify); med = med.dropna(subset=["cls"])
med["t"] = pd.to_datetime(med["First_admin_time"].fillna(med["Order_time"]), errors="coerce", utc=True)
med = med.dropna(subset=["t"]); med["ord"] = med["cls"].map({c: i for i, c in enumerate(CLASS_ORDER)})
med = med.sort_values(["CSN","t","ord"])
fl = med.groupby("CSN").first()[["cls","Procedure_name"]].rename(columns={"cls":"first_class","Procedure_name":"agent_name"})
v = v.merge(fl, left_on="CSN", right_index=True, how="left")
v["treated"] = v["first_class"].notna()
v["age"] = pd.to_numeric(v["Age"], errors="coerce")
v["_arrival"] = pd.to_datetime(v.get("Arrival_time"), errors="coerce", utc=True)
coh_c = v[v["is_ha"] & v["treated"] & (v["age"] >= 18)].copy()
coh_c = coh_c.sort_values(["MRN","_arrival"]).groupby("MRN", as_index=False).first()
# reconcile against locked parquet class totals
locked = pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet")["first_class"].value_counts()
for cls in ["opioid","dopamine_antagonist"]:
    got = int((coh_c["first_class"] == cls).sum())
    assert got == int(locked.get(cls, -1)), f"MC-MED {cls} rederived {got} != locked {int(locked.get(cls,-1))}"
print(f"  MC-MED class totals reconciled: opioid {int((coh_c.first_class=='opioid').sum())}, dopa {int((coh_c.first_class=='dopamine_antagonist').sum())}")

mcmed_const = {}
for cls in ["opioid","dopamine_antagonist"]:
    names = coh_c.loc[coh_c["first_class"] == cls, "agent_name"]
    rows = constituents(names, cls)
    tot = sum(n for _, n in rows)
    assert tot == len(names), f"MC-MED {cls} agent counts {tot} != class total {len(names)}"
    mcmed_const[cls] = rows
    print(f"  MC-MED {cls}: total {tot} -> " + ", ".join(f"{a} {n}" for a, n in rows))

# ===================== raw (unweighted) event counts underlying the primary associations ===========
print("raw event counts ...")
def raw_events(df):
    """df has first_class, discharged, admitted, ret_72h, ret_7d. Returns per-arm events/n."""
    out = {}
    for cls, key in [("opioid","opioid"),("dopamine_antagonist","dopa")]:
        arm = df[df["first_class"] == cls]
        dis = arm[arm["discharged"]]
        out[key] = {
            "ret72": [int(dis["ret_72h"].sum()), int(len(dis))],
            "ret7d": [int(dis["ret_7d"].sum()), int(len(dis))],
            "admit": [int(arm["admitted"].sum()), int(len(arm))],
        }
    return out
m_raw = raw_events(coh_m)
c_raw = raw_events(pd.read_parquet(f"{OUT}/intermediate/cohort_mcmed.parquet"))
pooled = {}
for key in ["opioid","dopa"]:
    pooled[key] = {o: [m_raw[key][o][0] + c_raw[key][o][0], m_raw[key][o][1] + c_raw[key][o][1]]
                   for o in ["ret72","ret7d","admit"]}
def pct(ev_n): e, n = ev_n; return round(100 * e / n, 2) if n else None
# sanity: MIMIC crude rates must match eTable 6 (opioid 72h 6.58, dopa 3.81)
print(f"  MIMIC opioid 72h raw {m_raw['opioid']['ret72']} = {pct(m_raw['opioid']['ret72'])}% (eTable6 6.58)")
print(f"  MIMIC dopa   72h raw {m_raw['dopa']['ret72']} = {pct(m_raw['dopa']['ret72'])}% (eTable6 3.81)")

digest = {
    "constituents": {"MIMIC-IV-ED": mimic_const, "MC-MED": mcmed_const},
    "raw_events": {"MIMIC-IV-ED": m_raw, "MC-MED": c_raw, "Pooled": pooled},
}
json.dump(digest, open(f"{OUT}/results_constituents.json","w"), indent=2)
print("wrote results_constituents.json")

# emit ready-to-paste markdown rows
def const_rows(site, d):
    lines = []
    for cls, disp in [("opioid","Opioid"),("dopamine_antagonist","Dopamine antagonist")]:
        rows = d[cls]; tot = sum(n for _, n in rows)
        for i, (a, n) in enumerate(rows):
            lab = disp if i == 0 else ""
            lines.append(f"| {site if (cls=='opioid' and i==0) else ''} | {lab} | {a} | {n} | {100*n/tot:.1f} |")
        lines.append(f"|  | {disp}, all | All listed agents | {tot} | 100.0 |")
    return lines
print("\n--- eTable 15 rows ---")
for site, d in [("MIMIC-IV-ED", mimic_const), ("MC-MED", mcmed_const)]:
    for ln in const_rows(site, d): print(ln)
print("\n--- eTable 16 rows ---")
for out_lab, okey in [("72-hour ED return (discharged)","ret72"),("7-day ED return (discharged)","ret7d"),("Index admission (full cohort)","admit")]:
    for site, d in [("MIMIC-IV-ED", m_raw),("MC-MED", c_raw),("Pooled", pooled)]:
        o = d["opioid"][okey]; dp = d["dopa"][okey]
        print(f"| {out_lab} | {site} | {o[0]}/{o[1]} ({pct(o)}) | {dp[0]}/{dp[1]} ({pct(dp)}) |")
print("done")
