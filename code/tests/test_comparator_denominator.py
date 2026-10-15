#!/usr/bin/env python3
"""eTable 14's restricted comparator must reconcile with eTable 15's agent counts."""
import os as _os
import json, sys
OUT = _os.environ.get("HEADACHE_OUT", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results"))
tie = json.load(open(f"{OUT}/results_comparator_tie.json"))
con = json.load(open(f"{OUT}/results_constituents.json"))

adm = [r for r in tie["restricted_comparator"] if r["outcome"] == "admitted"][0]
implied_dopa = adm["n"] - adm["n_opioid"]
declared = tie["n_restricted_dopa"]
assert implied_dopa == declared, f"implied {implied_dopa} != declared {declared}"

mimic = dict(con["constituents"]["MIMIC-IV-ED"]["dopamine_antagonist"])
core = sum(v for k, v in mimic.items()
           if k.lower() in ("metoclopramide", "prochlorperazine", "droperidol"))
assert declared == core, f"restricted comparator {declared} != eTable 15 core-agent sum {core}"
print(f"OK: restricted comparator = {declared} = eTable 15 core agents")
