# First-line opioids versus dopamine-receptor antagonists for emergency department headache and short-term ED return

Analysis code and aggregate results for a two-center retrospective cohort study.

Gorenshtein A, Adiniaev Y, Liba T, Klang E, Daniel O. *Opioids as the Earliest Acute Treatment for
Emergency Department Headache and Short-Term All-Cause Emergency Department Return: A Two-Center
Comparative Cohort Study.* *Headache*.

## Study

Adults treated for headache in the emergency department (ED) at two sites, MIMIC-IV-ED
(Beth Israel Deaconess Medical Center, 2011-2019) and MC-MED (Stanford, 2020-2022). First-line
treatment is the earliest qualifying acute agent. The primary contrast is opioids versus
guideline-preferred dopamine-receptor antagonists; the primary outcome is 72-hour all-cause ED return
among patients discharged from the index visit, with 7-day return secondary. Effects are
overlap-weighted risk ratios with bootstrap 95% CIs, pooled and by site.

| | |
|---|---|
| Treated adults | 13,285 (10,799 MIMIC-IV-ED; 2,486 MC-MED) |
| Pooled 72-hour return, opioid vs dopamine antagonist | 6.8% vs 3.8%; RR 1.78 (95% CI 1.31-2.32) |
| Pooled 7-day return | 10.7% vs 6.6%; RR 1.62 (95% CI 1.28-1.99) |
| Headache-coded return, 72 hours | RR 1.31 (95% CI 0.91-1.88) |

The estimates are associations conditional on discharge and reflect all-cause ED utilization, not
confirmed headache recurrence.

## Repository layout

```
code/                     numbered analysis scripts plus the shared estimator (cer_common.py)
results/results_*.json    every reported estimate, as written by the scripts
results/figures/          manuscript and supplementary figures (PDF and PNG)
results/tables/           Table 1 (both sites) and the covariate-balance tables, CSV
```

## Scripts

| Script | Purpose |
|---|---|
| `00_recon.py`, `01_recon_mcmed.py` | Source-table checks for MIMIC-IV-ED and MC-MED |
| `02_cohort.py` | MIMIC-IV-ED analytic cohort (writes the cohort file read by later scripts) |
| `03_primary_analysis.py` | Primary MIMIC-IV-ED contrast |
| `04_mcmed_analysis.py` | MC-MED cohort and contrast |
| `05_pooled.py` | Pooled two-center analysis |
| `06_figures.py`, `10_efigures.py`, `21_schematic.py` | Figures |
| `07_table1.py`, `29_table1_mcmed.py` | Table 1 by site |
| `09_balance.py`, `19_balance_diagnostics.py` | Standardized mean differences and effective sample size |
| `14_sens_ha_return.py` | Headache-specific return outcome |
| `15_composite.py` | Composite of admission or 72-hour return |
| `16_sens_primary_headache.py`, `17_sens_exclude_secondary.py`, `24_sens_migraine_only.py` | Phenotype restrictions |
| `18_sens_comparator_tie.py`, `23_constituent_agents.py` | Comparator definition and constituent agents |
| `25_bounds_discharge_selection.py` | Bounds for conditioning on discharge |
| `26_sens_left_truncation.py` | Left truncation and washout |
| `27_diagnostics_positivity.py` | Positivity, trimming, collinearity |
| `28_cif_aalen_johansen.py` | Aalen-Johansen cumulative incidence with death as a competing event |
| `30_reporting_digest.py`, `31_supplement_digests.py` | Crude counts, E-values, rounded digests |

## Reproducing

The source databases are not redistributed. Obtain MIMIC-IV-ED 2.2, MIMIC-IV 3.1 (hosp module) and
MC-MED 1.0.0 from PhysioNet under their credentialed data use agreements, then:

```bash
pip install -r requirements.txt
export MIMIC_IV_ED_DIR=/path/to/mimic-iv-ed-2.2/ed
export MIMIC_IV_HOSP_DIR=/path/to/mimic-iv-3.1/hosp
export MCMED_DIR=/path/to/mc-med-1.0.0
export HEADACHE_OUT=$PWD/results      # optional; this is the default

python code/02_cohort.py              # writes results/intermediate/cohort_mimic.parquet
python code/03_primary_analysis.py    # then the remaining scripts in numeric order
```

Patient-level cohort files are written to `results/intermediate/` and are git-ignored.
`cer_common.py` fixes the random seed; reruns reproduce the committed JSON files.

## License

MIT. See `LICENSE`.
