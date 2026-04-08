# Experiment Log

---
### Experiment 0: Baseline
**Date:** 2026-04-09
**Hypothesis:** Establish baseline C-index with existing feature engineering and models.
**Changes:** None — ran existing pipeline as-is.
**Result:** C-index = 0.7137 | status: keep
**Analysis:** Baseline established. GBM (0.7131) and RSF (0.7125) are strongest. CoxPH (0.6652) is weakest. LightGBM (0.6854) underperforms. 571 features after variance filter.
**Next idea:** Implement ELN 2022 risk classification proxy features from cytogenetics + mutations.

---
### Experiment 1: ELN risk classification + pathway features + clone burden
**Date:** 2026-04-09
**Hypothesis:** Adding ELN 2022 risk group proxies, mutation pathway aggregation, and clone burden features will improve C-index by capturing known prognostic categories.
**Changes:**
- data_transform.py: Added `_add_eln_risk()` — parses cytogenetics for favorable/adverse categories
- data_transform.py: Added `_add_pathway_features()` — groups genes into 6 pathways with count/has/max_vaf features
- data_transform.py: Added `_add_clone_burden()` — extracts cell counts from ISCN bracket notation
**Result:** C-index = 0.7135 (prev: 0.7137) | status: discard
**Analysis:** Slight degradation. ELN categories are too coarse when the model already has individual gene/cyto features. Pathway features are correlated with individual gene features. Clone burden only applies to ~60% of patients with cytogenetics. Reverted.
**Next idea:** Try targeted high-value interaction features (NPM1+FLT3-ITD, TP53 multi-hit, monosomal karyotype) rather than full classification. Also try improving LightGBM which underperforms.

---
### Experiment 2: Targeted interactions + LightGBM tuning
**Date:** 2026-04-09
**Hypothesis:** Lean targeted interaction features (NPM1 without FLT3-ITD, TP53 multi-hit, FLT3-ITD flag, monosomal karyotype, pathway counts, clone fraction) plus LightGBM tuning (more trees, deeper, lower LR) will improve C-index.
**Changes:**
- data_transform.py: Added `_add_targeted_interactions()` with 9 focused features: npm1_no_flt3itd, tp53_multihit, flt3_itd, monosomal_karyotype, n_monosomies, path_epigenetic/splicing/signaling_count, cyto_major_clone_frac
- train.py: Tuned LightGBM — n_estimators 400→600, max_depth 4→5, learning_rate 0.05→0.03, num_leaves 20→31, min_child_samples 20→15
**Result:** C-index = 0.7139 (prev: 0.7137) | status: keep
**Analysis:** Small improvement. The targeted interaction features helped the ensemble despite LightGBM not improving much. GBM and RSF remain dominant. The new features (npm1_no_flt3itd, tp53_multihit, flt3_itd, monosomal_karyotype, etc.) add useful signal. LightGBM tuning didn't help much — may need more aggressive changes.
**Next idea:** Try more aggressive LightGBM tuning or different model approach. Also try GBM tuning and adding more mutation interaction features.