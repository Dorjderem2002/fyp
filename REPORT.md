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

---
### Experiment 3: GBM/RSF tuning + VAF-weighted driver features
**Date:** 2026-04-09
**Hypothesis:** Tuning the strongest models (GBM, RSF) with more trees and relaxed min samples will improve their C-index, and adding FLT3-ITD VAF and driver gene VAF features will add signal.
**Changes:**
- train.py: GBM n_estimators 300→500, lr 0.05→0.03, min_samples_split 20→15, min_samples_leaf 10→8
- train.py: RSF n_estimators 300→500, max_depth 8→10, min_samples_split 20→15, min_samples_leaf 10→6
- data_transform.py: Added flt3_itd_vaf, max_driver_vaf, n_driver_mutations features
**Result:** C-index = 0.7143 (prev: 0.7139) | status: keep
**Analysis:** GBM and RSF tuning helped significantly. GBM: 0.7130→0.7154, RSF: 0.7129→0.7146. More trees + lower LR helped both. Driver VAF features also contribute signal. LightGBM remains weak at 0.6841 — needs more work or may need a different approach entirely. Elapsed time increased to ~7.3 min but within budget.
**Next idea:** Try more aggressive LightGBM tuning or replace it. Also try adding interaction features between clinical and molecular features (e.g. blast% × FLT3).

---
### Experiment 4: Clinical-molecular interactions + LightGBM event_weight tuning
**Date:** 2026-04-09
**Hypothesis:** Interacting clinical features (BM_BLAST, WBC, etc.) with key mutation flags will capture that the same mutation has different prognostic impact depending on disease burden. Higher event_weight for LightGBM may help it focus more on events.
**Changes:**
- data_transform.py: Added clinical × molecular interaction features (5 clinical cols × 3 mutation flags = 30 new features)
- train.py: LightGBM event_weight 2.0→3.0
**Result:** C-index = 0.7149 (prev: 0.7143) | status: keep
**Analysis:** Clinical-molecular interactions helped. RSF improved to 0.7163 (best single model so far). LightGBM slightly improved with higher event_weight. GBM stable at 0.7150. The interactions (BM_BLAST × FLT3_ITD, WBC × TP53, etc.) capture that mutations have different prognostic meaning depending on disease burden. 612 features, still reasonable.
**Next idea:** Try adding a second RSF with different hyperparameters for ensemble diversity and tuning GBM subsample.

---
### Experiment 9: Second RSF model (rsf2) for ensemble diversity
**Date:** 2026-04-09
**Hypothesis:** Adding a second RSF with different hyperparameters (deeper, more trees, different max_features) will improve ensemble through diversity. RSF is the best single model so different RSF configs may capture different patterns.
**Changes:**
- train.py: Added rsf2 — n_estimators=700, max_depth=12, min_samples_split=10, min_samples_leaf=4, max_features=0.5
**Result:** C-index = 0.7171 (prev: 0.7149) | status: keep
**Analysis:** Big improvement! Second RSF with different hyperparameters (rsf2: 0.7149) adds valuable ensemble diversity. Both RSFs together with GBM create a stronger ensemble. But run time increased to 21 min — could be an issue if we need faster iterations. The two RSF models have complementary strengths: rsf (sqrt features, shallower) vs rsf2 (0.5 features, deeper).
**Next idea:** Try adding a second GBM with different hyperparameters for more diversity. Also consider reducing rsf2 n_estimators to speed up (e.g. 500 instead of 700) while checking if C-index holds.