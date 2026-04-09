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
**Next idea:** Try adding a second GBM with different hyperparameters for more diversity. Also try reducing rsf2 n_estimators to speed up.

---
### Experiment 10: Second GBM model (gbsurv2) for ensemble diversity
**Date:** 2026-04-09
**Hypothesis:** Adding a second GBM with different hyperparameters (shallower, faster LR, lower subsample) will add ensemble diversity like rsf2 did.
**Changes:**
- train.py: Added gbsurv2 — n_estimators=300, max_depth=2, learning_rate=0.05, min_samples_split=20, min_samples_leaf=10, subsample=0.7
**Result:** C-index = 0.7177 (prev: 0.7171) | status: keep
**Analysis:** Another improvement! gbsurv2 (0.7139) adds diversity. 6 models now in ensemble. The pattern is clear: model diversity is the most powerful lever right now. gbsurv2 is shallower (depth=2) with different LR and subsample than gbsurv — it captures different patterns. Run time 22 min, still within budget.
**Next idea:** Try adding yet more model diversity — a third RSF or GBM variant. Also try raising MIN_CINDEX to drop weak models.

---
### Experiment 11: Third RSF + third GBM for more diversity
**Date:** 2026-04-09
**Hypothesis:** More model variants with diverse hyperparameters continue to improve the ensemble. rsf3 (no max_depth, log2 features) and gbsurv3 (depth=4, very low LR) capture different patterns.
**Changes:**
- train.py: Added rsf3 — n_estimators=500, max_depth=None, min_samples_split=20, min_samples_leaf=8, max_features="log2"
- train.py: Added gbsurv3 — n_estimators=400, max_depth=4, learning_rate=0.01, min_samples_split=10, min_samples_leaf=5, subsample=0.9
**Result:** C-index = 0.7184 (prev: 0.7177) | status: keep
**Analysis:** Continued improvement with more model diversity. rsf3 (0.7142) and gbsurv3 (0.7148) add useful diversity. 8 models now, all included. Run time 26 min — still within budget but growing. CoxPH (0.6638) is the weakest and may be dragging down the ensemble slightly.
**Next idea:** Try raising MIN_CINDEX to drop CoxPH (weakest model at 0.664). Also try one more model variant.

---
### Experiment 12: Raise MIN_CINDEX to 0.68 to exclude CoxPH
**Date:** 2026-04-09
**Hypothesis:** CoxPH (0.664) is dragging down the ensemble. Excluding it should improve the ensemble by letting stronger models dominate.
**Changes:**
- train.py: MIN_CINDEX 0.52→0.68
**Result:** C-index = 0.7186 (prev: 0.7184) | status: keep
**Analysis:** Small improvement. Dropping CoxPH (0.664) helps slightly — it was adding noise to the ensemble. 7 models remain. LightGBM (0.686) is now the weakest included model.
**Next idea:** Try dropping LightGBM too (raise MIN_CINDEX to 0.69) or try another feature engineering approach. Also try adding more interaction features or mutation count × clinical interactions beyond just the 3 gene flags.