# Kaggle Competition: Blood Cancer Survival Prediction

Autonomous AI agent pipeline for the QRT/Institut Gustave Roussy blood cancer survival challenge.

## Goal

**Maximize the IPCW C-index** (concordance index for right-censored data with inverse probability of censoring weights, clipped at tau=7 years). Higher is better. 0.5 = random, 1.0 = perfect.

## Competition Summary

Predict overall survival (OS) for 1,193 test patients diagnosed with blood cancer (adult myeloid leukemias). The prediction is a **risk score** — only the ranking matters, not the scale. Input data: clinical features (blood counts, cytogenetics) and molecular data (somatic mutations with gene, position, VAF, effect).

## Architecture

| File | Role | Modifiable? |
|------|------|-------------|
| `data_transform.py` | Feature engineering: `AutoFeatureBuilder`, `build_features()` | **YES — PRIMARY** |
| `train.py` | Models, hyperparameters, ensemble, CV, submission output | **YES** |
| `prepare.py` | Data loading, evaluation, CV utilities. Imports from `data_transform.py` | **NO — READ-ONLY** |
| `program.md` | This file. Instructions for the agent | **NO** |
| `results.tsv` | Experiment log (untracked by git) | Append only |

### Key interfaces

- `prepare.py` exports: `load_raw_data()`, `evaluate()`, `get_cv_folds()`, `make_survival_array()`, constants (`N_FOLDS`, `RANDOM_STATE`, `TAU`)
- `data_transform.py` exports: `build_features(data, builder_params=None)` → `(train_feat_df, test_feat_df, builder)`, `AutoFeatureBuilder`
- `train.py`: uses both, outputs `submission.csv`

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr9`). The branch `kaggle/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b kaggle/<tag>`
3. **Read all in-scope files**: `data_transform.py`, `train.py`, `prepare.py`, `program.md`
4. **Verify data exists**: Check that `X_train/`, `X_test/`, and `target_train.csv` exist.
5. **Initialize results.tsv** if it doesn't exist (header row only).
6. **Confirm and go**.

## Experimentation Strategy

### Priority order (highest impact first)

1. **Feature engineering in `data_transform.py`** — this is where the biggest gains come from:
   - **CYTOGENETICS**: Rich ISCN karyotype strings. Parse deeper: specific translocations (e.g. t(8;21), t(15;17), inv(16)), monosomal karyotype, complex karyotype (≥3 abnormalities), hypo/hyperdiploidy, specific deletion targets, clone burden from cell counts in brackets like `[15]/[5]`.
   - **Molecular data**: Gene pathway groupings, mutation hotspots, multi-hit patterns, VAF distribution features, driver vs passenger mutation classification.
   - **Clinical interactions**: Domain-specific ratios (e.g. ANC/WBC = neutrophil fraction, blast/WBC), nonlinear transforms.
   - **Risk stratification proxies**: ELN 2017/2022 risk classification features, WHO classification approximations based on mutation + cytogenetic patterns.

2. **Model tuning in `train.py`**:
   - Hyperparameter sweeps for existing models (GBM, RSF, CoxPH, LightGBM)
   - New model families (e.g. XGBoost survival, neural survival models)
   - Ensemble weights and blending strategies
   - Feature selection methods

3. **AutoFeatureBuilder parameters**: thresholds for gene/cyto/effect inclusion.

### What you CAN do

- Modify `data_transform.py` — add features, improve parsing, add new feature families
- Modify `train.py` — change models, hyperparameters, ensemble strategy, feature selection
- Add new helper functions in `data_transform.py`
- Install packages listed in `requirements.txt` (numpy, pandas, scikit-learn, scikit-survival, lightgbm)

### What you CANNOT do

- Modify `prepare.py` — it is the fixed evaluation harness
- Break the interface: `build_features(data)` must return `(train_df, test_df, builder)`
- Break the output: `train.py` must produce `submission.csv` with columns `ID` and `risk_score`
- Add external data or pretrained models
- Install packages not in `requirements.txt` without asking

## Running an Experiment

```bash
python train.py > run.log 2>&1
```

### Reading results

```bash
grep "ENSEMBLE\|FINAL REPORT" run.log
```

The key metric is the **ENSEMBLE OOF C-index** line printed by train.py. This is the out-of-fold cross-validated C-index on the training set — the best proxy for leaderboard score.

To get per-model scores:
```bash
grep "INCLUDED\|EXCLUDED" run.log
```

### If the run crashes

```bash
tail -n 50 run.log
```

## Output Format

train.py prints a summary like:

```
=================================================================
  FINAL REPORT
=================================================================
  coxph       : 0.6823 ± 0.0215  (0.6634, 0.7012, 0.6891, 0.6754, 0.6824)
  gbsurv      : 0.7045 ± 0.0189  (0.6912, 0.7234, 0.7001, 0.6998, 0.7080)
  rsf         : 0.6912 ± 0.0201  (0.6756, 0.7123, 0.6890, 0.6845, 0.6948)
  lgbm        : 0.7102 ± 0.0178  (0.6934, 0.7289, 0.7056, 0.7012, 0.7219)
  ENSEMBLE    : 0.7156  (OOF, rank-blended)
  Features:     287 (after variance filter)
  Folds:        5
  Elapsed:      45.2s
=================================================================
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated). The TSV has a header row and 5 columns:

```
commit	cindex	n_features	status	description
```

1. git commit hash (short, 7 chars)
2. ENSEMBLE OOF C-index (e.g. 0.7156) — use 0.0000 for crashes
3. Number of features after variance filter
4. status: `keep`, `discard`, or `crash`
5. Short text description of what this experiment tried

Example:

```
commit	cindex	n_features	status	description
a1b2c3d	0.7156	287	keep	baseline
b2c3d4e	0.7198	312	keep	add ELN risk proxy features from cytogenetics
c3d4e5f	0.7102	295	discard	remove clinical pairwise products
d4e5f6g	0.0000	0	crash	bad regex in cyto parser
```

## The Experiment Loop

LOOP FOREVER:

1. Look at the current state: `git log --oneline -5` and `tail -5 results.tsv`
2. Choose a hypothesis: feature engineering idea, model change, or hyperparameter tweak
3. Implement it in `data_transform.py` and/or `train.py`
4. `git add data_transform.py train.py && git commit -m "experiment: <description>"`
5. Run: `python train.py > run.log 2>&1`
6. Read results: `grep "ENSEMBLE" run.log` and `grep "Features:" run.log`
7. If grep is empty → crash. Run `tail -n 50 run.log` and attempt fix
8. Log to `results.tsv`
9. **If C-index improved** → keep the commit, advance the branch
10. **If C-index equal or worse** → `git reset --hard HEAD~1` to revert

### Decision rules

- **Keep** if ENSEMBLE C-index improves by any amount
- **Discard** if ENSEMBLE C-index is equal or worse, UNLESS complexity decreased significantly
- **Crash** → diagnose, fix if simple, otherwise revert and move on
- If stuck after 3+ failed experiments on the same idea, move to a different approach

### Feature engineering ideas backlog

Ordered by expected impact:

1. **ELN risk classification proxy**: parse cytogenetics + mutations to approximate ELN 2017/2022 favorable/intermediate/adverse categories
2. **Monosomal karyotype**: detect ≥2 monosomies or 1 monosomy + structural abnormality
3. **Clone burden from cytogenetics**: extract cell counts from bracket notation `[N]` to compute clone fraction
4. **Specific translocation groups**: t(8;21), inv(16)/t(16;16), t(15;17) = favorable; t(6;9), t(v;11q23.3), inv(3)/t(3;3) = adverse
5. **Mutation pathway features**: group genes by pathway (e.g. epigenetic: TET2/DNMT3A/IDH1/IDH2, splicing: SF3B1/SRSF2/U2AF1/ZRSR2, signaling: FLT3/NRAS/KRAS/KIT)
6. **FLT3-ITD specific features**: detect ITD vs TKD from protein change, VAF interaction
7. **NPM1 + FLT3 interaction**: NPM1-mutated without FLT3-ITD = favorable
8. **TP53 multi-hit**: biallelic TP53 via multiple mutations or del(17p) + mutation
9. **Mutation load categories**: low/medium/high based on total mutation count
10. **VAF-weighted gene features**: weight gene indicators by clonal dominance

### Model tuning ideas backlog

1. Tune GBM: n_estimators, max_depth, learning_rate, subsample
2. Tune RSF: n_estimators, max_depth, min_samples_leaf, max_features
3. Tune CoxPH: alpha (0.01 … 10.0)
4. Tune LightGBM: num_leaves, reg_alpha, reg_lambda, colsample_bytree
5. Try XGBoost survival (if installable)
6. Try stacking instead of rank blending
7. Adaptive ensemble weighting

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the data, examine feature importances, try combinations of previous near-misses. The loop runs until the human interrupts you.
