# autoresearch-survival

This is an autonomous experimentation protocol for a Kaggle-style survival analysis problem.

## Mission

Build a model that maximizes robust cross-validation performance for right-censored survival prediction while explicitly minimizing overfitting risk.

Primary objective:

- maximize out-of-fold IPCW C-index (higher is better)

Secondary objectives:

- reduce fold variance (stability)
- keep train-vs-CV gap small
- keep implementation simple and reproducible

## Competition Context

Data sources:

- `X_train/clinical_train.csv`: one row per patient with clinical covariates
- `X_train/molecular_train.csv`: one row per mutation event
- `target_train.csv`: `ID`, `OS_YEARS`, `OS_STATUS`
- `X_test/clinical_test.csv` and `X_test/molecular_test.csv`

Task:

- predict a `risk_score` per patient ID
- ranking quality matters, not absolute score scale
- metric is IPCW C-index truncated at 7 years

## Setup

To start a new run:

1. Propose a run tag from today's date (example: `apr9-surv`) and confirm branch `autoresearch/<tag>` does not exist.
2. Create branch from current mainline: `git checkout -b autoresearch/<tag>`.
3. Read in-scope files for full context:
   - `README.md`
   - this file (`program.md`)
   - benchmark notebook(s) in `notebooks/`
   - any training scripts already used in this folder
4. Verify data is present and readable:
   - `X_train/clinical_train.csv`
   - `X_train/molecular_train.csv`
   - `target_train.csv`
5. Initialize tracking files if missing:
   - `results.tsv` with header row only
   - `run.log` (overwritten each run)
6. Confirm setup and begin with a baseline run.

## Rules Of Engagement

What you CAN do:

- edit modeling code, feature engineering, CV design, hyperparameters, ensembling, and calibration transforms
- create helper scripts or notebooks for reproducible training and inference
- refactor for clarity if behavior is preserved

What you CANNOT do:

- leak test information into training or validation
- use patient IDs as predictive signals
- evaluate model choices on test labels (not available)
- hand-tune on a single lucky fold only

## Overfitting Guardrails (Mandatory)

Every experiment must satisfy all checks:

1. Use patient-level splitting only (no patient appears in both train and validation).
2. Preserve event/censor balance across folds whenever possible (`OS_STATUS` stratification).
3. Run at least 5 folds for selection decisions.
4. Report per-fold IPCW C-index, mean, and std.
5. Report train metric and CV metric; track generalization gap.
6. Prefer improvements that hold across multiple random seeds.

Suggested anti-leakage split hierarchy (highest priority first):

1. Group by `CENTER` if center shift is significant.
2. Stratify by `OS_STATUS` and binned `OS_YEARS`.
3. If both cannot be done simultaneously, log the tradeoff explicitly.

## Baseline First

First run must establish the baseline pipeline without aggressive tuning.

Minimum baseline:

- simple clinical-only model
- simple clinical + molecular aggregate model
- consistent CV protocol and deterministic seed

## Experiment Output

For each run, print and log this summary block:

```
---
cv_ipcw_cindex_mean: 0.000000
cv_ipcw_cindex_std:  0.000000
train_ipcw_cindex:   0.000000
generalization_gap:  0.000000
folds:               5
seeds:               1
peak_vram_mb:        0.0
total_seconds:       0.0
```

Where:

- `generalization_gap = train_ipcw_cindex - cv_ipcw_cindex_mean`
- lower gap is better, as long as CV mean does not drop materially

## Logging Results

Track every completed attempt in `results.tsv` (tab-separated only):

```
commit	cv_mean	cv_std	train	gap	memory_gb	status	description
```

Column definitions:

1. short commit hash
2. CV IPCW C-index mean
3. CV IPCW C-index std
4. train IPCW C-index
5. train-CV gap
6. peak memory in GB (`peak_vram_mb / 1024`, 1 decimal)
7. status: `keep`, `discard`, `crash`
8. short description (no tabs)

Example:

```
commit	cv_mean	cv_std	train	gap	memory_gb	status	description
a1b2c3d	0.684200	0.021300	0.715100	0.030900	8.4	keep	baseline clinical+molecular aggregates
b2c3d4e	0.691500	0.017600	0.727800	0.036300	9.1	keep	add center-aware fold split and VAF robust stats
c3d4e5f	0.688100	0.028900	0.752000	0.063900	9.0	discard	deeper booster overfits
d4e5f6g	0.000000	0.000000	0.000000	0.000000	0.0	crash	invalid fold construction
```

Do not commit `results.tsv`.

## Keep/Discard Policy

Keep a run only if it improves model quality robustly.

Keep when ALL are true:

1. `cv_mean` improves by at least 0.0010, OR improves by at least 0.0005 with reduced `cv_std` and no larger gap.
2. `gap` does not materially worsen (rule of thumb: increase <= 0.01 unless CV gain is substantial).
3. no fold collapses (no severe outlier fold without explanation).

Discard when ANY are true:

- CV does not improve materially
- CV variance increases sharply
- train metric rises but CV stagnates/drops
- approach adds major complexity for marginal gain

## Experiment Loop

LOOP CONTINUOUSLY:

1. Read current git branch and commit.
2. Select exactly one experimental idea.
3. Implement and commit.
4. Run training/evaluation, redirecting all logs to `run.log`.
5. Parse summary metrics from the log.
6. If run failed, inspect tail of log, fix obvious issues, retry a few times max.
7. Append result to `results.tsv`.
8. Keep commit only if policy says `keep`; otherwise hard revert to previous good commit.

Use this idea order:

1. data leakage prevention and robust CV
2. feature quality and aggregation robustness
3. regularization and model simplicity
4. seed stability and ensembling
5. architecture or objective changes

## Recommended Experiment Menu

Prioritize high signal / low complexity ideas:

- robust molecular aggregation per patient (counts, unique genes, max/mean VAF, high-risk gene flags)
- center-aware validation strategy
- target transformations for survival ranking stability
- stronger regularization (subsample, colsample, L1/L2, min data in leaf)
- monotonic constraints only when clinically justified
- seed averaging for final risk score

Avoid first:

- very deep/high-capacity models with weak regularization
- many handcrafted interactions without validation evidence
- over-optimized single-fold tuning

## Submission Rules

Final submission file must contain:

- index or column: `ID`
- prediction column: `risk_score`

Higher risk score should imply higher death risk (shorter survival).

## Stop Conditions

Continue autonomously until interrupted by human, unless blocked by:

- unrecoverable environment failure
- corrupted input data
- repeated crashes with no actionable stack trace

If blocked, write a short blocker note with attempted fixes and the last stable commit.

