"""
train.py — MODIFIABLE.
Model training, automated feature selection (via model internals),
cross-validated evaluation, feature importance analysis, and submission.
"""

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import lightgbm as lgb
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
)

from prepare import (
    load_raw_data,
    evaluate,
    get_cv_folds,
    make_survival_array,
    N_FOLDS,
    RANDOM_STATE,
    TAU,
)

from data_transform import build_features, AutoFeatureBuilder

warnings.filterwarnings("ignore")


# =====================================================================
# LIGHTGBM SURVIVAL PROXY
# =====================================================================
class LGBMSurvivalProxy:
    """Wraps LightGBM regression to produce survival risk scores.
    Predicts OS_YEARS; risk = negative predicted time."""

    def __init__(self, event_weight=2.0, **params):
        self.event_weight = event_weight
        defaults = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
        defaults.update(params)
        self.params = defaults
        self.model = None

    def fit(self, X, y_surv):
        times = y_surv["OS_YEARS"]
        events = y_surv["OS_STATUS"]
        weights = np.where(events, self.event_weight, 1.0)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, times, sample_weight=weights)
        return self

    def predict(self, X):
        return -self.model.predict(X)

    @property
    def feature_importances_(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None


# =====================================================================
# MODEL FACTORY — fresh instances each fold
# =====================================================================
def create_models():
    """Return dict of {name: model} — one fresh set per CV fold."""
    return {
        "coxph": CoxPHSurvivalAnalysis(alpha=1.0),
        "gbsurv": GradientBoostingSurvivalAnalysis(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "rsf": RandomSurvivalForest(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lgbm": LGBMSurvivalProxy(event_weight=2.0),
    }


# Minimum mean CV C-index to include a model in ensemble
MIN_CINDEX = 0.52


# =====================================================================
# MAIN
# =====================================================================
def run():
    t0 = time.time()
    print("=" * 65)
    print("  BLOOD CANCER SURVIVAL — Auto-Discovery Pipeline")
    print("=" * 65)

    # ---- 1. Load data ---------------------------------------------------
    print("\n[1/7] Loading data …")
    data = load_raw_data()
    target = data["target"]

    # ---- 2. Auto-discover features --------------------------------------
    print("[2/7] Auto-discovering feature vocabulary from training data …")
    train_feat, test_feat, builder = build_features(data, builder_params=dict(
        min_gene_patients=5,
        min_cyto_patients=8,
        min_effect_patients=5,
        n_top_comut_genes=20,
        min_comut_patients=8,
    ))
    print(f"       Discovered: {len(builder.genes_)} genes, "
          f"{len(builder.cyto_tokens_)} cyto tokens, "
          f"{len(builder.effects_)} effects, "
          f"{len(builder.comut_pairs_)} co-mut pairs, "
          f"{len(builder.chroms_)} chromosomes")

    # ---- 3. Transform features ------------------------------------------
    print("[3/7] Building feature matrices …")

    # Align with target
    merged = train_feat.merge(target, on="ID", how="inner")
    merged = merged.sort_values("ID").reset_index(drop=True)
    test_feat = test_feat.sort_values("ID").reset_index(drop=True)

    feature_cols = [c for c in train_feat.columns if c != "ID"]
    X_all = merged[feature_cols].values.astype(np.float64)
    y_all = make_survival_array(merged[["OS_STATUS", "OS_YEARS"]])
    X_test = test_feat[feature_cols].values.astype(np.float64)
    test_ids = test_feat["ID"].values

    n_train, n_feat_raw = X_all.shape
    n_test = X_test.shape[0]
    n_events = int(merged["OS_STATUS"].sum())
    print(f"       Train: {n_train} patients ({n_events} events)")
    print(f"       Test:  {n_test} patients")
    print(f"       Raw features: {n_feat_raw}")

    # ---- 4. Variance filter (target-agnostic, removes degenerate cols) --
    print("[4/7] Applying variance threshold …")
    # Fit on training data only (using median imputation for NaN handling)
    imp_temp = SimpleImputer(strategy="median")
    X_temp = imp_temp.fit_transform(X_all)
    vt = VarianceThreshold(threshold=0.001)
    vt.fit(X_temp)
    keep_mask = vt.get_support()
    feature_cols_filtered = [c for c, k in zip(feature_cols, keep_mask) if k]
    n_dropped = n_feat_raw - len(feature_cols_filtered)
    print(f"       Kept {len(feature_cols_filtered)} features, "
          f"dropped {n_dropped} near-constant")

    X_all = X_all[:, keep_mask]
    X_test = X_test[:, keep_mask]
    feature_cols = feature_cols_filtered
    del X_temp, imp_temp

    # ---- 5. Cross-validation --------------------------------------------
    print(f"\n[5/7] {N_FOLDS}-fold stratified CV …")
    folds = get_cv_folds(merged[["OS_STATUS", "OS_YEARS"]])

    model_names = list(create_models().keys())
    cv_scores = {name: [] for name in model_names}
    oof_preds = {name: np.full(n_train, np.nan) for name in model_names}
    test_preds = {name: np.zeros(n_test) for name in model_names}

    for fi, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n  ── Fold {fi + 1}/{N_FOLDS}  "
              f"(train={len(tr_idx)}, val={len(va_idx)}) ──")

        # Per-fold imputation + scaling (no leakage)
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        Xtr = scl.fit_transform(imp.fit_transform(X_all[tr_idx]))
        Xva = scl.transform(imp.transform(X_all[va_idx]))
        Xte = scl.transform(imp.transform(X_test))
        ytr = y_all[tr_idx]
        yva = y_all[va_idx]

        models = create_models()
        for name, model in models.items():
            try:
                model.fit(Xtr, ytr)
                pred_va = model.predict(Xva)
                pred_te = model.predict(Xte)

                score = evaluate(ytr, yva, pred_va, tau=TAU)
                cv_scores[name].append(score)
                oof_preds[name][va_idx] = pred_va
                test_preds[name] += pred_te / N_FOLDS

                print(f"    {name:12s}  C-index = {score:.4f}")
            except Exception as exc:
                cv_scores[name].append(0.5)
                print(f"    {name:12s}  FAILED — {exc}")

    # ---- 6. Ensemble -----------------------------------------------------
    print("\n" + "=" * 65)
    print("[6/7] Model summary & ensemble")
    print("=" * 65)

    # Select models above threshold
    best_models = []
    for name in model_names:
        arr = np.array(cv_scores[name])
        mean_s = arr.mean()
        std_s = arr.std()
        tag = "INCLUDED" if mean_s >= MIN_CINDEX else "EXCLUDED"
        if mean_s >= MIN_CINDEX:
            best_models.append(name)
        print(f"  {name:12s}:  {mean_s:.4f} ± {std_s:.4f}  [{tag}]")

    if not best_models:
        print("  ⚠ No model above threshold; using all.")
        best_models = model_names

    # Auto-weight by CV score
    raw_w = {n: np.mean(cv_scores[n]) for n in best_models}
    total_w = sum(raw_w.values())
    weights = {n: w / total_w for n, w in raw_w.items()}
    print(f"\n  Ensemble weights: { {n: f'{w:.3f}' for n, w in weights.items()} }")

    # Rank-based blending (scale-invariant)
    ensemble_test = np.zeros(n_test)
    ensemble_oof = np.zeros(n_train)

    for name, w in weights.items():
        ensemble_test += w * rankdata(test_preds[name])

        oof = oof_preds[name].copy()
        nan_mask = np.isnan(oof)
        if nan_mask.any():
            oof[nan_mask] = np.nanmedian(oof)
        ensemble_oof += w * rankdata(oof)

    try:
        oof_score = evaluate(y_all, y_all, ensemble_oof, tau=TAU)
    except Exception:
        oof_score = float("nan")

    print(f"\n  >>> ENSEMBLE OOF C-index: {oof_score:.4f} <<<")

    # ---- 7. Feature importance (what the model discovered) ---------------
    print(f"\n[7/7] Feature importance (GBM tells us what matters) …")

    # Train final GBM on all data for importance analysis
    imp_final = SimpleImputer(strategy="median")
    scl_final = StandardScaler()
    X_final = scl_final.fit_transform(imp_final.fit_transform(X_all))

    try:
        final_gbm = GradientBoostingSurvivalAnalysis(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10, subsample=0.8,
            random_state=RANDOM_STATE,
        )
        final_gbm.fit(X_final, y_all)
        importances = final_gbm.feature_importances_
        top_idx = np.argsort(importances)[::-1][:30]
        print("\n  Top 30 features (auto-discovered by GBM):")
        for rank, idx in enumerate(top_idx, 1):
            if importances[idx] > 0:
                print(f"    {rank:3d}. {feature_cols[idx]:45s}  "
                      f"importance = {importances[idx]:.4f}")
    except Exception as exc:
        print(f"  Could not compute importances: {exc}")

    # Also show LightGBM importances
    try:
        final_lgbm = LGBMSurvivalProxy(event_weight=2.0)
        final_lgbm.fit(X_final, y_all)
        lgbm_imp = final_lgbm.feature_importances_
        if lgbm_imp is not None:
            top_lgbm = np.argsort(lgbm_imp)[::-1][:20]
            print("\n  Top 20 features (auto-discovered by LightGBM):")
            for rank, idx in enumerate(top_lgbm, 1):
                if lgbm_imp[idx] > 0:
                    print(f"    {rank:3d}. {feature_cols[idx]:45s}  "
                          f"importance = {lgbm_imp[idx]}")
    except Exception:
        pass

    # ---- Submission ------------------------------------------------------
    submission = pd.DataFrame({"risk_score": ensemble_test}, index=test_ids)
    submission.index.name = "ID"
    submission.to_csv("submission.csv")
    print(f"\n  Saved submission.csv ({len(submission)} rows)")

    # ---- Final report ----------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  FINAL REPORT")
    print("=" * 65)
    for name in model_names:
        arr = np.array(cv_scores[name])
        print(f"  {name:12s}: {arr.mean():.4f} ± {arr.std():.4f}  "
              f"({', '.join(f'{x:.4f}' for x in arr)})")
    print(f"  {'ENSEMBLE':12s}: {oof_score:.4f}  (OOF, rank-blended)")
    print(f"  Features:     {len(feature_cols)} (after variance filter)")
    print(f"  Folds:        {N_FOLDS}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print("=" * 65)

    return {
        "cv_scores": cv_scores,
        "oof_cindex": oof_score,
        "n_features": len(feature_cols),
        "weights": weights,
    }


if __name__ == "__main__":
    results = run()