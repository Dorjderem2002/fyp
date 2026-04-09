"""
train.py — MODIFIABLE.
Fast pipeline: CoxPH + LightGBM. Focus on features in data_transform.py.
"""

import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sksurv.linear_model import CoxPHSurvivalAnalysis

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


class LGBMSurvivalProxy:
    def __init__(self, event_weight=2.5, **params):
        self.event_weight = event_weight
        defaults = dict(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            num_leaves=25,
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


def create_models():
    return {
        "coxph": CoxPHSurvivalAnalysis(alpha=1.0),
        "lgbm": LGBMSurvivalProxy(),
    }


MIN_CINDEX = 0.52


def run():
    t0 = time.time()
    print("=" * 65 + "\n  BLOOD CANCER SURVIVAL — Fast Pipeline\n" + "=" * 65)

    data = load_raw_data()
    target = data["target"]

    print("[Features] Building …")
    train_feat, test_feat, builder = build_features(
        data,
        builder_params=dict(
            min_gene_patients=5,
            min_cyto_patients=8,
            min_effect_patients=5,
            n_top_comut_genes=20,
            min_comut_patients=8,
        ),
    )

    merged = (
        train_feat.merge(target, on="ID", how="inner")
        .sort_values("ID")
        .reset_index(drop=True)
    )
    test_feat = test_feat.sort_values("ID").reset_index(drop=True)

    feature_cols = [c for c in train_feat.columns if c != "ID"]
    X_all = merged[feature_cols].values.astype(np.float64)
    y_all = make_survival_array(merged[["OS_STATUS", "OS_YEARS"]])
    X_test = test_feat[feature_cols].values.astype(np.float64)
    test_ids = test_feat["ID"].values

    print(f"  Raw features: {X_all.shape[1]}")

    imp_temp = SimpleImputer(strategy="median")
    X_temp = imp_temp.fit_transform(X_all)
    vt = VarianceThreshold(threshold=0.005)
    vt.fit(X_temp)
    keep_mask = vt.get_support()
    feature_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
    X_all = X_all[:, keep_mask]
    X_test = X_test[:, keep_mask]
    del X_temp, imp_temp
    print(f"  After variance filter: {len(feature_cols)} features")

    folds = get_cv_folds(merged[["OS_STATUS", "OS_YEARS"]])
    model_names = list(create_models().keys())
    cv_scores = {name: [] for name in model_names}
    oof_preds = {name: np.full(X_all.shape[0], np.nan) for name in model_names}
    test_preds = {name: np.zeros(X_test.shape[0]) for name in model_names}

    for fi, (tr_idx, va_idx) in enumerate(folds):
        print(f"  Fold {fi + 1}/{N_FOLDS}")
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        Xtr = scl.fit_transform(imp.fit_transform(X_all[tr_idx]))
        Xva = scl.transform(imp.transform(X_all[va_idx]))
        Xte = scl.transform(imp.transform(X_test))
        ytr = y_all[tr_idx]

        for name, model in create_models().items():
            try:
                model.fit(Xtr, ytr)
                pred_va = model.predict(Xva)
                pred_te = model.predict(Xte)
                cv_scores[name].append(evaluate(ytr, y_all[va_idx], pred_va, tau=TAU))
                oof_preds[name][va_idx] = pred_va
                test_preds[name] += pred_te / N_FOLDS
                print(f"    {name:12s}  C={cv_scores[name][-1]:.4f}")
            except Exception as exc:
                cv_scores[name].append(0.5)
                print(f"    {name:12s}  FAILED — {exc}")

    print("\n" + "=" * 65)
    best_models = []
    for name in model_names:
        arr = np.array(cv_scores[name])
        mean_s = arr.mean()
        tag = "INC" if mean_s >= MIN_CINDEX else "EXC"
        if mean_s >= MIN_CINDEX:
            best_models.append(name)
        print(f"  {name:12s}:  {mean_s:.4f} ± {arr.std():.4f}  [{tag}]")
    if not best_models:
        best_models = model_names

    raw_w = {n: np.mean(cv_scores[n]) for n in best_models}
    total_w = sum(raw_w.values())
    weights = {n: w / total_w for n, w in raw_w.items()}

    ensemble_test = np.zeros(X_test.shape[0])
    ensemble_oof = np.zeros(X_all.shape[0])
    for name, w in weights.items():
        ensemble_test += w * rankdata(test_preds[name])
        oof = oof_preds[name].copy()
        oof[np.isnan(oof)] = np.nanmedian(oof)
        ensemble_oof += w * rankdata(oof)
    oof_score = evaluate(y_all, y_all, ensemble_oof, tau=TAU)
    print(f"\n  >>> ENSEMBLE OOF C-index: {oof_score:.4f} <<<")

    elapsed = time.time() - t0
    print(
        f"\n  ENSEMBLE: {oof_score:.4f}  Features: {len(feature_cols)}  Time: {elapsed:.0f}s"
    )
    print("=" * 65)
    return {
        "cv_scores": cv_scores,
        "oof_cindex": oof_score,
        "n_features": len(feature_cols),
        "weights": weights,
    }


if __name__ == "__main__":
    results = run()
