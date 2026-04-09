"""
train.py — MODIFIABLE.
Simplified pipeline for fast iteration. Focus on features in data_transform.py.
"""

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

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


def create_models():
    return {
        "gbsurv": GradientBoostingSurvivalAnalysis(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "rsf": RandomSurvivalForest(
            n_estimators=500,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "rsf2": RandomSurvivalForest(
            n_estimators=700,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


MIN_CINDEX = 0.52


def run():
    t0 = time.time()
    print("=" * 65 + "\n  BLOOD CANCER SURVIVAL — Pipeline\n" + "=" * 65)

    data = load_raw_data()
    target = data["target"]

    print("[2/6] Building features …")
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
    print(
        f"  Genes: {len(builder.genes_)}  Cyto: {len(builder.cyto_tokens_)}  "
        f"Effects: {len(builder.effects_)}  Comut: {len(builder.comut_pairs_)}"
    )

    print("[3/6] Feature matrices …")
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
    n_train, n_feat_raw = X_all.shape
    print(f"  Train: {n_train}  Raw features: {n_feat_raw}")

    print("[4/6] Variance filter …")
    imp_temp = SimpleImputer(strategy="median")
    X_temp = imp_temp.fit_transform(X_all)
    vt = VarianceThreshold(threshold=0.005)
    vt.fit(X_temp)
    keep_mask = vt.get_support()
    feature_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
    print(
        f"  Kept {len(feature_cols)} features, dropped {n_feat_raw - len(feature_cols)}"
    )
    X_all = X_all[:, keep_mask]
    X_test = X_test[:, keep_mask]
    del X_temp, imp_temp

    print(f"[5/6] {N_FOLDS}-fold CV …")
    folds = get_cv_folds(merged[["OS_STATUS", "OS_YEARS"]])
    model_names = list(create_models().keys())
    cv_scores = {name: [] for name in model_names}
    oof_preds = {name: np.full(n_train, np.nan) for name in model_names}
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
    ensemble_oof = np.zeros(n_train)
    for name, w in weights.items():
        ensemble_test += w * rankdata(test_preds[name])
        oof = oof_preds[name].copy()
        oof[np.isnan(oof)] = np.nanmedian(oof)
        ensemble_oof += w * rankdata(oof)
    oof_score = evaluate(y_all, y_all, ensemble_oof, tau=TAU)
    print(f"\n  >>> ENSEMBLE OOF C-index: {oof_score:.4f} <<<")

    # Feature importance
    imp_f = SimpleImputer(strategy="median")
    scl_f = StandardScaler()
    X_f = scl_f.fit_transform(imp_f.fit_transform(X_all))
    try:
        gbm_f = GradientBoostingSurvivalAnalysis(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
        gbm_f.fit(X_f, y_all)
        imps = gbm_f.feature_importances_
        top_idx = np.argsort(imps)[::-1][:20]
        print("\n  Top 20 features:")
        for rank, idx in enumerate(top_idx, 1):
            if imps[idx] > 0:
                print(f"    {rank:3d}. {feature_cols[idx]:50s} {imps[idx]:.4f}")
    except Exception:
        pass

    submission = pd.DataFrame({"risk_score": ensemble_test}, index=test_ids)
    submission.index.name = "ID"
    submission.to_csv("submission.csv")

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
