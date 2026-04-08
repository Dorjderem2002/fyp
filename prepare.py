"""
prepare.py — READ-ONLY.
Data loading, evaluation, CV utilities.

Feature engineering lives in data_transform.py (the modifiable file).
This file provides the fixed evaluation harness and data loading.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.util import Surv

# Re-export from data_transform so existing code keeps working
from data_transform import AutoFeatureBuilder, build_features  # noqa: F401

warnings.filterwarnings("ignore")

# =====================================================================
# CONSTANTS
# =====================================================================
RANDOM_STATE = 42
N_FOLDS = 5
TAU = 7.0
DATA_DIR = os.environ.get("DATA_DIR", ".")

# =====================================================================
# FILE DISCOVERY
# =====================================================================
_SEARCH = [
    DATA_DIR,
    os.path.join(DATA_DIR, "data"),
    os.path.join(DATA_DIR, "X_train"),
    os.path.join(DATA_DIR, "X_test"),
    "data",
    "X_train",
    "X_test",
]


def _find(name, alts=None):
    for d in _SEARCH:
        for n in [name] + (alts or []):
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"Cannot find {name} (also tried {alts})")


# =====================================================================
# DATA LOADING
# =====================================================================
def load_raw_data():
    d = {}
    d["clinical_train"] = pd.read_csv(_find("clinical_train.csv"))
    d["clinical_test"] = pd.read_csv(_find("clinical_test.csv"))
    d["molecular_train"] = pd.read_csv(_find("molecular_train.csv"))
    d["molecular_test"] = pd.read_csv(_find("molecular_test.csv"))
    t = pd.read_csv(_find("target_train.csv", ["Y_train.csv"]))
    t.dropna(subset=["OS_YEARS", "OS_STATUS"], inplace=True)
    t["OS_YEARS"] = pd.to_numeric(t["OS_YEARS"], errors="coerce")
    t["OS_STATUS"] = t["OS_STATUS"].astype(bool)
    d["target"] = t.reset_index(drop=True)
    return d


# =====================================================================
# EVALUATION
# =====================================================================
def evaluate(y_train_surv, y_test_surv, risk_scores, tau=TAU):
    """IPCW C-index with fallback to standard C-index."""
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    try:
        return concordance_index_ipcw(
            y_train_surv, y_test_surv, risk_scores, tau=tau
        )[0]
    except Exception:
        ev = np.array([s[0] for s in y_test_surv])
        ti = np.array([s[1] for s in y_test_surv])
        return concordance_index_censored(ev, ti, risk_scores)[0]


# =====================================================================
# CROSS-VALIDATION
# =====================================================================
def get_cv_folds(target_df, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    labels = target_df["OS_STATUS"].astype(int).values
    return list(skf.split(target_df, labels))


def make_survival_array(df):
    d = df[["OS_STATUS", "OS_YEARS"]].copy()
    d["OS_STATUS"] = d["OS_STATUS"].astype(bool)
    d["OS_YEARS"] = d["OS_YEARS"].astype(float)
    return Surv.from_dataframe("OS_STATUS", "OS_YEARS", d)


# =====================================================================
# MAIN — sanity check
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  prepare.py — sanity check")
    print("=" * 60)

    data = load_raw_data()
    for k, v in data.items():
        print(f"  {k:20s}  {str(v.shape):>14s}")

    train_feat, test_feat, builder = build_features(data)
    fcols = [c for c in train_feat.columns if c != "ID"]

    print(f"\nAuto-discovered vocabulary:")
    print(f"  Genes:          {len(builder.genes_)}")
    print(f"  Effects:        {len(builder.effects_)}")
    print(f"  Cyto tokens:    {len(builder.cyto_tokens_)}")
    print(f"  Co-mut pairs:   {len(builder.comut_pairs_)}")
    print(f"  Chromosomes:    {len(builder.chroms_)}")

    print(f"\nTrain: {train_feat.shape[0]} patients x {len(fcols)} features")
    print(f"Test:  {test_feat.shape[0]} patients x {len([c for c in test_feat.columns if c != 'ID'])} features")
    print(f"Train NaN frac: {train_feat[fcols].isna().mean().mean():.4f}")
    print(f"\nAll OK.  Run  python train.py  to train.\n")
