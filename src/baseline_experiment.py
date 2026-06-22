"""
baseline_experiment.py
Preliminary baseline: classical anomaly detectors applied directly
to engineered node features (no GNN yet).
Computes ROC-AUC against known-vulnerable packages.
"""

import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/igalimi1/thesis/data"


def main():
    # load features
    feats = pd.read_csv(f"{DATA_DIR}/node_features.csv")
    print(f"Loaded {len(feats)} feature rows")

    # ground-truth labels: positive if it has any CVE
    y = (feats["cve_count"] > 0).astype(int).values
    print(f"Positives: {y.sum()}, Normals: {(y==0).sum()}")

    # Use STRUCTURAL features only — excluding cve_count and max_cvss,
    # which directly encode the label (avoids evaluation leakage)
    structural_cols = [
        "out_degree", "in_degree",
        "version_age", "recency", "is_latest",
    ]
    X = feats[structural_cols].values
    X = StandardScaler().fit_transform(X)

    results = {}

    # ---- Isolation Forest ----
    iso = IsolationForest(random_state=42, n_estimators=100)
    iso.fit(X)
    iso_scores = -iso.decision_function(X)
    results["IsolationForest"] = roc_auc_score(y, iso_scores)

    # ---- Local Outlier Factor ----
    lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
    lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_
    results["LOF"] = roc_auc_score(y, lof_scores)

    # ---- One-Class SVM (on a sample; OCSVM is slow on 136k) ----
    idx = np.random.RandomState(42).choice(
        len(X), size=min(20000, len(X)), replace=False
    )
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
    ocsvm.fit(X[idx])
    ocsvm_scores = -ocsvm.decision_function(X)
    results["OneClassSVM"] = roc_auc_score(y, ocsvm_scores)

    # ---- Report ----
    print("\n=== Baseline ROC-AUC (structural features only) ===")
    for name, auc in results.items():
        print(f"{name:20s}: {auc:.4f}")


if __name__ == "__main__":
    main()