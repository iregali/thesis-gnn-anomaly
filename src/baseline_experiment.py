"""
baseline_experiment.py
Preliminary baseline: classical anomaly detectors applied directly
to engineered package features (no GNN yet).
Evaluates against known-malicious packages (malregistry).

Reports
  1. ROC-AUC on all malware and on "GuardDog-hard" malware
     (malware GuardDog did not flag, including failed scans)
  2. Operating points: detection rate (TPR) at 1% and 5% false positives,
     and at GuardDog's own false-positive rate (like-for-like comparison),
     plus precision at realistic base rates (1 in 200, 1 in 1000 uploads)
  3. Single-feature separability ranking
Results are also saved as CSV in RESULTS_DIR.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

DATA_DIR = "/home/igalimi1/thesis/data"
RESULTS_DIR = "/home/igalimi1/thesis/results"

# identifiers and bookkeeping columns, not features
ID_COLS = ["Name", "Version", "path", "archive_type", "read_failed", "error", "truncated"]

# GuardDog rule groups (source-code rules only)
GD_GROUPS = ["code_execution", "network_exfiltration", "obfuscation",
             "command_abuse", "sensitive_access"]

FPR_TARGETS = [0.01, 0.05]
BASE_RATES = [200, 1000]  # "1 in N uploads is malicious"


def load(cls, label):
    """Static + GuardDog features for one class, merged per archive."""
    static = pd.read_csv(f"{DATA_DIR}/static_{cls}.csv", dtype={"Version": str})
    gd = pd.read_csv(f"{DATA_DIR}/guarddog_{cls}.csv", dtype={"Version": str})

    # Only readable sdists: wheels and unreadable archives exist only in
    # the malicious set, so keeping them would leak the class.
    static = static[(static["archive_type"] == "sdist") & (static["read_failed"] == 0)]

    gd = gd[["path", "scan_failed", "code_issue_count"] + GD_GROUPS].drop_duplicates("path")
    df = static.merge(gd, on="path", how="left")
    df = df.drop_duplicates(subset=["Name", "Version"])  # one archive per release
    df["label"] = label
    print(f"{cls:9s}: {len(df)} packages")
    return df


def tpr_at_fpr(scores, y, fpr):
    """Detection rate when the threshold lets `fpr` of normal packages through."""
    threshold = np.quantile(scores[y == 0], 1 - fpr)
    return np.mean(scores[y == 1] > threshold), np.mean(scores[y == 0] > threshold)


def precision_at(tpr, fpr, n):
    """Share of alerts that are real if 1 in n uploads is malicious."""
    p = 1 / n
    return tpr * p / (tpr * p + fpr * (1 - p)) if (tpr * p + fpr * (1 - p)) else float("nan")


def main():
    # load features
    feats = pd.concat([load("malicious", 1), load("normal", 0)], ignore_index=True)
    y_all = feats["label"].values
    print(f"Positives: {y_all.sum()}, Normals: {(y_all == 0).sum()}")

    # GuardDog columns: MISSING where the scan failed (not "clean"), then 0 for
    # the models. scan_failed itself is never a feature: failure rates differ
    # by class, and the two runs used different timeouts.
    gd_cols = ["code_issue_count"] + GD_GROUPS
    failed = feats["scan_failed"].fillna(1) == 1
    feats.loc[failed, gd_cols] = np.nan
    feats[gd_cols] = feats[gd_cols].fillna(0)
    gd_flagged = (feats[GD_GROUPS].sum(axis=1) > 0).astype(int).values

    static_cols = [c for c in feats.columns
                   if c not in ID_COLS + gd_cols + ["scan_failed", "label"]]
    feature_sets = {
        "static features": static_cols,
        "static + GuardDog features": static_cols + gd_cols,
    }

    # Train on NORMAL packages only. The dataset is ~70% malware, unlike
    # reality, so fitting on everything would teach the detectors that
    # malware is normal. Hold out 40% of normal packages for evaluation.
    rng = np.random.RandomState(42)
    normal_idx = np.where(y_all == 0)[0]
    test_normal = rng.choice(normal_idx, size=int(0.4 * len(normal_idx)), replace=False)
    train_idx = np.setdiff1d(normal_idx, test_normal)
    eval_idx = np.concatenate([test_normal, np.where(y_all == 1)[0]])
    y = y_all[eval_idx]

    # GuardDog-hard: malware that GuardDog did not flag (incl. failed scans)
    hard = (y == 1) & (gd_flagged[eval_idx] == 0)
    print(f"Train: {len(train_idx)} normal | Eval: {(y == 0).sum()} normal + "
          f"{(y == 1).sum()} malicious ({hard.sum()} GuardDog-hard)")

    # ---- GuardDog rule (baseline) ----
    gd_eval = gd_flagged[eval_idx]
    gd_tpr, gd_fpr = gd_eval[y == 1].mean(), gd_eval[y == 0].mean()
    print(f"\nGuardDog flags {gd_tpr:.1%} of malware and {gd_fpr:.1%} of normal packages")

    scores = {"GuardDog rule": feats["code_issue_count"].values[eval_idx] + gd_eval}

    for set_name, cols in feature_sets.items():
        X = feats[cols].fillna(0).values.astype(float)

        # log1p on non-negative columns (tames heavy tails, e.g. a
        # 42,000-character line), then standardize on the normal train split
        nonneg = X.min(axis=0) >= 0
        X[:, nonneg] = np.log1p(X[:, nonneg])
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_eval = scaler.transform(X[train_idx]), scaler.transform(X[eval_idx])

        # ---- Isolation Forest ----
        iso = IsolationForest(random_state=42, n_estimators=100)
        iso.fit(X_train)
        scores[f"IsolationForest ({set_name})"] = -iso.decision_function(X_eval)

        # ---- Local Outlier Factor (novelty mode: learns normal, scores new) ----
        lof = LocalOutlierFactor(n_neighbors=35, novelty=True)
        lof.fit(X_train)
        scores[f"LOF ({set_name})"] = -lof.decision_function(X_eval)

        # ---- One-Class SVM (train set is small now, no subsampling needed) ----
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        ocsvm.fit(X_train)
        scores[f"OneClassSVM ({set_name})"] = -ocsvm.decision_function(X_eval)

    # ---- Random Forest (supervised REFERENCE, uses labels) ----
    # Shows how separable the data is when labels are available: an upper
    # reference, not a competitor to the self-supervised method.
    # 5-fold cross-validation, all releases of a package name in the same fold.
    # Still optimistic: campaigns under different names can span folds.
    X = feats[static_cols + gd_cols].fillna(0).values
    groups = feats["Name"].map(lambda n: re.sub(r"[-_.]+", "-", str(n)).lower())
    oof = np.zeros(len(feats))
    for tr, te in GroupKFold(n_splits=5).split(X, y_all, groups):
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=42, n_jobs=2)
        rf.fit(X[tr], y_all[tr])
        oof[te] = rf.predict_proba(X[te])[:, 1]
    scores["RandomForest (supervised reference)"] = oof[eval_idx]

    # ---- Metrics ----
    keep_hard = (y == 0) | hard
    rows = []
    for name, s in scores.items():
        row = {"method": name,
               "auc_all": roc_auc_score(y, s),
               "auc_hard": roc_auc_score(y[keep_hard], s[keep_hard])}
        if name == "GuardDog rule":  # a fixed rule has one operating point
            tpr, fpr = gd_tpr, gd_fpr
            tpr_hard = 0.0  # by definition it flags none of the hard set
            for f in FPR_TARGETS:
                row[f"tpr@fpr{int(f * 100)}%"] = np.nan
        else:
            for f in FPR_TARGETS:
                row[f"tpr@fpr{int(f * 100)}%"] = tpr_at_fpr(s, y, f)[0]
            tpr, fpr = tpr_at_fpr(s, y, gd_fpr)
            tpr_hard = tpr_at_fpr(s[keep_hard], y[keep_hard], gd_fpr)[0]
        row["tpr@gd_fpr"], row["fpr@gd_fpr"], row["tpr_hard@gd_fpr"] = tpr, fpr, tpr_hard
        for n in BASE_RATES:
            row[f"precision@1:{n}"] = precision_at(tpr, fpr, n)
        rows.append(row)
    results = pd.DataFrame(rows)

    # ---- Single-feature separability (all packages) ----
    feat_rows = []
    for col in static_cols + gd_cols:
        x = feats[col].fillna(0)
        if x.nunique() < 2:
            continue
        a = roc_auc_score(y_all, x)
        feat_rows.append({"feature": col, "auc": max(a, 1 - a),
                          "higher_in": "malicious" if a >= 0.5 else "normal",
                          "mean_malicious": x[y_all == 1].mean(),
                          "mean_normal": x[y_all == 0].mean()})
    feat_auc = pd.DataFrame(feat_rows).sort_values("auc", ascending=False)

    # ---- Report ----
    print("\n=== Baseline ROC-AUC ===")
    print(f"{'':48s}{'all malware':>12s}{'GuardDog-hard':>15s}")
    for _, r in results.iterrows():
        print(f"{r['method']:48s}{r['auc_all']:12.4f}{r['auc_hard']:15.4f}")

    print(f"\n=== Operating points (GuardDog FPR = {gd_fpr:.1%}) ===")
    header = ["tpr@fpr1%", "tpr@fpr5%", "tpr@gd_fpr", "tpr_hard@gd_fpr",
              "precision@1:200", "precision@1:1000"]
    print(f"{'':48s}" + "".join(f"{h:>17s}" for h in header))
    for _, r in results.iterrows():
        print(f"{r['method']:48s}" + "".join(
            f"{'-':>17s}" if pd.isna(r[h]) else f"{r[h]:17.3f}" for h in header))

    print("\n=== Top 15 single features (ROC-AUC, direction-free) ===")
    print(feat_auc.head(15).round(4).to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/baseline_results.csv", index=False)
    feat_auc.to_csv(f"{RESULTS_DIR}/feature_auc.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()