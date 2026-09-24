"""
baseline_experiment.py
Preliminary baseline: classical anomaly detectors applied directly
to engineered package features (no GNN yet).
Evaluates against known-malicious packages (malregistry).

Protocol (train / validation / test)
  - The split is read from DATA_DIR/splits.csv, created once by
    make_splits.py and shared by all methods (baselines, GNN).
  - Normal packages: ~50% train, ~20% validation, ~30% test.
  - Malicious packages: by package name (all releases of a name stay
    together): ~20% validation, ~80% test. Malware is never used to fit
    the unsupervised detectors.
  - Each detector's preprocessing and settings are chosen on VALIDATION,
    using partial ROC-AUC up to 5% false positives (the region that matters
    in practice). Results are reported on the untouched TEST set only.
  - Isolation Forest: averaged over several random seeds (mean +- std),
    since single runs vary by a few points.

Reports (test set)
  1. ROC-AUC on all malware and on "GuardDog-hard" malware
     (malware GuardDog did not flag, including failed scans)
  2. Operating points: detection rate (TPR) at 1% and 5% false positives,
     and at GuardDog's own false-positive rate (like-for-like comparison)
  3. Precision at realistic base rates (1 in 200, 1 in 1000 uploads),
     at GuardDog's FPR and at 1% FPR
  4. Single-feature separability ranking (descriptive, all packages)
Results are also saved as CSV in RESULTS_DIR.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/igalimi1/thesis/data"
RESULTS_DIR = "/home/igalimi1/thesis/results"

# identifiers and bookkeeping columns, not features
ID_COLS = ["Name", "Version", "path", "archive_type", "read_failed", "error", "truncated"]

# redundant: author_repo_defined is 1 exactly when a repo link exists
DROP_COLS = ["author_repo_defined"]

# GuardDog rule groups (source-code rules only)
GD_GROUPS = ["code_execution", "network_exfiltration", "obfuscation",
             "command_abuse", "sensitive_access"]

SPLIT_SEED = 42
SELECT_MAX_FPR = 0.05            # model selection: partial ROC-AUC up to this FPR
IF_SEEDS = [0, 1, 2, 3, 4]

# Candidate settings, chosen per detector on the validation set.
# "scale_all": standardize every column (rare 0/1 flags get weighted by rarity)
# "binary_raw": leave 0/1 flags as they are, standardize continuous columns only
PREPROCESSING = ["scale_all", "binary_raw"]
OCSVM_NU = [0.01, 0.05, 0.1]
OCSVM_GAMMA = [0.1, 1.0, 10.0]   # multiplied by 1 / n_features
LOF_NEIGHBORS = [10, 35, 100]

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


def preprocess(X, fit_idx, mode):
    """log1p on non-negative continuous columns, then standardize
    (statistics from the normal TRAIN rows only)."""
    X = X.copy()
    if mode == "scale_all":
        cont = np.ones(X.shape[1], dtype=bool)
    else:  # binary_raw
        cont = ~np.all(np.isin(X, [0, 1]), axis=0)
    nonneg = cont & (X.min(axis=0) >= 0)
    X[:, nonneg] = np.log1p(X[:, nonneg])
    scaler = StandardScaler().fit(X[fit_idx][:, cont])
    X[:, cont] = scaler.transform(X[:, cont])
    return X


def tpr_at_fpr(scores, y, fpr):
    """Detection rate when the threshold lets `fpr` of normal packages through."""
    threshold = np.quantile(scores[y == 0], 1 - fpr)
    return np.mean(scores[y == 1] > threshold), np.mean(scores[y == 0] > threshold)


def precision_at(tpr, fpr, n):
    """Share of alerts that are real if 1 in n uploads is malicious."""
    p = 1 / n
    denom = tpr * p + fpr * (1 - p)
    return tpr * p / denom if denom else float("nan")


def evaluate(name, s, y, hard, gd_tpr, gd_fpr):
    """All test metrics for one method's scores `s`."""
    keep_hard = (y == 0) | hard
    row = {"method": name,
           "auc_all": roc_auc_score(y, s),
           "auc_hard": roc_auc_score(y[keep_hard], s[keep_hard])}
    if name == "GuardDog rule":  # a fixed rule has one operating point
        tpr, fpr, tpr_hard = gd_tpr, gd_fpr, 0.0
        tpr1 = fpr1 = np.nan
        for f in FPR_TARGETS:
            row[f"tpr@fpr{int(f * 100)}%"] = np.nan
    else:
        for f in FPR_TARGETS:
            row[f"tpr@fpr{int(f * 100)}%"] = tpr_at_fpr(s, y, f)[0]
        tpr, fpr = tpr_at_fpr(s, y, gd_fpr)
        tpr_hard = tpr_at_fpr(s[keep_hard], y[keep_hard], gd_fpr)[0]
        tpr1, fpr1 = tpr_at_fpr(s, y, 0.01)
    row["tpr@gd_fpr"], row["fpr@gd_fpr"], row["tpr_hard@gd_fpr"] = tpr, fpr, tpr_hard
    for n in BASE_RATES:
        row[f"precision@1:{n}"] = precision_at(tpr, fpr, n)
        row[f"precision_fpr1%@1:{n}"] = precision_at(tpr1, fpr1, n)
    return row


def main():
    # ---- Load features ----
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
                   if c not in ID_COLS + DROP_COLS + gd_cols + ["scan_failed", "label"]]
    feature_sets = {
        "static features": static_cols,
        "static + GuardDog features": static_cols + gd_cols,
    }

    # ---- Split: fixed assignment from splits.csv (created by make_splits.py) ----
    # All methods (baselines, GNN) use this same file, so they are evaluated
    # on exactly the same packages.
    splits = pd.read_csv(f"{DATA_DIR}/splits.csv", dtype={"Version": str})
    split = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    if split.isna().any():
        raise SystemExit(f"{int(split.isna().sum())} packages are missing from splits.csv: "
                         "run src/make_splits.py first")
    train_idx = np.where((split == "train") & (y_all == 0))[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    y_val, y = y_all[val_idx], y_all[test_idx]
    hard = (y == 1) & (gd_flagged[test_idx] == 0)
    print(f"Train: {len(train_idx)} normal | Validation: {(y_val == 0).sum()} normal + "
          f"{(y_val == 1).sum()} malicious | Test: {(y == 0).sum()} normal + {(y == 1).sum()} "
          f"malicious ({hard.sum()} GuardDog-hard)")

    def select_score(s_val):
        return roc_auc_score(y_val, s_val, max_fpr=SELECT_MAX_FPR)

    # ---- GuardDog rule (baseline) ----
    gd_test = gd_flagged[test_idx]
    gd_tpr, gd_fpr = gd_test[y == 1].mean(), gd_test[y == 0].mean()
    print(f"\nGuardDog flags {gd_tpr:.1%} of test malware and {gd_fpr:.1%} of test normal packages")
    rows = [evaluate("GuardDog rule", feats["code_issue_count"].values[test_idx] + gd_test,
                     y, hard, gd_tpr, gd_fpr)]
    chosen = []

    for set_name, cols in feature_sets.items():
        X_raw = feats[cols].fillna(0).values.astype(float)
        X_prep = {mode: preprocess(X_raw, train_idx, mode) for mode in PREPROCESSING}
        n_feat = X_raw.shape[1]

        # ---- Isolation Forest (no scaling needed; averaged over seeds) ----
        seed_rows = []
        for seed in IF_SEEDS:
            iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=2)
            iso.fit(X_raw[train_idx])
            seed_rows.append(evaluate(f"IsolationForest ({set_name})",
                                      -iso.decision_function(X_raw[test_idx]),
                                      y, hard, gd_tpr, gd_fpr))
        seed_df = pd.DataFrame(seed_rows)
        row = seed_df.drop(columns="method").mean().to_dict()
        row["method"] = f"IsolationForest ({set_name})"
        row["auc_all_std"] = seed_df["auc_all"].std()
        rows.append(row)
        chosen.append({"method": row["method"], "setting": f"300 trees, {len(IF_SEEDS)} seeds"})

        # ---- Local Outlier Factor (novelty mode: learns normal, scores new) ----
        best = None
        for mode in PREPROCESSING:
            for k in LOF_NEIGHBORS:
                lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
                lof.fit(X_prep[mode][train_idx])
                score = select_score(-lof.decision_function(X_prep[mode][val_idx]))
                if best is None or score > best[0]:
                    best = (score, f"{mode}, n_neighbors={k}", lof, mode)
        name = f"LOF ({set_name})"
        rows.append(evaluate(name, -best[2].decision_function(X_prep[best[3]][test_idx]),
                             y, hard, gd_tpr, gd_fpr))
        chosen.append({"method": name, "setting": best[1], "val_pauc": best[0]})

        # ---- One-Class SVM ----
        best = None
        for mode in PREPROCESSING:
            for nu in OCSVM_NU:
                for g in OCSVM_GAMMA:
                    ocsvm = OneClassSVM(kernel="rbf", gamma=g / n_feat, nu=nu)
                    ocsvm.fit(X_prep[mode][train_idx])
                    score = select_score(-ocsvm.decision_function(X_prep[mode][val_idx]))
                    if best is None or score > best[0]:
                        best = (score, f"{mode}, nu={nu}, gamma={g}/n_features", ocsvm, mode)
        name = f"OneClassSVM ({set_name})"
        rows.append(evaluate(name, -best[2].decision_function(X_prep[best[3]][test_idx]),
                             y, hard, gd_tpr, gd_fpr))
        chosen.append({"method": name, "setting": best[1], "val_pauc": best[0]})

    # ---- Random Forest (supervised REFERENCE, uses labels) ----
    # Shows how separable the data is when labels are available: an upper
    # reference, not a competitor to the self-supervised method. Trained on
    # everything outside the test set (normal train + validation, malware
    # validation names), evaluated on the same test set as the detectors.
    cols = static_cols + gd_cols
    fit_idx = np.concatenate([train_idx, val_idx])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SPLIT_SEED, n_jobs=2)
    rf.fit(feats[cols].values[fit_idx], y_all[fit_idx])
    rows.append(evaluate("RandomForest (supervised reference)",
                         rf.predict_proba(feats[cols].values[test_idx])[:, 1],
                         y, hard, gd_tpr, gd_fpr))
    chosen.append({"method": "RandomForest (supervised reference)",
                   "setting": "300 trees, class_weight=balanced"})

    results = pd.DataFrame(rows)
    chosen = pd.DataFrame(chosen)

    # ---- Single-feature separability (descriptive, all packages) ----
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
    print(f"\n=== Settings chosen on validation (partial ROC-AUC, FPR <= {SELECT_MAX_FPR:.0%}) ===")
    for _, r in chosen.iterrows():
        extra = f"   (val pAUC {r['val_pauc']:.3f})" if pd.notna(r.get("val_pauc")) else ""
        print(f"{r['method']:48s}{r['setting']}{extra}")

    print("\n=== Test ROC-AUC ===")
    print(f"{'':48s}{'all malware':>12s}{'GuardDog-hard':>15s}")
    for _, r in results.iterrows():
        std = f" +- {r['auc_all_std']:.3f}" if pd.notna(r.get("auc_all_std")) else ""
        print(f"{r['method']:48s}{r['auc_all']:12.4f}{r['auc_hard']:15.4f}{std}")

    print(f"\n=== Test operating points (GuardDog FPR = {gd_fpr:.1%}) ===")
    header = ["tpr@fpr1%", "tpr@fpr5%", "tpr@gd_fpr", "tpr_hard@gd_fpr"]
    print(f"{'':48s}" + "".join(f"{h:>17s}" for h in header))
    for _, r in results.iterrows():
        print(f"{r['method']:48s}" + "".join(
            f"{'-':>17s}" if pd.isna(r[h]) else f"{r[h]:17.3f}" for h in header))

    print("\n=== Test precision at realistic base rates ===")
    print("(share of alerts that are real malware, if 1 in N uploads is malicious)")
    header2 = ["precision@1:200", "precision@1:1000", "precision_fpr1%@1:200",
               "precision_fpr1%@1:1000"]
    labels2 = ["@GD-FPR 1:200", "@GD-FPR 1:1000", "@1%FPR 1:200", "@1%FPR 1:1000"]
    print(f"{'':48s}" + "".join(f"{h:>17s}" for h in labels2))
    for _, r in results.iterrows():
        print(f"{r['method']:48s}" + "".join(
            f"{'-':>17s}" if pd.isna(r[h]) else f"{r[h]:17.3f}" for h in header2))

    print("\n=== Top 15 single features (ROC-AUC, direction-free, all packages) ===")
    print(feat_auc.head(15).round(4).to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/baseline_results.csv", index=False)
    chosen.to_csv(f"{RESULTS_DIR}/baseline_chosen_settings.csv", index=False)
    feat_auc.to_csv(f"{RESULTS_DIR}/feature_auc.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()