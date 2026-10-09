"""
baseline_protocol.py
Classical baselines under the GNN evaluation protocol (Methodology section 13
revision), with the full per-package feature set. The GNN will be evaluated on
exactly the same data, roles, contamination samples and metrics.

Protocol (from splits_campaign_v2.csv)
  - Training data: training normal packages + a seeded sample of the
    contamination pool, at each level in LEVELS (0 = clean training). The
    sample is drawn with build_graph.draw_contamination(), so the baselines
    use EXACTLY the same contamination packages as the GNN graph bundles.
  - Settings (preprocessing, LOF neighbours, OCSVM nu/gamma) chosen on the
    validation set, per level, with seed 0; reused for the other seeds.
  - Results on the test set only. Per level, test scores are averaged over
    seeds as ranks (rank-average ensemble); seed spread reported separately.

Features (per package, the same columns the GNN gets as node features)
  - static features (text-only extractor)
  - dependency counts: n_core, n_optional (dependency status is NOT used:
    `dynamic` partly reflects what the parser can read)
  - name features: min_dist_to_popular, n_typosquat_targets,
    has_typosquat_edge (is_popular is NOT used: removed malware can never be
    on a popularity list)
  - GuardDog features (second feature set), missing where the scan failed

Metrics (test set)
  - package-level and family-weighted ROC-AUC; detection rate at 1% FPR
  - GuardDog-hard ROC-AUC; detection rate at GuardDog's FPR
  - precision at 1% FPR at realistic base rates (BASE_RATES)
  - 95% confidence intervals by cluster bootstrap: normal packages resampled
    individually, malicious packages resampled BY FAMILY

Normal-population check (only if normal_release_history.csv exists; see
fetch_release_history.py). Test normal packages are split into first releases
vs later releases, and into new (project <= 30 days old at the release) vs
established projects. Per group: false-positive rate at the global 1% threshold,
ROC-AUC with only that group as the normal class, and detection at 1% FPR with
the threshold set on that group (= if that group were the deployed population).
Release history is used to define populations only, never as a feature.

Diagnostics (contamination analysis)
  - per-seed spread of family-weighted ROC-AUC and detection at 1% FPR
    (mean, std, min, max), and the per-seed values in the log
  - subset metrics: ROC-AUC and detection at 1% FPR on GuardDog-FLAGGED test
    malware vs GuardDog-HARD test malware (masking check: if contamination
    teaches a detector that "GuardDog-flagged" is normal, the flagged subset
    collapses while the hard subset does not)
  - composition of each contamination draw (GuardDog-flagged share, families)
  - chosen settings per level, printed next to the results
  - precision ceiling (perfect detector at 1% FPR) next to the precision columns

Variants (--variants; rows of the naive variant are identical to earlier runs)
  naive   the original grid: preprocessing x detector settings
  robust  naive grid + robust preprocessing (median/IQR, flags raw) + trim-and-refit:
          fit, drop the k most anomalous fraction of the TRAINING packages (no
          labels), refit; k in TRIM_FRACTIONS, k = 0 included. Chosen on validation
          like every other setting, so at 0% it can always reproduce the naive model.
  fixed   diagnostic: the naive settings chosen at 0%, reused at every level
          (separates "contamination damages model selection" from "contamination
          damages the model"); only reported for levels > 0.

Connectivity subgroups (connectivity.py; label-blind, relative to the training
normal packages): connected (any dependency edge after insertion) vs isolated,
and has_peers (shares a dependency with, or depends on / is depended on by, a
training package) vs no_peers. Every detector is also reported per subgroup:
ROC-AUC, detection at 1% FPR with the threshold set within the subgroup, and
detection of the subgroup's malware at the GLOBAL 1% threshold.

PEER detector (dependency-peer deviation, no training): mean distance from a
package's features to its k nearest peers among the training packages (packages
sharing a dependency or directly linked). Defined only for packages with peers;
preprocessing and k chosen on validation (packages with peers).
Combined rows (same feature set and level, naive):
  "OCSVM (subgroup-calibrated)"  OCSVM scores turned into tail probabilities
                                 against validation normals of the same subgroup
                                 (with / without peers): control for calibration
  "PEER -> OCSVM"                PEER for packages with peers, OCSVM otherwise,
                                 both calibrated the same way

Per-subgroup models (--variants split): one detector fitted on the training
packages WITH dependency peers and one on those WITHOUT (peers relative to the
training data itself, leave-one-out); each scores the held-out packages of its
own subgroup, settings chosen on validation within the subgroup; the two scores
are combined with the same per-subgroup calibration as above. The contamination
share inside each subgroup is printed (contamination is mostly isolated malware).

Ablation (--feature-sets): static | static+GuardDog | static+deps+names |
static+deps+names+GuardDog (default: the last two, as before).

Normal resampling (--normal-salt S): reassigns the NORMAL packages to
train/val/test (50/20/30, by salted hash of the name, all releases together);
malware roles unchanged. Measures how much results depend on which normal
packages were drawn (not covered by the bootstrap). Outputs get a suffix.

Usage
  python src/baseline_protocol.py
  python src/baseline_protocol.py --levels 0 0.05 --seeds 0 --boot 200   # quicker
  python src/baseline_protocol.py --variants naive robust fixed \
      --feature-sets static static+GuardDog static+deps+names static+deps+names+GuardDog --tag _robust
  python src/baseline_protocol.py --levels 0 0.01 --normal-salt 1        # resampling check
"""

import argparse
import hashlib
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from build_graph import draw_contamination
from connectivity import PeerIndex, connectivity_table, load_core_deps

DATA_DIR = "/home/igalimi1/thesis/data"
RESULTS_DIR = "/home/igalimi1/thesis/results"
SPLIT_FILE = "splits_campaign_v2.csv"

LEVELS = [0.0, 0.01, 0.02, 0.05]   # malware share of the training data (0 = clean training)
SEEDS = [0, 1, 2]                  # same seeds as slurm/build_graph.sh
# Realistic base rates from PyPI's 2025 review (>130,000 new projects, >3.9M new
# files, >2,000 malware reports): ~1 in 65 per new project, ~1 in 2,000 per file.
BASE_RATES = {"per_project_1:65": 65, "per_file_1:2000": 2000}

ID_COLS = ["Name", "Version", "path", "archive_type", "read_failed", "error", "truncated"]
DROP_COLS = ["author_repo_defined"]  # duplicates has_source_repo
GD_GROUPS = ["code_execution", "network_exfiltration", "obfuscation", "command_abuse", "sensitive_access"]
DEP_COLS = ["n_core", "n_optional"]
HISTORY_FILE = "normal_release_history.csv"
NEW_PROJECT_DAYS = 30
NAME_COLS = ["min_dist_to_popular", "n_typosquat_targets", "has_typosquat_edge"]

PREPROCESSING = ["scale_all", "binary_raw"]
PREPROCESSING_ROBUST = PREPROCESSING + ["robust"]
TRIM_FRACTIONS = [0.0, 0.01, 0.02, 0.05]  # share of training packages dropped before refitting
EXTRA_FPRS = {"tpr05_fw": 0.005, "tpr2_fw": 0.02}  # extra operating points (family-weighted)
PEER_K = [5, 20, 50]  # peers averaged by the PEER detector
OCSVM_NU = [0.01, 0.05, 0.1]
OCSVM_GAMMA = [0.1, 1.0, 10.0]  # times 1 / n_features
LOF_NEIGHBORS = [10, 35, 100]
SELECT_MAX_FPR = 0.05


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(cls, label):
    """Static + GuardDog + dependency counts + name features, one row per archive."""
    static = pd.read_csv(f"{DATA_DIR}/static_{cls}.csv", dtype={"Version": str})
    # names like "null" / "nan" are real package names, not missing values
    static["Name"] = pd.read_csv(f"{DATA_DIR}/static_{cls}.csv", usecols=["Name"], dtype=str,
                                 keep_default_na=False)["Name"].values
    static = static[(static["archive_type"] == "sdist") & (static["read_failed"] == 0)]

    gd = pd.read_csv(f"{DATA_DIR}/guarddog_{cls}.csv", dtype={"Version": str})
    gd = gd[["path", "scan_failed", "code_issue_count"] + GD_GROUPS].drop_duplicates("path")
    deps = pd.read_csv(f"{DATA_DIR}/deps_{cls}_packages.csv", usecols=["path"] + DEP_COLS)
    names = pd.read_csv(f"{DATA_DIR}/name_features.csv", usecols=["name"] + NAME_COLS, keep_default_na=False)

    df = static.merge(gd, on="path", how="left").merge(deps, on="path", how="left")
    df["_name"] = df["Name"].map(pep503)
    df = df.merge(names, left_on="_name", right_on="name", how="left").drop(columns=["_name", "name"])
    df = df.drop_duplicates(subset=["Name", "Version"])
    df["label"] = label
    return df


def load_features():
    """All packages with features; returns (feats, column groups)."""
    feats = pd.concat([load("malicious", 1), load("normal", 0)], ignore_index=True)
    gd_cols = ["code_issue_count"] + GD_GROUPS
    failed = feats["scan_failed"].fillna(1) == 1
    feats.loc[failed, gd_cols] = np.nan  # missing, not "clean"
    feats[gd_cols] = feats[gd_cols].fillna(0)
    feats[DEP_COLS] = feats[DEP_COLS].fillna(0)
    feats[NAME_COLS] = feats[NAME_COLS].fillna({"min_dist_to_popular": 3}).fillna(0)
    feats["gd_flagged"] = (feats[GD_GROUPS].sum(axis=1) > 0).astype(int)

    excluded = set(ID_COLS + DROP_COLS + gd_cols + DEP_COLS + NAME_COLS +
                   ["scan_failed", "label", "gd_flagged"])
    static_cols = [c for c in feats.columns if c not in excluded]
    groups = {"static": static_cols, "deps": DEP_COLS, "names": NAME_COLS, "guarddog": gd_cols}
    return feats, groups


def preprocess(X, fit_idx, mode):
    """log1p on non-negative continuous columns, then scale with statistics of fit_idx.
    scale_all / binary_raw: standardize (all columns / continuous only).
    robust: continuous columns centred on the median and scaled by IQR/1.349
    (fallback: std, then 1); 0/1 flags left raw. Median and IQR barely move
    when a few percent of the fit rows are malware, unlike mean and std."""
    X = X.copy()
    cont = np.ones(X.shape[1], dtype=bool) if mode == "scale_all" else ~np.all(np.isin(X, [0, 1]), axis=0)
    nonneg = cont & (X.min(axis=0) >= 0)
    X[:, nonneg] = np.log1p(X[:, nonneg])
    if mode == "robust":
        F = X[fit_idx][:, cont]
        med = np.median(F, axis=0)
        iqr = (np.percentile(F, 75, axis=0) - np.percentile(F, 25, axis=0)) / 1.349
        std = F.std(axis=0)
        scale = np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0))
        X[:, cont] = (X[:, cont] - med) / scale
        return X
    scaler = StandardScaler().fit(X[fit_idx][:, cont])
    X[:, cont] = scaler.transform(X[:, cont])
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def tpr_at_fpr(s, y, w, fpr):
    thr = np.quantile(s[y == 0], 1 - fpr)
    hit = s[y == 1] > thr
    return float(hit.mean()), float(np.sum(w[y == 1] * hit) / np.sum(w[y == 1]))


def precision_at(tpr, fpr, n):
    p = 1 / n
    return tpr * p / (tpr * p + fpr * (1 - p))


def point_metrics(s, y, w, hard, gd_fpr, flag=None):
    """Point estimates. flag = binary decisions for a fixed rule (GuardDog)."""
    keep_hard = (y == 0) | hard
    keep_flag = (y == 0) | ((y == 1) & ~hard)   # normal + GuardDog-flagged malware
    m = {"auc": roc_auc_score(y, s), "auc_fw": roc_auc_score(y, s, sample_weight=w),
         "auc_hard": roc_auc_score(y[keep_hard], s[keep_hard]),
         "auc_flagged": roc_auc_score(y[keep_flag], s[keep_flag])}
    if flag is not None:
        m["tpr1"] = m["tpr1_fw"] = np.nan
        m["tpr_gd"] = float(flag[y == 1].mean())
        m["tpr_gd_fw"] = float(np.sum(w[y == 1] * flag[y == 1]) / np.sum(w[y == 1]))
        tpr, fpr = m["tpr_gd"], float(flag[y == 0].mean())
        for k, n in BASE_RATES.items():
            m[f"prec@ownFPR_{k}"] = precision_at(tpr, fpr, n)
    else:
        m["tpr1"], m["tpr1_fw"] = tpr_at_fpr(s, y, w, 0.01)
        m["tpr_gd"], m["tpr_gd_fw"] = tpr_at_fpr(s, y, w, gd_fpr)
        for key, f in EXTRA_FPRS.items():
            m[key] = tpr_at_fpr(s, y, w, f)[1]
        # detection at 1% FPR within each malware subset (threshold from ALL normals)
        thr = np.quantile(s[y == 0], 0.99)
        for sub, mask in [("flagged", (y == 1) & ~hard), ("hard", hard)]:
            m[f"tpr1_{sub}_fw"] = float(np.sum(w[mask] * (s[mask] > thr)) / np.sum(w[mask])) \
                if mask.any() else np.nan
        for k, n in BASE_RATES.items():
            m[f"prec@1%FPR_{k}"] = precision_at(m["tpr1"], 0.01, n)          # package-level TPR
            m[f"prec_fw@1%FPR_{k}"] = precision_at(m["tpr1_fw"], 0.01, n)    # family-weighted TPR
    return m


def normal_groups(feats, test_idx, y):
    """Boolean masks over the test rows: subsets of the test NORMAL packages,
    from the release history. Returns ({} , None) if the history is missing."""
    path = f"{DATA_DIR}/{HISTORY_FILE}"
    if not os.path.exists(path):
        print(f"[info] {HISTORY_FILE} not found: normal-population check skipped")
        return {}, None
    h = pd.read_csv(path, dtype=str, keep_default_na=False)
    h = h[h["status"] == "ok"].drop_duplicates("path").set_index("path")
    paths = feats["path"].values[test_idx]
    known = np.array([p in h.index for p in paths]) & (y == 0)
    first_of = pd.to_numeric(h["is_first_release"]) == 1
    first = np.array([bool(first_of.get(p, False)) for p in paths])
    age = np.array([float(h.at[p, "project_age_days"]) if p in h.index else np.nan for p in paths])
    groups = {"first": known & first, "later": known & ~first,
              "new": known & (age <= NEW_PROJECT_DAYS), "estab": known & (age > NEW_PROJECT_DAYS)}
    return groups, h


def population_metrics(s, y, w, groups, flag=None):
    """Per normal group: FPR at the global 1% threshold (or the rule's own flags),
    ROC-AUC (fw) against that group only, detection (fw) at 1% FPR of that group."""
    m = {}
    if not groups:
        return m
    thr = np.quantile(s[y == 0], 0.99)
    pos = y == 1
    for g, gm in groups.items():
        if gm.sum() < 20:
            continue
        keep = gm | pos
        m[f"auc_fw_{g}"] = roc_auc_score(y[keep], s[keep], sample_weight=w[keep])
        if flag is not None:
            m[f"fpr_{g}"] = float(flag[gm].mean())
        else:
            m[f"fpr_{g}"] = float((s[gm] > thr).mean())
            thr_g = np.quantile(s[gm], 0.99)
            m[f"tpr1_fw_{g}"] = float(np.sum(w[pos] * (s[pos] > thr_g)) / np.sum(w[pos]))
            m[f"prec_fw_{g}_1:65"] = precision_at(m[f"tpr1_fw_{g}"], 0.01, 65)
    return m


def subgroup_metrics(s, y, w, sg, flag=None):
    """Per label-blind subgroup mask (both classes): ROC-AUC (fw), detection at 1%
    FPR with the threshold set inside the subgroup, and detection / FPR of the
    subgroup at the GLOBAL 1% threshold (what a single deployed threshold does)."""
    m_out = {}
    thr = np.quantile(s[y == 0], 0.99)
    for g, m in sg.items():
        neg, pos = m & (y == 0), m & (y == 1)
        if neg.sum() < 50 or pos.sum() < 20:
            continue
        m_out[f"sg_auc_fw_{g}"] = roc_auc_score(y[m], s[m], sample_weight=w[m])
        if flag is not None:
            m_out[f"sg_tpr_fw_{g}"] = float(np.sum(w[pos] * flag[pos]) / np.sum(w[pos]))
            m_out[f"sg_fpr_{g}"] = float(flag[neg].mean())
            continue
        thr_g = np.quantile(s[neg], 0.99)
        m_out[f"sg_tpr1_fw_{g}"] = float(np.sum(w[pos] * (s[pos] > thr_g)) / np.sum(w[pos]))
        m_out[f"sg_tpr1glob_fw_{g}"] = float(np.sum(w[pos] * (s[pos] > thr)) / np.sum(w[pos]))
        m_out[f"sg_fpr1glob_{g}"] = float((s[neg] > thr).mean())
    return m_out


def peer_distances(Xm, fit_idx, q_idx, P, kmax, chunk=512):
    """Sorted distances [len(q_idx), kmax] from each query package to its nearest
    peers among fit_idx (P: sparse peer matrix, > 0 = peer); inf where fewer."""
    F = Xm[fit_idx].astype(np.float64)
    f2 = (F ** 2).sum(axis=1)
    out = np.full((len(q_idx), kmax), np.inf)
    for a in range(0, len(q_idx), chunk):
        Q = Xm[q_idx[a:a + chunk]].astype(np.float64)
        D = np.sqrt(np.clip((Q ** 2).sum(axis=1)[:, None] + f2[None, :] - 2 * Q @ F.T, 0, None))
        D[~(P[a:a + chunk].toarray() > 0)] = np.inf
        kk = min(kmax, D.shape[1])
        part = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1)
        out[a:a + len(Q), :kk] = part
    return out


def peer_score(sorted_d, k):
    """Mean distance to the k nearest peers (fewer if fewer exist); NaN without peers."""
    d = sorted_d[:, :k]
    fin = np.isfinite(d)
    n = fin.sum(axis=1)
    return np.where(n > 0, np.where(fin, d, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)


def fill_missing(s):
    """Packages a detector cannot score (NaN) are never flagged: below every score."""
    s = s.copy()
    s[np.isnan(s)] = np.nanmin(s) - 1.0 if np.isfinite(s).any() else 0.0
    return s


def calibrated(ref, s, p_floor):
    """Tail probability of s against reference normal scores, as -log p, plus a tiny
    within-reference z term so packages beyond every reference score stay ordered.
    p is floored at p_floor (the same for every subgroup), otherwise the subgroup
    with more reference normals could reach smaller p-values and win every tie."""
    ref = np.sort(ref)
    n_ge = len(ref) - np.searchsorted(ref, s, side="left")
    p = np.maximum((1.0 + n_ge) / (1.0 + len(ref)), p_floor)
    med = np.median(ref)
    spread = max(np.quantile(ref, 0.99) - med, 1e-12)
    return -np.log(p) + 1e-3 * np.clip((s - med) / spread, -1e3, 1e3)


def combine(a_val, a_test, b_val, b_test, y_val):
    """Score a where it is defined (not NaN), b elsewhere; each calibrated against
    validation normals of its own subgroup (defined / not defined for a)."""
    gv, gt = ~np.isnan(a_val), ~np.isnan(a_test)
    ref_a, ref_b = a_val[gv & (y_val == 0)], b_val[~gv & (y_val == 0)]
    p_floor = 1.0 / (1.0 + min(len(ref_a), len(ref_b)))
    out = np.empty(len(a_test))
    out[gt] = calibrated(ref_a, a_test[gt], p_floor)
    out[~gt] = calibrated(ref_b, b_test[~gt], p_floor)
    return out


def bootstrap_ci(s, y, w, fam, n_boot, seed=0, flag=None):
    """95% CIs for family-weighted ROC-AUC and detection at 1% FPR (or GuardDog's
    own rate): normal packages resampled individually, malware BY FAMILY."""
    rng = np.random.RandomState(seed)
    neg = np.where(y == 0)[0]
    fams = pd.Series(np.where(y == 1)[0]).groupby(fam[y == 1]).apply(list).tolist()
    aucs, tprs = [], []
    for _ in range(n_boot):
        idx_neg = rng.choice(neg, size=len(neg), replace=True)
        picked = rng.randint(0, len(fams), size=len(fams))
        idx_pos = np.concatenate([fams[i] for i in picked])
        idx = np.concatenate([idx_neg, idx_pos])
        ys, ss, ws = y[idx], s[idx], w[idx]
        aucs.append(roc_auc_score(ys, ss, sample_weight=ws))
        if flag is None:
            tprs.append(tpr_at_fpr(ss, ys, ws, 0.01)[1])
        else:
            f = flag[idx]
            tprs.append(np.sum(ws[ys == 1] * f[ys == 1]) / np.sum(ws[ys == 1]))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))  # noqa: E731
    (a_lo, a_hi), (t_lo, t_hi) = q(aucs), q(tprs)
    t = "tpr_fw" if flag is None else "tpr_gd_fw"   # CI of detection @1% FPR, or of GuardDog's own rate
    return {"auc_fw_lo": a_lo, "auc_fw_hi": a_hi, f"{t}_lo": t_lo, f"{t}_hi": t_hi}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def make_model(kind, setting, seed, n_features):
    if kind == "IF":
        return IsolationForest(n_estimators=300, random_state=seed, n_jobs=2)
    if kind == "LOF":
        return LocalOutlierFactor(n_neighbors=setting["k"], novelty=True)
    return OneClassSVM(kernel="rbf", gamma=setting["g"] / n_features, nu=setting["nu"])


def fit_model(kind, setting, X_fit, seed, base=None):
    """Fit; with setting["trim"] = k > 0, drop the k most anomalous fraction of
    the fit rows (by the first model's own training scores) and refit. `base` =
    an already fitted untrimmed model with the same setting (saves a fit)."""
    m = base if base is not None else make_model(kind, setting, seed, X_fit.shape[1]).fit(X_fit)
    k = setting.get("trim", 0.0)
    if k > 0:
        s_fit = -m.negative_outlier_factor_ if kind == "LOF" else -m.decision_function(X_fit)
        keep = s_fit <= np.quantile(s_fit, 1 - k)
        m = make_model(kind, setting, seed, X_fit.shape[1]).fit(X_fit[keep])
    return m


def fit_score(kind, setting, X_fit, X_score, seed):
    return -fit_model(kind, setting, X_fit, seed).decision_function(X_score)


def candidates(kind, robust=False):
    """Settings grid. robust=True adds robust preprocessing and trim-and-refit."""
    preps = PREPROCESSING_ROBUST if robust else PREPROCESSING
    trims = TRIM_FRACTIONS if robust else [0.0]
    if kind == "IF":  # tree splits are scale-free: preprocessing does not matter
        base = [{"mode": "raw"}]
    elif kind == "LOF":
        base = [{"mode": p, "k": k} for p in preps for k in LOF_NEIGHBORS]
    else:
        base = [{"mode": p, "nu": nu, "g": g} for p in preps for nu in OCSVM_NU for g in OCSVM_GAMMA]
    return [{**b, "trim": t} for b in base for t in trims]


def setting_key(setting):
    """Setting without the trim fraction (candidates sharing it share the first fit)."""
    return str({k: v for k, v in setting.items() if k != "trim"})


def to_ranks(s):
    return pd.Series(s).rank(pct=True).values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Baselines under the GNN protocol")
    ap.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--variants", nargs="*", default=["naive"], choices=["naive", "robust", "fixed", "split"])
    ap.add_argument("--feature-sets", nargs="*", default=["static+deps+names", "static+deps+names+GuardDog"],
                    choices=["static", "static+GuardDog", "static+deps+names", "static+deps+names+GuardDog"])
    ap.add_argument("--normal-salt", default=None, help="reassign normal packages with this salt")
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    ap.add_argument("--detectors", nargs="*", default=["IF", "LOF", "OCSVM", "PEER"],
                    choices=["IF", "LOF", "OCSVM", "PEER"])
    ap.add_argument("--save-scores", action="store_true",
                    help="also save per-seed test scores (baseline_protocol_scores{tag}.csv) for "
                         "paired comparisons with paired_compare.py")
    args = ap.parse_args()
    args.levels = sorted(args.levels)
    if "fixed" in args.variants and (0.0 not in args.levels or "naive" not in args.variants):
        raise SystemExit("--variants fixed needs level 0 and the naive variant")
    tag = args.tag if args.tag is not None else (f"_salt{args.normal_salt}" if args.normal_salt else "")

    feats, groups = load_features()
    y_all = feats["label"].values
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    if role.isna().any():
        raise SystemExit(f"{int(role.isna().sum())} packages missing from {SPLIT_FILE}")
    if args.normal_salt is not None:
        u = feats["Name"].map(lambda n: int(hashlib.sha256(f"{args.normal_salt}:{pep503(n)}".encode())
                                            .hexdigest(), 16) / 16 ** 64)
        new = np.where(u < 0.5, "train", np.where(u < 0.7, "val", "test"))
        normal = y_all == 0
        role = role.where(~normal, pd.Series(new, index=role.index))
        print(f"[normal resampling] salt {args.normal_salt}: normal train/val/test = "
              f"{int((role[normal] == 'train').sum())}/{int((role[normal] == 'val').sum())}/"
              f"{int((role[normal] == 'test').sum())} (malware roles unchanged)")
    fam_all = feats["path"].map(dict(zip(splits["path"], splits["campaign_id"]))).fillna("").values
    w_all = feats["path"].map(dict(zip(splits["path"], splits["family_weight"].astype(float)))).values

    train_idx = np.where((role == "train") & (y_all == 0))[0]
    val_idx, test_idx = np.where(role == "val")[0], np.where(role == "test")[0]
    pool_paths = feats.loc[role == "contam", "path"].tolist()
    path_to_idx = {p: i for i, p in enumerate(feats["path"])}
    y_val, y = y_all[val_idx], y_all[test_idx]
    w, fam = w_all[test_idx], fam_all[test_idx]
    hard = (y == 1) & (feats["gd_flagged"].values[test_idx] == 0)
    print(f"train normal {len(train_idx)} | contamination pool {len(pool_paths)} | "
          f"val {int((y_val == 0).sum())}+{int((y_val == 1).sum())} | "
          f"test {int((y == 0).sum())}+{int((y == 1).sum())} ({int(hard.sum())} GuardDog-hard, "
          f"{len(set(fam[y == 1]))} families)")

    pop_groups, hist = normal_groups(feats, test_idx, y)

    deps_of = load_core_deps(DATA_DIR)
    conn = connectivity_table(feats, train_idx, test_idx, deps_of)
    sg = {"conn": conn["connected"].values, "iso": ~conn["connected"].values,
          "peers": conn["has_peers"].values, "nopeers": ~conn["has_peers"].values}
    print("\n=== Connectivity subgroups (test; relative to training normals) ===")
    rows_sg = []
    for g, m in sg.items():
        rows_sg.append({"group": g, "normal": int((m & (y == 0)).sum()), "malicious": int((m & (y == 1)).sum()),
                        "families": len(set(fam[m & (y == 1)])),
                        "share_of_malware_fw": round(float(w[m & (y == 1)].sum() / w[y == 1].sum()), 3)})
    print(pd.DataFrame(rows_sg).to_string(index=False))
    print(f"median peers, connected test packages: {conn.loc[conn['connected'], 'n_peers'].median():.0f}\n")
    if hist is not None:
        print("\n=== Normal population (release history) ===")
        rows_c = []
        for split in ["train", "val", "test"]:
            idx = np.where((role == split) & (y_all == 0))[0]
            hp = [p for p in feats["path"].values[idx] if p in hist.index]
            sub = hist.loc[hp]
            age = pd.to_numeric(sub["project_age_days"])
            rows_c.append({"split": split, "normal": len(idx), "with_history": len(hp),
                           "first_release": int((pd.to_numeric(sub["is_first_release"]) == 1).sum()),
                           f"new_<={NEW_PROJECT_DAYS}d": int((age <= NEW_PROJECT_DAYS).sum()),
                           "median_age_days": round(float(age.median()), 1)})
        print(pd.DataFrame(rows_c).to_string(index=False))
        print("test groups used: " + ", ".join(f"{g} {int(m.sum())}" for g, m in pop_groups.items()))
        print("(1% FPR within a group is decided by ~1% of its size: treat groups < 1,000 as rough)\n")

    all_sets = {
        "static": groups["static"],
        "static+GuardDog": groups["static"] + groups["guarddog"],
        "static+deps+names": groups["static"] + groups["deps"] + groups["names"],
        "static+deps+names+GuardDog": groups["static"] + groups["deps"] + groups["names"] + groups["guarddog"],
    }
    feature_sets = {k: all_sets[k] for k in args.feature_sets}

    rows, chosen = [], []
    # GuardDog rule: no training, same for every level
    gd_flag = feats["gd_flagged"].values[test_idx]
    gd_fpr = float(gd_flag[y == 0].mean())
    gd_score = feats["code_issue_count"].values[test_idx] + gd_flag
    r = {"level": "all", "method": "GuardDog rule", **point_metrics(gd_score, y, w, hard, gd_fpr, flag=gd_flag),
         **bootstrap_ci(gd_score, y, w, fam, args.boot, flag=gd_flag),
         **population_metrics(gd_score, y, w, pop_groups, flag=gd_flag),
         **subgroup_metrics(gd_score, y, w, sg, flag=gd_flag)}
    rows.append(r)
    print(f"GuardDog: flags {r['tpr_gd']:.1%} of test malware, FPR {gd_fpr:.1%} | "
          f"family-weighted {r['tpr_gd_fw']:.1%} [{r['tpr_gd_fw_lo']:.1%}, {r['tpr_gd_fw_hi']:.1%}]")
    print(f"test malware: {int(((y == 1) & ~hard).sum())} GuardDog-flagged, {int(hard.sum())} GuardDog-hard")

    # composition of the contamination draws (same for every detector at a level)
    gd_of_path = dict(zip(feats["path"], feats["gd_flagged"]))
    fam_of_path = dict(zip(feats["path"], fam_all))
    comp = []
    for level in args.levels:
        if level == 0:
            continue
        for seed in args.seeds:
            drawn = draw_contamination(pool_paths, level, len(train_idx), seed)
            comp.append({"level": level, "seed": seed, "n": len(drawn),
                         "gd_flagged_share": np.mean([gd_of_path[p] for p in drawn]),
                         "families": len({fam_of_path[p] for p in drawn})})
    if comp:
        comp = pd.DataFrame(comp)
        print("\n=== Contamination draws ===")
        print(comp.round(3).to_string(index=False))
        pool_flag = np.mean([gd_of_path[p] for p in pool_paths])
        print(f"(whole pool: {len(pool_paths)} packages, GuardDog-flagged share {pool_flag:.3f})\n")

    naive_at_0 = {}  # (feature set, detector) -> naive setting chosen at level 0 (for the fixed variant)
    labels = {"naive": "", "robust": "+robust", "fixed": " [0% settings]", "split": " per subgroup"}

    store = {}        # (level, set, kind, variant) -> per-seed (val scores, test scores)
    score_rows = []   # --save-scores: per-seed test scores (naive / robust / fixed variants)
    test_paths = feats["path"].values[test_idx]
    peer_cache = {}   # (level, seed) -> sparse peer matrices for validation and test

    def peers_for(level, seed, fit_idx):
        if (level, seed) not in peer_cache:
            idx = PeerIndex(feats["path"].values[fit_idx], feats["Name"].values[fit_idx], deps_of)
            peer_cache[(level, seed)] = {
                which: idx.query(feats["path"].values[q], feats["Name"].values[q], deps_of)
                for which, q in [("val", val_idx), ("test", test_idx)]}
        return peer_cache[(level, seed)]

    def run(level, set_name, X_raw, kind, variant):
        """Choose settings on validation (seed 0), score val + test for every seed, return the row."""
        setting = naive_at_0[(set_name, kind)] if variant == "fixed" else None
        val_pauc, seed_scores, seed_rows, seed_store = np.nan, [], [], []
        for seed in args.seeds:
            drawn = draw_contamination(pool_paths, level, len(train_idx), seed) if level > 0 else []
            contam = [path_to_idx[p] for p in drawn]
            fit_idx = np.concatenate([train_idx, np.array(contam, dtype=int)])
            preps, dists = {}, {}

            def X_for(mode):
                if mode == "raw":
                    return X_raw
                if mode not in preps:
                    preps[mode] = preprocess(X_raw, fit_idx, mode)
                return preps[mode]

            def peer_scores(cand, which):
                key = (cand["mode"], which)
                if key not in dists:
                    P = peers_for(level, seed, fit_idx)[which]
                    q = val_idx if which == "val" else test_idx
                    dists[key] = peer_distances(X_for(cand["mode"]), fit_idx, q, P, max(PEER_K))
                return peer_score(dists[key], cand["k"])

            if setting is None:  # choose on validation with the first seed
                best, bases = None, {}
                cands = ([{"mode": m, "k": k} for m in PREPROCESSING_ROBUST for k in PEER_K] if kind == "PEER"
                         else candidates(kind, robust=(variant == "robust")))
                for cand in cands:
                    if kind == "PEER":
                        sv = peer_scores(cand, "val")
                        ok = ~np.isnan(sv)
                        sc = roc_auc_score(y_val[ok], sv[ok], max_fpr=SELECT_MAX_FPR)
                    else:
                        Xm = X_for(cand["mode"])
                        key = setting_key(cand)
                        if key not in bases:
                            bases[key] = fit_model(kind, {**cand, "trim": 0.0}, Xm[fit_idx], seed)
                        mdl = fit_model(kind, cand, Xm[fit_idx], seed, base=bases[key])
                        sc = roc_auc_score(y_val, -mdl.decision_function(Xm[val_idx]), max_fpr=SELECT_MAX_FPR)
                    if best is None or sc > best[0]:
                        best = (sc, cand)
                val_pauc, setting = best
            if kind == "PEER":
                sv, s = peer_scores(setting, "val"), peer_scores(setting, "test")
            else:
                Xm = X_for(setting["mode"])
                mdl = fit_model(kind, setting, Xm[fit_idx], seed)
                sv, s = -mdl.decision_function(Xm[val_idx]), -mdl.decision_function(Xm[test_idx])
            seed_store.append((sv, s))
            s_f = fill_missing(s)
            if args.save_scores:
                score_rows.append(pd.DataFrame({"level": level, "features": set_name, "detector": kind,
                                                "variant": variant, "seed": seed, "path": test_paths,
                                                "score": s_f}))
            seed_scores.append(to_ranks(s_f))
            seed_rows.append(point_metrics(s_f, y, w, hard, gd_fpr))

        store[(level, set_name, kind, variant)] = seed_store
        chosen.append({"level": level, "features": set_name, "method": kind, "variant": variant,
                       "setting": str(setting), "val_pauc": val_pauc})
        if variant == "naive" and level == 0:
            naive_at_0[(set_name, kind)] = setting
        name = f"{kind}{labels[variant]} ({set_name})"
        if kind == "PEER":
            name = f"PEER ({set_name}; packages with peers only)"
        return summarise(level, name, variant, set_name, kind, seed_scores, seed_rows, setting)

    def run_split(level, set_name, X_raw, kind):
        """One detector per subgroup (with / without dependency peers), calibrated and combined."""
        settings = {}
        seed_scores, seed_rows = [], []
        for seed in args.seeds:
            drawn = draw_contamination(pool_paths, level, len(train_idx), seed) if level > 0 else []
            contam = [path_to_idx[p] for p in drawn]
            fit_idx = np.concatenate([train_idx, np.array(contam, dtype=int)])
            is_contam = np.isin(fit_idx, contam)
            fit_has = connectivity_table(feats, fit_idx, fit_idx, deps_of)["has_peers"].values
            val_has = connectivity_table(feats, fit_idx, val_idx, deps_of)["has_peers"].values
            test_has = connectivity_table(feats, fit_idx, test_idx, deps_of)["has_peers"].values
            sc_v, sc_t = {}, {}
            for g in [True, False]:
                sub = fit_idx[fit_has == g]
                vsel = val_has == g
                if seed == args.seeds[0]:
                    share = is_contam[fit_has == g].mean()
                    print(f"    [{'peers' if g else 'no peers'}] {len(sub)} training packages, "
                          f"contamination share {share:.1%}", flush=True)
                preps = {}

                def X_for(mode):
                    if mode == "raw":
                        return X_raw
                    if mode not in preps:
                        preps[mode] = preprocess(X_raw, sub, mode)
                    return preps[mode]

                if g not in settings:  # choose on validation within the subgroup, first seed
                    best = None
                    for cand in candidates(kind):
                        Xm = X_for(cand["mode"])
                        mdl = fit_model(kind, cand, Xm[sub], seed)
                        sc = roc_auc_score(y_val[vsel], -mdl.decision_function(Xm[val_idx][vsel]),
                                           max_fpr=SELECT_MAX_FPR)
                        if best is None or sc > best[0]:
                            best = (sc, cand)
                    settings[g] = best[1]
                    chosen.append({"level": level, "features": set_name, "method": kind, "variant": "split",
                                   "setting": f"{'peers' if g else 'no peers'}: {best[1]}", "val_pauc": best[0]})
                Xm = X_for(settings[g]["mode"])
                mdl = fit_model(kind, settings[g], Xm[sub], seed)
                sc_v[g], sc_t[g] = -mdl.decision_function(Xm[val_idx]), -mdl.decision_function(Xm[test_idx])
            comb = combine(np.where(val_has, sc_v[True], np.nan), np.where(test_has, sc_t[True], np.nan),
                           sc_v[False], sc_t[False], y_val)
            seed_scores.append(to_ranks(comb))
            seed_rows.append(point_metrics(comb, y, w, hard, gd_fpr))
        return summarise(level, f"{kind} per subgroup ({set_name})", "split", set_name, kind,
                         seed_scores, seed_rows, str(settings))

    def summarise(level, name, variant, set_name, kind, seed_scores, seed_rows, setting):
        s_ens = np.mean(seed_scores, axis=0)
        row = {"level": level, "method": name, "variant": variant, "features": set_name, "detector": kind,
               **point_metrics(s_ens, y, w, hard, gd_fpr), **bootstrap_ci(s_ens, y, w, fam, args.boot),
               **population_metrics(s_ens, y, w, pop_groups), **subgroup_metrics(s_ens, y, w, sg)}
        per_seed = pd.DataFrame(seed_rows)
        for col in ["auc_fw", "tpr1_fw", "tpr1_flagged_fw", "tpr1_hard_fw", *EXTRA_FPRS]:
            v = per_seed[col]
            row[f"{col}_seed_mean"] = v.mean()
            row[f"{col}_seed_std"] = v.std() if len(v) > 1 else 0.0
            row[f"{col}_seed_min"], row[f"{col}_seed_max"] = v.min(), v.max()
        row["setting"] = str(setting)
        print(f"level {level:.0%} | {name:52s} ROC-AUC (fw) {row['auc_fw']:.3f} "
              f"[{row['auc_fw_lo']:.3f}, {row['auc_fw_hi']:.3f}] | caught@1%FPR (fw) "
              f"{row['tpr1_fw']:.1%} [{row['tpr_fw_lo']:.1%}, {row['tpr_fw_hi']:.1%}] | "
              f"connected: AUC {row.get('sg_auc_fw_conn', np.nan):.3f}, "
              f"caught {row.get('sg_tpr1_fw_conn', np.nan):.1%}", flush=True)
        print(f"    per seed: caught@1%FPR (fw) "
              f"{', '.join(f'{v:.1%}' for v in per_seed['tpr1_fw'])} | ROC-AUC (fw) "
              f"{', '.join(f'{v:.3f}' for v in per_seed['auc_fw'])} | setting {setting}", flush=True)
        return row

    for level in args.levels:
        for set_name, cols in feature_sets.items():
            X_raw = feats[cols].fillna(0).values.astype(float)
            for kind in args.detectors:
                for variant in ["naive", "robust", "fixed", "split"]:
                    if variant not in args.variants or (variant == "fixed" and level == 0):
                        continue
                    if kind == "PEER" and variant != "naive":
                        continue
                    if variant == "split":
                        rows.append(run_split(level, set_name, X_raw, kind))
                        continue
                    rows.append(run(level, set_name, X_raw, kind, variant))
            # combined rows: calibration control, and PEER for packages with peers + OCSVM otherwise
            a, b = store.get((level, set_name, "PEER", "naive")), store.get((level, set_name, "OCSVM", "naive"))
            if a is not None and b is not None:
                for name, first in [("OCSVM (subgroup-calibrated)", "ocsvm"), ("PEER -> OCSVM", "peer")]:
                    seed_scores, seed_rows = [], []
                    for (pv, pt), (ov, ot) in zip(a, b):
                        av = np.where(np.isnan(pv), np.nan, ov) if first == "ocsvm" else pv
                        at = np.where(np.isnan(pt), np.nan, ot) if first == "ocsvm" else pt
                        sc = combine(av, at, ov, ot, y_val)
                        seed_scores.append(to_ranks(sc))
                        seed_rows.append(point_metrics(sc, y, w, hard, gd_fpr))
                    rows.append(summarise(level, f"{name} ({set_name})", "combined", set_name, name,
                                          seed_scores, seed_rows, "calibrated on validation normals per subgroup"))

    # Supervised reference: trained on training normals + the whole contamination pool (labelled)
    cols = all_sets["static+deps+names+GuardDog"]
    fit_idx = np.concatenate([train_idx, np.array([path_to_idx[p] for p in pool_paths], dtype=int)])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=2)
    rf.fit(feats[cols].fillna(0).values[fit_idx], y_all[fit_idx])
    s = rf.predict_proba(feats[cols].fillna(0).values[test_idx])[:, 1]
    rows.append({"level": "ref", "method": "RandomForest (supervised reference)",
                 **point_metrics(s, y, w, hard, gd_fpr), **bootstrap_ci(s, y, w, fam, args.boot),
                 **population_metrics(s, y, w, pop_groups), **subgroup_metrics(s, y, w, sg)})

    results, chosen = pd.DataFrame(rows), pd.DataFrame(chosen)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/baseline_protocol_results{tag}.csv", index=False)
    chosen.to_csv(f"{RESULTS_DIR}/baseline_protocol_settings{tag}.csv", index=False)
    if len(comp):
        comp.to_csv(f"{RESULTS_DIR}/baseline_protocol_contamination{tag}.csv", index=False)
    if args.save_scores and score_rows:
        pd.concat(score_rows).to_csv(f"{RESULTS_DIR}/baseline_protocol_scores{tag}.csv", index=False)

    pd.set_option("display.width", 220)
    show = ["level", "method", "auc", "auc_fw", "auc_fw_lo", "auc_fw_hi", "tpr1", "tpr1_fw",
            "tpr_fw_lo", "tpr_fw_hi", "tpr_gd", "tpr_gd_fw", "auc_hard"]
    print("\n=== Test results (fw = family-weighted; lo/hi = 95% bootstrap CI) ===")
    print(results[show].round(3).to_string(index=False))

    print("\n=== Masking check: GuardDog-flagged vs GuardDog-hard test malware ===")
    sub = ["level", "method", "auc_flagged", "auc_hard", "tpr1_fw", "tpr1_flagged_fw", "tpr1_hard_fw"]
    print(results[sub].round(3).to_string(index=False))

    print("\n=== Seed spread (family-weighted; one contamination draw per seed) ===")
    spread = ["level", "method"] + [f"{c}_seed_{k}" for c in ["auc_fw", "tpr1_fw"]
                                    for k in ["mean", "std", "min", "max"]]
    print(results.loc[results["level"].isin(args.levels), spread].round(3).to_string(index=False))

    print("\n=== Settings chosen on validation ===")
    print(chosen.round(3).to_string(index=False))

    prec = [c for c in results.columns if c.startswith("prec")]
    print("\n=== Precision at realistic base rates ===")
    print("ceiling (perfect detector at 1% FPR): " +
          ", ".join(f"{k} {precision_at(1.0, 0.01, n):.3f}" for k, n in BASE_RATES.items()))
    print("prec@ = package-level TPR, prec_fw@ = family-weighted TPR; GuardDog at its own FPR")
    print(results[["level", "method"] + prec].round(3).to_string(index=False))
    if pop_groups:
        print("\n=== Normal-population check (test normals split by release history) ===")
        print("fpr_* = false-positive rate at the global 1% threshold (GuardDog: its own flags)")
        print("auc_fw_* / tpr1_fw_* / prec_fw_* = that group alone as the normal population")
        cols_p = ["level", "method"] + [c for g in pop_groups for c in
                                        (f"fpr_{g}", f"auc_fw_{g}", f"tpr1_fw_{g}") if c in results.columns]
        print(results[cols_p].round(3).to_string(index=False))
        prec_p = [c for c in results.columns if c.startswith("prec_fw_") and c.endswith("_1:65")]
        print(results[["level", "method", "prec_fw@1%FPR_per_project_1:65"] + prec_p].round(3).to_string(index=False))
    print("\n=== Operating points (family-weighted, rank-average; seed mean for 1%) ===")
    op = ["level", "method", "tpr05_fw", "tpr1_fw", "tpr1_fw_seed_mean", "tpr2_fw"]
    print(results.loc[results["level"].isin(args.levels), op].round(3).to_string(index=False))

    if len(args.variants) > 1:
        det = results[results["level"].isin(args.levels)]
        print("\n=== Variants side by side (ROC-AUC fw | caught @ 1% FPR fw, seed mean) ===")
        for metric in ["auc_fw", "tpr1_fw_seed_mean"]:
            piv = det.pivot_table(index=["features", "detector", "level"], columns="variant",
                                  values=metric, aggfunc="first")
            print(f"\n{metric}")
            print(piv.round(3).to_string())

    sg_cols = [c for g in ["conn", "iso", "peers"] for c in
               (f"sg_auc_fw_{g}", f"sg_tpr1_fw_{g}", f"sg_tpr1glob_fw_{g}") if c in results.columns]
    if sg_cols:
        print("\n=== Connectivity subgroups: ROC-AUC fw | caught @ 1% FPR within subgroup | "
              "caught at the global 1% threshold ===")
        print(results[["level", "method"] + sg_cols].round(3).to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/baseline_protocol_*{tag}.csv")


if __name__ == "__main__":
    main()