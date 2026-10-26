"""
operating_points.py  (revision, October 2026)
Detection across operating points from the saved per-package scores. Answers the supervisors' point
on the contamination results (neighbouring false-positive rates, subgroup calibration of the proposed
model).
No training: it only re-reads score files.

For every method x contamination level x training seed:
  auc            family-weighted ROC-AUC (alpha column 0)
  caught         share of malware caught (family-weighted) at FPR alpha, threshold = (1 - alpha)
                 quantile of the NORMAL TEST scores (the convention of the thesis, eq. 9);
                 alpha on a grid from 0.2% to 5% (detection-vs-FPR curve)
  caught_ens     caught of the rank-averaged seed ensemble (as Table 6)
Reported as mean and sd over the 3 seeds (+ per-seed values), for plain and subgroup-calibrated scores.

Methods (skipped with a message when their score file is missing)
  raw detectors   baseline_protocol_scores_ops.csv (slurm/ops_scores.sh): One-Class SVM, its per-group
                  calibrated variant, Isolation Forest (all features, GuardDog included)
  contrastive     contrastive_scores{tag}.csv, configuration chosen on validation per level
                  (contrastive_settings{tag}.csv, as paired_compare.py): Triplet + GIN tuned base and N3
                  (2% / 5%: tags _c25);
                  calibrated with the same per-subgroup calibration as the raw detectors
                  (gnn_eval.Evaluator.calibrated; peers among the training releases of the seed's bundle)

Check: the plain and calibrated caught at 1% FPR must reproduce contrastive_results{tag}.csv
(tpr1_fw_seed_mean); the script prints both.

Outputs (RESULTS_DIR): operating_points.csv (long format, one row per method x level x metric x alpha)
Usage:  python src/operating_points.py            (Snellius, after ops_scores.sh)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import RESULTS_DIR, combine, to_ranks  # noqa: E402

GD = "static+deps+names+GuardDog"
LEVELS = [0.0, 0.01, 0.02, 0.05]
TABLE_ALPHAS = [0.005, 0.01, 0.02, 0.05]
CURVE_ALPHAS = sorted({float(a) for a in np.round(np.geomspace(0.002, 0.05, 15), 5)} | set(TABLE_ALPHAS))


def _c25(base):
    return {0.0: base, 0.01: base, 0.02: f"{base}_c25", 0.05: f"{base}_c25"}


# (method, GuardDog features?, tag per level)
CONTRASTIVE = [
    ("Triplet + GIN, base", True, _c25("_triplet_gin_tuned")),
    ("Triplet + GIN, N3", True, _c25("_triplet_gin_n3")),
]
# (method, calibrated?, detector, variant) in baseline_protocol_scores_ops.csv (GuardDog features)
BASELINES = [
    ("One-Class SVM", False, "OCSVM", "naive"),
    ("One-Class SVM", True, "OCSVM (subgroup-calibrated)", "combined"),
    ("Isolation Forest", False, "IF", "naive"),
]


# ---------------------------------------------------------------------------
# Metrics (pure functions)
# ---------------------------------------------------------------------------
def _wshare(hit, w):
    return float(np.sum(w * hit) / np.sum(w)) if len(w) else np.nan


def seed_metrics(st, y, w, curve_alphas=CURVE_ALPHAS):
    """Metrics of one seed. st: test scores (higher = more anomalous), in the order of y.
    w: family weights of the test releases.
    Returns {(metric, alpha): value}."""
    out = {("auc", 0.0): float(roc_auc_score(y, st, sample_weight=w))}   # family-weighted ROC-AUC
    neg, pos = st[y == 0], y == 1
    s_pos, w_pos = st[pos], w[pos]
    for a in curve_alphas:
        out[("caught", a)] = _wshare(s_pos > np.quantile(neg, 1 - a), w_pos)
    return out


def summarise(per_seed, ens):
    """per_seed: list of seed_metrics dicts; ens: {(metric, alpha): value} of the seed ensemble.
    Returns rows (metric, alpha, mean, sd, n_seeds, per_seed)."""
    rows = []
    for key in per_seed[0]:
        v = np.array([m[key] for m in per_seed], dtype=float)
        rows.append({"metric": key[0], "alpha": key[1], "mean": float(np.nanmean(v)),
                     "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n_seeds": len(v),
                     "per_seed": " ".join(f"{x:.4f}" for x in v)})
    for key, val in ens.items():
        rows.append({"metric": key[0], "alpha": key[1], "mean": val, "sd": np.nan, "n_seeds": len(per_seed),
                     "per_seed": ""})
    return rows


def ensemble_metrics(tests, y, w, alphas=TABLE_ALPHAS):
    """Caught of the rank-averaged seed ensemble (threshold from the normal test releases)."""
    s = np.mean([to_ranks(t) for t in tests], axis=0)
    neg, pos = s[y == 0], y == 1
    return {("caught_ens", a): _wshare(s[pos] > np.quantile(neg, 1 - a), w[pos]) for a in alphas}


def calibrate(sv, st, has_v, has_t, y_val):
    """Per-subgroup calibration (with / without peers), test and validation releases, exactly as
    gnn_eval.Evaluator.calibrated and baseline_protocol.combine."""
    a_v = np.where(has_v, sv, np.nan)
    return combine(a_v, a_v, sv, sv, y_val), combine(a_v, np.where(has_t, st, np.nan), sv, st, y_val)


# ---------------------------------------------------------------------------
# Loading scores (Snellius)
# ---------------------------------------------------------------------------
def _ordered(g, split, paths, label):
    s = g[g["split"] == split] if split is not None else g
    s = pd.Series(s["score"].astype(float).values, index=s["path"].values)
    missing = set(paths) - set(s.index)
    if missing:
        raise ValueError(f"{label}: {len(missing)} {split} releases without a score")
    return s.loc[paths].values


_cache = {}


def _read(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(f"{RESULTS_DIR}/{name}", keep_default_na=False, dtype={"path": str})
    return _cache[name]


def contrastive_seeds(tag, level, val_paths, test_paths):
    """{seed: (val scores, test scores)} of the configuration chosen on validation, or None."""
    from paired_compare import contrastive_config
    if not os.path.exists(f"{RESULTS_DIR}/contrastive_scores{tag}.csv"):
        return None, None
    depth, scorer = contrastive_config({"tag": tag, "level": level}, RESULTS_DIR)
    sc = _read(f"contrastive_scores{tag}.csv")
    sc = sc[np.isclose(sc["level"].astype(float), level) & (sc["depth"].astype(int) == depth)
            & (sc["scorer"] == scorer)]
    out = {int(seed): (_ordered(g, "val", val_paths, tag), _ordered(g, "test", test_paths, tag))
           for seed, g in sc.groupby("seed")}
    return (out or None), f"{'no graph' if depth == 0 else f'depth {depth}'}, {scorer}"


def baseline_seeds(detector, variant, features, level, val_paths, test_paths):
    name = "baseline_protocol_scores_ops.csv"
    if not os.path.exists(f"{RESULTS_DIR}/{name}"):
        return None
    sc = _read(name)
    split = "test" if "split" in sc.columns else None   # older score files hold test scores only
    sc = sc[np.isclose(sc["level"].astype(float), level) & (sc["detector"] == detector)
            & (sc["variant"] == variant) & (sc["features"] == features)]
    out = {int(seed): (None, _ordered(g, split, test_paths, detector)) for seed, g in sc.groupby("seed")}
    return out or None


class PeerFlags:
    """Which validation / test releases have dependency peers among the training releases of a
    bundle (level, seed): the subgroups of the per-subgroup calibration."""

    def __init__(self, ev):
        self.ev, self.cache = ev, {}

    def __call__(self, level, seed):
        from build_graph import load_bundle
        from connectivity import PeerIndex
        from train_cola import bundle_path
        key = bundle_path(level, seed)
        if key not in self.cache:
            b = load_bundle(key)
            fit_paths = b["nodes"].loc[b["nodes"]["kind"] == "evaluated", "path"].values
            ev, f = self.ev, self.ev.feats
            fit_mask = f["path"].isin(set(fit_paths)).values
            idx = PeerIndex(f["path"].values[fit_mask], f["Name"].values[fit_mask], ev.deps_of)

            def has(paths, names):
                return np.asarray((idx.query(paths, names, ev.deps_of) > 0).sum(1)).ravel() > 0

            self.cache[key] = (has(ev.val_paths, f["Name"].values[ev.val_idx]),
                               has(ev.test_paths, f["Name"].values[ev.test_idx]))
        return self.cache[key]


def check_against_results(tag, level, config, plain, cal):
    """Print our 1% FPR seed mean next to the one stored by train_contrastive.py."""
    f = f"{RESULTS_DIR}/contrastive_results{tag}.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    dep, scorer = config.split(", ")
    d = d[np.isclose(pd.to_numeric(d["level"], errors="coerce"), level)
          & d["method"].str.contains(f" {scorer}, ", regex=False)
          & d["method"].str.contains("no graph" if dep == "no graph" else dep, regex=False)]
    for lab, mine, calib in [("plain", plain, False), ("calibrated", cal, True)]:
        r = d[d["method"].str.contains("calibrated") == calib]
        if len(r) and mine is not None:
            print(f"    check {lab:10s} caught@1%: here {mine:.1%} | contrastive_results {r['tpr1_fw_seed_mean'].iloc[0]:.1%}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Detection across operating points (revision)")
    ap.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    ap.add_argument("--no-calibration", action="store_true", help="skip calibrated contrastive rows (no bundles)")
    args = ap.parse_args()
    from gnn_eval import Evaluator
    ev = Evaluator(boot=0)
    y_val, y, w = ev.y_val, ev.y, ev.w
    flags = None if args.no_calibration else PeerFlags(ev)
    rows = []

    def add(level, method, calibrated, guarddog, source, config, seeds):
        per = [seed_metrics(st, y, w) for _, st in seeds.values()]
        ens = ensemble_metrics([st for _, st in seeds.values()], y, w)
        for r in summarise(per, ens):
            rows.append({"level": level, "method": method, "calibrated": calibrated, "guarddog": guarddog,
                         "source": source, "config": config, **r})
        m1 = float(np.mean([p[("caught", 0.01)] for p in per]))
        print(f"  {level:.0%} | {method}{', calibrated' if calibrated else ''}"
              f" [{config}]: caught@1% {m1:.1%}",
              flush=True)
        return m1

    for level in args.levels:
        print(f"\n=== contamination {level:.0%} ===", flush=True)
        for method, calib, det, var in BASELINES:
            seeds = baseline_seeds(det, var, GD, level, ev.val_paths, ev.test_paths)
            if seeds is None:
                print(f"  [skip] {method}: no scores in baseline_protocol_scores_ops.csv")
                continue
            add(level, method, calib, True, f"{det} / {var}", GD, seeds)
        for method, gd_on, tags in CONTRASTIVE:
            tag = tags.get(level)
            if tag is None:
                continue
            seeds, config = contrastive_seeds(tag, level, ev.val_paths, ev.test_paths)
            if seeds is None:
                print(f"  [skip] {method} ({tag}): no contrastive_scores{tag}.csv")
                continue
            plain = add(level, method, False, gd_on, tag, config, seeds)
            cal = None
            if flags is not None:
                cal_seeds = {}
                for seed, (sv, st) in seeds.items():
                    has_v, has_t = flags(level, seed)
                    cal_seeds[seed] = calibrate(sv, st, has_v, has_t, y_val)
                cal = add(level, method, True, gd_on, tag, config, cal_seeds)
            check_against_results(tag, level, config, plain, cal)

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(f"{RESULTS_DIR}/operating_points.csv", index=False)
    print(f"\nsaved {RESULTS_DIR}/operating_points.csv ({len(out)} rows)")

    pd.set_option("display.width", 220)
    t = out[(out["metric"] == "caught") & out["alpha"].isin(TABLE_ALPHAS)].copy()
    t["name"] = t["method"] + np.where(t["calibrated"], ", calibrated", "")
    piv = t.pivot_table(index=["level", "name"], columns="alpha", values="mean", sort=False)
    print("\n=== Caught (seed mean) at each FPR ===")
    print((100 * piv).round(1).to_string())


if __name__ == "__main__":
    main()
