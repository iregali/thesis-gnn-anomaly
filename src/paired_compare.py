"""
paired_compare.py
Pre-registered comparisons between two methods (Methodology section 38).

For each comparison A vs B on the same test packages:
  - difference of the SEED-MEAN metric (the headline statistic), A minus B
  - 95% CI of that difference by PAIRED cluster bootstrap: in every resample,
    normal test packages are drawn individually and malicious ones BY FAMILY
    (as bootstrap_ci in baseline_protocol.py), and both methods are evaluated
    on the SAME resample, so shared test-set noise cancels out
  - training-seed variability: in every resample the training seeds of each
    method are also drawn with replacement (two-level bootstrap), so the CI
    covers both the choice of test packages and the choice of training seeds.
    With 3 seeds the seed part is coarse; per-seed values are saved as well.
    (--no-seed-resample: package variability only, given the seeds)
  - the 1% FPR threshold is recomputed from the resampled normal packages in
    every resample
  - two-sided bootstrap p-value of "difference = 0"; the smallest possible
    value is about 2 / B, so B = 5,000 (default) leaves room below Holm's
    strictest threshold (0.05 / 8 = 0.006)
  - Holm correction within each family of comparisons (the "primary" family is
    the one that answers RQ3; other families are corrected separately)

The comparisons are listed in a spec file written BEFORE the runs
(config/primary_comparisons.json), so they cannot be chosen after seeing the
results. Every test package of the split must be scored by both methods.

Method specs in the json
  {"type": "contrastive", "tag": "_infonce_gin_tuned", "level": 0}
      scores from contrastive_scores{tag}.csv; depth and scorer = the
      configuration chosen on validation (contrastive_settings{tag}.csv,
      column `chosen`), unless "depth" / "scorer" are given explicitly.
      Optional "calibrated": not supported (plain scores only).
  {"type": "baseline", "tag": "_primary", "level": 0, "detector": "OCSVM",
   "variant": "naive", "features": "static+deps+names+GuardDog"}
      scores from baseline_protocol_scores{tag}.csv (baseline_protocol.py --save-scores)
  {"best_of": [spec, spec, ...]}
      the candidate with the highest mean validation pAUC (contrastive only),
      i.e. chosen on validation, never on test

Metrics: auc_fw (family-weighted ROC-AUC), tpr1_fw (family-weighted share of
malware caught at 1% FPR, threshold = 99th percentile of the resampled normals).

Usage
  python src/paired_compare.py                                  # config/primary_comparisons.json
  python src/paired_compare.py --spec config/x.json --boot 5000
Outputs: RESULTS_DIR/paired_comparisons{tag}.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

METRICS = ("auc_fw", "tpr1_fw")


# ---------------------------------------------------------------------------
# Statistics (pure functions, tested on synthetic data)
# ---------------------------------------------------------------------------
def tpr1_fw(s, y, w):
    thr = np.quantile(s[y == 0], 0.99)
    pos = y == 1
    return float(np.sum(w[pos] * (s[pos] > thr)) / np.sum(w[pos]))


def metric(name, s, y, w):
    if name == "auc_fw":
        return roc_auc_score(y, s, sample_weight=w)
    if name == "tpr1_fw":
        return tpr1_fw(s, y, w)
    raise ValueError(name)


def seed_mean(name, S, y, w):
    """Headline statistic: metric per seed, averaged. S: [n_seeds, n_packages]."""
    return float(np.mean([metric(name, s, y, w) for s in S]))


def resample_indices(y, fam, n_boot, seed=0):
    """Cluster bootstrap index sets, generated once and shared by all comparisons:
    normal packages resampled individually, malicious packages by family."""
    rng = np.random.RandomState(seed)
    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]
    groups = pd.Series(pos).groupby(fam[pos]).apply(list).tolist()
    out = []
    for _ in range(n_boot):
        idx_neg = rng.choice(neg, size=len(neg), replace=True)
        picked = rng.randint(0, len(groups), size=len(groups))
        out.append(np.concatenate([idx_neg, np.concatenate([groups[i] for i in picked])]))
    return out


def paired_bootstrap(A, B, y, w, boots, metrics=METRICS, seed_resample=True, seed=0):
    """Difference (A - B) of the seed-mean metric, with a paired bootstrap.
    A, B: [n_seeds, n_packages] test scores in the same package order (seed counts
    may differ). seed_resample: also draw each method's training seeds with
    replacement in every resample (independently for A and B).
    Returns {metric: dict(a, b, diff, lo, hi, p, boot_sd, a_seeds, b_seeds)}."""
    out = {}
    for m in metrics:
        a, b = seed_mean(m, A, y, w), seed_mean(m, B, y, w)
        rs = np.random.RandomState(seed)            # same seed draws for every metric
        d = np.empty(len(boots))
        for k, idx in enumerate(boots):
            ys, ws = y[idx], w[idx]
            sa = rs.randint(0, len(A), len(A)) if seed_resample else np.arange(len(A))
            sb = rs.randint(0, len(B), len(B)) if seed_resample else np.arange(len(B))
            d[k] = seed_mean(m, A[sa][:, idx], ys, ws) - seed_mean(m, B[sb][:, idx], ys, ws)
        n = len(d)
        # two-sided p of H0 "difference = 0": twice the smaller tail, +1 correction
        p = min(1.0, 2.0 * min(np.sum(d <= 0) + 1, np.sum(d >= 0) + 1) / (n + 1))
        out[m] = {"a": a, "b": b, "diff": a - b, "lo": float(np.percentile(d, 2.5)),
                  "hi": float(np.percentile(d, 97.5)), "p": p, "boot_sd": float(d.std(ddof=1)),
                  "a_seeds": [metric(m, s, y, w) for s in A], "b_seeds": [metric(m, s, y, w) for s in B]}
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Loading scores
# ---------------------------------------------------------------------------
def _close(series, value):
    return np.isclose(series.astype(float), float(value))


def contrastive_config(spec, results_dir):
    """(depth, scorer) used for a contrastive spec: explicit, or chosen on validation."""
    if "depth" in spec and "scorer" in spec:
        return int(spec["depth"]), spec["scorer"]
    st = pd.read_csv(f"{results_dir}/contrastive_settings{spec['tag']}.csv")
    ch = st[st["chosen"].astype(bool) & _close(st["level"], spec["level"])][["depth", "scorer"]].drop_duplicates()
    if len(ch) != 1:
        raise ValueError(f"{spec['tag']} level {spec['level']}: expected one chosen configuration, found {len(ch)}")
    return int(ch.iloc[0]["depth"]), ch.iloc[0]["scorer"]


def val_pauc(spec, results_dir):
    """Mean validation pAUC over seeds of the configuration a spec resolves to."""
    if spec.get("type") != "contrastive":
        raise ValueError("best_of supports contrastive specs only")
    depth, scorer = contrastive_config(spec, results_dir)
    v = pd.read_csv(f"{results_dir}/contrastive_valseeds{spec['tag']}.csv")
    v = v[_close(v["level"], spec["level"]) & (v["depth"] == depth) & (v["scorer"] == scorer)]
    if v.empty:
        raise ValueError(f"no validation pAUC for {spec}")
    return float(v["val_pauc"].mean())


def resolve(spec, results_dir):
    """Turn a best_of spec into the candidate chosen on validation."""
    if "best_of" not in spec:
        return spec
    scored = [(val_pauc(c, results_dir), c) for c in spec["best_of"]]
    best = max(scored, key=lambda t: t[0])
    print(f"  best_of: " + ", ".join(f"{c['tag']} {p:.3f}" for p, c in scored) + f" -> {best[1]['tag']}")
    return best[1]


def load_scores(spec, results_dir, test_paths):
    """[n_seeds, n_test] matrix in test_paths order, plus a readable label."""
    spec = resolve(spec, results_dir)
    if spec["type"] == "contrastive":
        depth, scorer = contrastive_config(spec, results_dir)
        sc = pd.read_csv(f"{results_dir}/contrastive_scores{spec['tag']}.csv", keep_default_na=False,
                         dtype={"path": str})
        sc = sc[(sc["split"] == "test") & _close(sc["level"], spec["level"]) & (sc["depth"] == depth)
                & (sc["scorer"] == scorer)]
        label = f"{spec['tag'].strip('_')} depth {depth} {scorer}"
    elif spec["type"] == "baseline":
        sc = pd.read_csv(f"{results_dir}/baseline_protocol_scores{spec['tag']}.csv", keep_default_na=False,
                         dtype={"path": str})
        sc = sc[_close(sc["level"], spec["level"]) & (sc["detector"] == spec["detector"])
                & (sc["variant"] == spec["variant"]) & (sc["features"] == spec["features"])]
        label = f"{spec['detector']} {spec['variant']} ({spec['features']})"
    else:
        raise ValueError(f"unknown spec type: {spec}")
    if sc.empty:
        raise ValueError(f"no scores for {spec}")
    rows = []
    for seed, g in sc.groupby("seed"):
        s = pd.Series(g["score"].astype(float).values, index=g["path"].values)
        missing = set(test_paths) - set(s.index)
        if missing:
            raise ValueError(f"{label}, seed {seed}: {len(missing)} test packages without a score")
        rows.append(s.loc[test_paths].values)
    return np.vstack(rows), label


def load_test_data():
    """Test packages, labels, family weights and families: the same as gnn_eval.Evaluator."""
    from baseline_protocol import DATA_DIR, SPLIT_FILE, load_features
    feats, _ = load_features()
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    test_idx = np.where(role == "test")[0]
    fam = feats["path"].map(dict(zip(splits["path"], splits["campaign_id"]))).fillna("").values[test_idx]
    w = feats["path"].map(dict(zip(splits["path"], splits["family_weight"].astype(float)))).values[test_idx]
    return feats["path"].values[test_idx], feats["label"].values[test_idx], w, fam


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(spec_file, results_dir, test_data, boot=None, seed_resample=None):
    with open(spec_file) as f:
        spec = json.load(f)
    paths, y, w, fam = test_data
    n_boot = boot or spec.get("boot", 5000)
    if seed_resample is None:
        seed_resample = spec.get("seed_resample", True)
    boots = resample_indices(y, fam, n_boot, seed=spec.get("bootstrap_seed", 0))
    print(f"test {int((y == 0).sum())} normal + {int((y == 1).sum())} malicious "
          f"({len(set(fam[y == 1]))} families) | {n_boot} paired resamples | training seeds "
          f"{'resampled' if seed_resample else 'fixed'}", flush=True)
    cache, rows = {}, []
    for family, comps in spec["families"].items():
        fam_rows = []
        for c in comps:
            key = json.dumps([c["a"], c["b"]], sort_keys=True)
            if key not in cache:
                A, la = load_scores(c["a"], results_dir, paths)
                B, lb = load_scores(c["b"], results_dir, paths)
                print(f"  computing {la} vs {lb}", flush=True)
                cache[key] = (la, lb, paired_bootstrap(A, B, y, w, boots, seed_resample=seed_resample,
                                                       seed=spec.get("bootstrap_seed", 0)))
            la, lb, res = cache[key]
            r = res[c["metric"]]
            fam_rows.append({"family": family, "id": c["id"], "question": c.get("question", ""),
                             "metric": c["metric"], "a": la, "b": lb, "a_value": r["a"], "b_value": r["b"],
                             "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"], "p": r["p"],
                             "boot_sd": r["boot_sd"], "seeds_resampled": seed_resample,
                             "a_per_seed": " ".join(f"{v:.4f}" for v in r["a_seeds"]),
                             "b_per_seed": " ".join(f"{v:.4f}" for v in r["b_seeds"])})
        adj = holm([r["p"] for r in fam_rows])
        alpha = spec.get("alpha", 0.05)
        for r, pa in zip(fam_rows, adj):
            r["p_holm"], r["significant"] = pa, bool(pa < alpha)
        rows += fam_rows
    out = pd.DataFrame(rows)
    for fam_name, g in out.groupby("family", sort=False):
        print(f"\n=== {fam_name}: {len(g)} comparisons, Holm at alpha {spec.get('alpha', 0.05)} ===")
        for r in g.itertuples():
            pct = r.metric == "tpr1_fw"
            f = (lambda v: f"{100 * v:+.1f} pts") if pct else (lambda v: f"{v:+.3f}")
            g_ = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
            print(f"{r.id:5s} {r.metric:8s} {r.a} = {g_(r.a_value)} vs {r.b} = {g_(r.b_value)}: "
                  f"diff {f(r.diff)} [{f(r.ci_lo)}, {f(r.ci_hi)}] p {r.p:.4f} Holm {r.p_holm:.4f}"
                  f"{'  *' if r.significant else ''}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Pre-registered paired comparisons (Methodology section 38)")
    ap.add_argument("--spec", default="config/primary_comparisons.json")
    ap.add_argument("--boot", type=int, default=None, help="overrides the spec (default 5000)")
    ap.add_argument("--no-seed-resample", action="store_true",
                    help="package variability only (training seeds fixed); default: seeds resampled too")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()
    from baseline_protocol import RESULTS_DIR
    out = run(args.spec, RESULTS_DIR, load_test_data(), boot=args.boot,
              seed_resample=False if args.no_seed_resample else None)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(f"{RESULTS_DIR}/paired_comparisons{args.tag}.csv", index=False)
    print(f"\nsaved {RESULTS_DIR}/paired_comparisons{args.tag}.csv")


if __name__ == "__main__":
    main()
