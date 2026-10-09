"""
sensitivity_check.py
Does the contrastive encoder ignore the features that barely vary among normal
packages? (Methodology §36.) Diagnostic only: nothing is tuned on it.

Hypothesis (point 1 of §35): an encoder trained only on normal packages learns
to represent the variation that exists among them. A feature that is (almost)
constant among normal packages gets no training signal, so when a malicious
package differs in exactly that feature, the embedding barely moves. The raw
One-Class SVM has no such problem: every feature is a dimension.

Procedure (validation packages; labels only to describe features, never to
train or select):
  1. Train the encoder as in train_contrastive.py (default InfoNCE + GIN,
     depths 0 and 1, clean bundle) and keep an UNTRAINED copy of the same
     architecture (same seed) as reference.
  2. For every feature j: take validation NORMAL packages and replace their own
     value of j by values of the same feature drawn from validation MALWARE
     (realistic malware-like values; neighbourhood unchanged), and measure
         gain_j = mean ||z(x') - z(x)|| / scale(z)  /  mean |x'_j - x_j|
     (scale(z) = spread of the normal embeddings, so trained and untrained
     encoders are comparable). An identity map (the raw OCSVM's view) has the
     same gain for every feature.
  3. Relate gain_j to
       variability_j = share of TRAINING packages not at the feature's most
                       common value (low = almost constant among normals)
       separation_j  = |mean_malware - mean_normal| / sd_normal on validation
                       (how much the feature distinguishes malware)
     and to the untrained reference (ratio = trained / untrained gain:
     < 1 = training suppressed the feature, > 1 = training amplified it).

Reading
  low-variability and high-separation features have clearly LOWER gain (and a
  ratio < 1) than common features -> point 1 confirmed: the objective should
  force sensitivity to every feature (sensitivity or reconstruction term).
  no relation -> point 1 is not the explanation either.

Outputs: RESULTS_DIR/sensitivity_check{tag}.csv (one row per feature x depth x
seed) and a printed summary (rank correlations, group means, top separating
features).

Usage
  python src/sensitivity_check.py                         # InfoNCE + GIN, depths 0 1, seeds 0 1 2
  python src/sensitivity_check.py --encoder gps --depths 1
  # mechanism check of the proposed loss (Methodology section 38): same settings as the run, e.g.
  python src/sensitivity_check.py --loss triplet --depths 1 --epochs 25 --dim 128 \
      --strata conn --shift rare --rare-nu 0.05 --shift-k 4 --shift-weight 1 --tag _triplet_gin_n3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import RESULTS_DIR  # noqa: E402
from build_graph import load_bundle  # noqa: E402
from train_cola import ScoringGraph, bundle_path  # noqa: E402
import train_contrastive as TC  # noqa: E402


def variability(x):
    """Share of rows not at the column's most common value (rounded), per column."""
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        _, cnt = np.unique(np.round(x[:, j], 6), return_counts=True)
        out[j] = 1.0 - cnt.max() / len(x)
    return out


@torch.no_grad()
def gains(model, graph, anchors, mal_rows, depth, args, device, seed, n_draw):
    """Per-feature normalised embedding gain for perturbations of the package's own row."""
    model.eval()
    rng = np.random.default_rng(20_000 + seed)
    inp = list(TC.view(graph, anchors, depth, rng, device, args, training=False))   # one fixed neighbourhood
    x0 = inp[0]
    z0 = model(*inp)
    scale = torch.sqrt(((z0 - z0.mean(0)) ** 2).sum(1).mean()).item() + 1e-12
    F = x0.shape[1]
    out = np.full(F, np.nan)
    for j in range(F):
        vals = mal_rows[:, j]
        acc_z, acc_x = 0.0, 0.0
        for _ in range(n_draw):
            new = torch.from_numpy(rng.choice(vals, size=len(anchors))).float().to(device)
            dx = (new - x0[:, j]).abs()
            if dx.sum() <= 1e-9:
                continue
            x1 = x0.clone()
            x1[:, j] = new
            dz = (model(*([x1] + inp[1:])) - z0).norm(dim=1)
            keep = dx > 1e-9
            acc_z += dz[keep].sum().item() / scale
            acc_x += dx[keep].sum().item()
        if acc_x > 0:
            out[j] = acc_z / acc_x
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description="Embedding sensitivity per feature (point 1 diagnostic)")
    ap.add_argument("--loss", default="infonce", choices=["infonce", "triplet", "dgi"])
    ap.add_argument("--encoder", default="gin", choices=["gin", "gat", "tf", "tconv", "gps"])
    ap.add_argument("--depths", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-anchors", type=int, default=1000, help="validation normal packages perturbed")
    ap.add_argument("--n-draw", type=int, default=3, help="malware values drawn per feature and package")
    ap.add_argument("--tag", default=None)
    # training settings: the same defaults as train_contrastive.py
    ap.add_argument("--gps-memory", type=int, default=512)
    ap.add_argument("--fanout", type=int, default=TC.FANOUT)
    ap.add_argument("--isolated", default="self", choices=["self", "empty"])
    ap.add_argument("--skip", default="concat", choices=["concat", "none"])
    ap.add_argument("--rescale", default="all", choices=["bundle", "all"])
    ap.add_argument("--agg", default="mean", choices=["sum", "mean"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--mining", default="random", choices=["random", "hard"])
    ap.add_argument("--mask-p", type=float, default=0.2)
    ap.add_argument("--drop-p", type=float, default=0.2)
    ap.add_argument("--context-features", default="users", choices=["none", "users"])
    ap.add_argument("--context-min-users", type=int, default=3)
    ap.add_argument("--dim", type=int, default=TC.DIM, help="embedding size (tuned runs: tuning_choice.csv)")
    # proposed loss (structure_rarity.py); off by default, same meaning as in train_contrastive.py
    ap.add_argument("--strata", default="none", choices=["none", "conn"])
    ap.add_argument("--shift", default="none", choices=["none", "rare", "uniform"])
    ap.add_argument("--rare-nu", type=float, default=0.05)
    ap.add_argument("--shift-k", type=int, default=1)
    ap.add_argument("--shift-nfeat", type=int, default=1)
    ap.add_argument("--shift-weight", type=float, default=1.0)
    args = ap.parse_args()
    args.mem = args.gps_memory if args.encoder == "gps" else 0
    tag = args.tag if args.tag is not None else f"_{args.loss}_{args.encoder}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    b = load_bundle(bundle_path(0.0, 0))
    graph = ScoringGraph(b, args.context_features, args.context_min_users, args.rescale)
    graph.feature_names = list(b["feature_names"])
    names = list(b["feature_names"])
    h = b["heldout"]
    split = h["split"].values if "split" in h.columns else None
    lab = h["label"].astype(int).values
    is_val = (split == "val") if split is not None else np.ones(len(h), bool)
    val_nor = graph.held_node[is_val & (lab == 0)]
    val_mal = graph.held_node[is_val & (lab == 1)]
    X = graph.x.numpy()
    var_train = variability(X[graph.train_anchors])
    mn, mm = X[val_nor], X[val_mal]
    separation = np.abs(mm.mean(0) - mn.mean(0)) / (mn.std(0) + 1e-6)
    print(f"device {device} | {args.loss} + {args.encoder} | validation: {len(val_nor)} normal, {len(val_mal)} "
          f"malware | features {len(names)}", flush=True)

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        anchors = rng.choice(val_nor, size=min(args.n_anchors, len(val_nor)), replace=False)
        for depth in args.depths:
            torch.manual_seed(seed)
            untrained = TC.Contrastive(graph.x.shape[1], depth, args).to(device)
            model = TC.train(graph, depth, seed, args, device)
            g_tr = gains(model, graph, anchors, mm, depth, args, device, seed, args.n_draw)
            g_un = gains(untrained, graph, anchors, mm, depth, args, device, seed, args.n_draw)
            for j, nm in enumerate(names):
                rows.append({"seed": seed, "depth": depth, "feature": nm, "variability": var_train[j],
                             "separation": separation[j], "gain_trained": g_tr[j], "gain_untrained": g_un[j],
                             "ratio": g_tr[j] / g_un[j] if g_un[j] and np.isfinite(g_un[j]) else np.nan})
            print(f"  seed {seed} depth {depth}: done", flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = f"{RESULTS_DIR}/sensitivity_check{tag}.csv"
    res.to_csv(out, index=False)
    m = (res.groupby(["depth", "feature"])[["variability", "separation", "gain_trained", "gain_untrained", "ratio"]]
         .mean().reset_index().dropna(subset=["gain_trained"]))
    m = m[m["feature"] != "is_context"]
    pd.set_option("display.width", 250)
    print("\n=== Rank correlations (features; mean over seeds) ===")
    for depth, d in m.groupby("depth"):
        r1 = spearmanr(d["variability"], d["gain_trained"]).correlation
        r2 = spearmanr(d["variability"], d["ratio"]).correlation
        r3 = spearmanr(d["separation"], d["gain_trained"]).correlation
        r4 = spearmanr(d["separation"], d["ratio"]).correlation
        print(f"depth {depth}: gain vs variability {r1:+.2f} | ratio vs variability {r2:+.2f} | "
              f"gain vs separation {r3:+.2f} | ratio vs separation {r4:+.2f}  ({len(d)} features)")
    print("  (point 1 predicts: gain and ratio rise with variability (> 0), and fall with separation (< 0))")
    print("\n=== Mean gain by how much the feature varies among normal training packages ===")
    m["variability_group"] = pd.cut(m["variability"], [-0.01, 0.05, 0.25, 1.0],
                                    labels=["almost constant (<5%)", "rare (5-25%)", "common (>25%)"])
    print(m.groupby(["depth", "variability_group"], observed=True)[["gain_trained", "gain_untrained", "ratio"]]
          .agg(["mean", "count"]).round(3).to_string())
    print("\n=== The 15 features that separate malware most (validation) ===")
    top = m.sort_values("separation", ascending=False).groupby("depth").head(15)
    print(top[["depth", "feature", "separation", "variability", "gain_trained", "gain_untrained", "ratio"]]
          .round(3).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()