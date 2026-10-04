"""
diagnose_cola.py
Why does plain CoLA fail (inverted ROC-AUC) on the package graph?  Two checks.

A. Degree dependence (from cola_scores.csv, no training)
   Does the CoLA score of a test package follow its connectivity (degree,
   number of peers) rather than its content? Computed on NORMAL test packages
   only (so the class cannot explain it): rank correlation of the score with
   out-degree / in-degree / number of peers, and how well the score separates
   connected from isolated normals. Plus mean score percentile by class and
   degree bucket.

B. Contextual-anomaly injection (the standard CoLA benchmark protocol)
   On the REAL clean graph (graph_c00_s0.pt), take M connected normal test
   packages and replace their features with those of another held-out normal
   package. Two donor choices:
     farthest  the most distant of 50 random donors (CoLA's benchmark protocol,
               Liu et al., 2021; Ding et al., 2019). With heavy-tailed features
               the farthest donor is often a global outlier itself, so per-
               package detectors can find it too -> not a pure contextual test.
     random    a random normal donor: the injected package looks like an
               ordinary package on its own and only mismatches its context.
               This is the pure test of whether the dependency context
               constrains what a package looks like (homophily).
   Evaluated among connected normal test packages (injected = positive).
     - CoLA (p2 = 0 and 1, 3 training seeds): finds them -> the graph supports
       contextual detection, and the failure on malware is about the anomaly
       type. Misses them -> the graph itself (featureless context nodes,
       non-homophilous neighbourhoods) prevents it.
     - OCSVM on the same bundle features: should be near chance (the injected
       feature vectors are ordinary), as a control.
   Training is unaffected by the injection: walks from training packages never
   enter held-out packages.

Usage
  python src/diagnose_cola.py                 # both parts (B needs a GPU job or a few CPU minutes)
  python src/diagnose_cola.py --skip-injection
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(__file__))
from baseline_protocol import DATA_DIR, RESULTS_DIR  # noqa: E402
from build_graph import load_bundle  # noqa: E402
from train_cola import ScoringGraph, score, train  # noqa: E402

M_INJECT = 150
CANDIDATES = 50


def part_a(tag):
    s = pd.read_csv(f"{RESULTS_DIR}/cola_scores{tag}.csv", dtype={"path": str})
    conn = pd.read_csv(f"{DATA_DIR}/heldout_connectivity.csv", dtype={"path": str})
    s = s[(s["level"] == 0.0) & (s["split"] == "test") & (s["seed"] == s["seed"].min())]
    print("\n=== A. Degree dependence of the CoLA score (level 0, test, first seed) ===")
    for p2, d in s.groupby("p2"):
        d = d.merge(conn, on="path", how="inner")
        n = d[d["label"] == 0]
        print(f"\np2 = {p2}: {len(d)} test packages")
        for col in ["out_deg", "in_deg", "n_peers"]:
            if n[col].nunique() < 2:
                print(f"  normals: Spearman(score, {col:8s}) = n/a (constant)")
                continue
            rho = spearmanr(n["score"], n[col]).correlation
            print(f"  normals: Spearman(score, {col:8s}) = {rho:+.3f}")
        auc_conn = roc_auc_score(n["connected"].astype(int), n["score"])
        print(f"  normals: ROC-AUC of the score for 'connected' = {auc_conn:.3f} "
              f"(0.5 = unrelated; < 0.5 = isolated packages score as MORE anomalous)")
        d["pct"] = d["score"].rank(pct=True)
        d["deg_bucket"] = pd.cut(d["out_deg"] + d["in_deg"], [-1, 0, 1, 3, 9, 1e9],
                                 labels=["0", "1", "2-3", "4-9", "10+"])
        tab = d.pivot_table(index="deg_bucket", columns="label", values="pct", aggfunc=["mean", "size"],
                            observed=False)
        tab.columns = [f"{a}_{'normal' if b == 0 else 'malicious'}" for a, b in tab.columns]
        print("  mean score percentile (higher = more anomalous) by degree bucket:")
        print(tab.round(3).to_string())


def inject(graph, targets, donors_pool, rng, mode):
    """Replace each target's features with a donor's: the most distant of
    CANDIDATES random donors ("farthest") or one random donor ("random")."""
    x = graph.x.clone()
    for t in targets:
        cand = rng.choice(donors_pool[donors_pool != t], size=CANDIDATES, replace=False)
        if mode == "farthest":
            dist = torch.cdist(x[t:t + 1], x[cand]).squeeze(0)
            x[t] = x[cand[int(torch.argmax(dist))]]
        else:
            x[t] = x[cand[0]]
    return x


def part_b(epochs, rounds, device):
    print("\n=== B. Contextual-anomaly injection on the real clean graph ===")
    b = load_bundle(f"{DATA_DIR}/graph_c00_s0.pt")
    graph = ScoringGraph(b)
    conn = pd.read_csv(f"{DATA_DIR}/heldout_connectivity.csv", dtype={"path": str})
    pos = {p: i for i, p in enumerate(graph.held_paths)}
    held = conn[conn["path"].isin(pos)].copy()
    held["node"] = [graph.held_node[pos[p]] for p in held["path"]]
    pop = held[(held["split"] == "test") & (held["label"] == 0) & held["connected"]]
    donors = held.loc[held["label"] == 0, "node"].values
    rng = np.random.default_rng(0)
    targets = rng.choice(pop["node"].values, size=min(M_INJECT, len(pop)), replace=False)
    y = np.isin(pop["node"].values, targets).astype(int)
    has_peers = pop["has_peers"].values
    print(f"population: {len(pop)} connected normal test packages, {int(y.sum())} injected "
          f"({int(has_peers[y == 1].sum())} of them with peers)")

    x_clean = graph.x
    q = pop["node"].values
    tr = graph.train_anchors
    oc = OneClassSVM(kernel="rbf", gamma=1.0 / x_clean.shape[1], nu=0.05).fit(x_clean[tr].numpy())
    models = {}
    for mode in ["farthest", "random"]:
        x_inj = inject(graph, targets, donors, np.random.default_rng(1), mode)
        s_oc = -oc.decision_function(x_inj[q].numpy())
        print(f"\n[{mode} donor] OCSVM (control, own features only): ROC-AUC {roc_auc_score(y, s_oc):.3f}")
        for p2 in [0.0, 1.0]:
            aucs, aucs_peers = [], []
            for seed in [0, 1, 2]:
                if (p2, seed) not in models:   # training never sees held-out features
                    graph.x = x_clean
                    models[(p2, seed)] = train(graph, p2, seed, epochs, 300, 1e-3, device)
                graph.x = x_inj
                s = score(models[(p2, seed)], graph, q, p2, seed, rounds, 1200, device)
                aucs.append(roc_auc_score(y, s))
                m = has_peers
                aucs_peers.append(roc_auc_score(y[m], s[m]) if 0 < y[m].sum() < m.sum() else np.nan)
            graph.x = x_clean
            print(f"[{mode} donor] CoLA p2={p2}: ROC-AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
                  f"(seeds {', '.join(f'{a:.3f}' for a in aucs)}) | targets with peers only: "
                  f"{np.nanmean(aucs_peers):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="CoLA diagnostics")
    ap.add_argument("--tag", default="", help="suffix of cola_scores{tag}.csv")
    ap.add_argument("--skip-injection", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=64)
    args = ap.parse_args()
    part_a(args.tag)
    if not args.skip_injection:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\ndevice {device}")
        part_b(args.epochs, args.rounds, device)


if __name__ == "__main__":
    main()
