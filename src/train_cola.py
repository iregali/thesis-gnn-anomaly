"""
train_cola.py
Plain CoLA (Liu et al., 2021) on the package graph bundles, evaluated exactly
like the baselines (baseline_protocol.py): same test packages, contamination
draws, seeds, family weights, metrics, bootstrap, connectivity subgroups and
per-subgroup calibration.

Method (contrastive self-supervised graph anomaly detection)
  For a target node v, sample a small subgraph around it (size SUBGRAPH,
  target included), hide the target's own features in it ("anonymisation"),
  embed it with a 1-layer GCN and average it into a summary r(v). The target is
  embedded from its own features alone: h(v). A bilinear discriminator (CoLA's,
  losses.BilinearDiscriminator) learns to tell the pair (h(v), r(v)) from
  (h(v), r(u)) for another node u. Anomaly score = mean over R sampling rounds
  of  s(v, r(u)) - s(v, r(v)): high when v does not fit its own neighbourhood
  better than a random one.

Subgraph sampler (CoLA uses random walks with restart, mostly 1 hop)
  Each of the SUBGRAPH - 1 slots is a random neighbour of v; with probability
  p2 it is replaced by a random neighbour of that neighbour (2 hops).
  In this graph the 1-hop neighbours are mostly context nodes (dependency
  targets, no archive features), and the packages with full features that use
  the same dependency are 2 hops away, so p2 decides whether CoLA sees
  co-dependents. p2 = 0 is the faithful 1-hop CoLA ("CoLA 1-hop"); p2 in P2 is
  also chosen on validation ("CoLA").
  Isolated nodes get no neighbours: their slots are padded with zero vectors,
  so their summary is constant and their score depends on h(v) only.
  Subgraph topology: star around v (sampled nodes linked to v), self-loops,
  mean aggregation. Deviation from CoLA's code: induced edges among the
  sampled nodes are not used.

Scoring held-out packages (Methodology §13: inserted, never in training)
  All validation and test packages are inserted at once, but WITHOUT links to
  each other: each gets its edges to training-graph nodes, its own private
  copies of new context nodes, and edges from training packages that depend
  on its name; walks starting elsewhere never enter a held-out node. This is
  equivalent to inserting one at a time (build_graph.insert_package).

Protocol
  Level 0: bundle graph_c00_s0.pt for every seed (seeds differ only in
  training randomness). Level > 0: bundle graph_cXX_s{seed}.pt (the same
  contamination draw as the baselines' seed). p2 chosen on validation (pAUC up
  to 5% FPR) with seed 0, per level; reused for the other seeds.

Output: ~/thesis/results/cola_results{tag}.csv, cola_settings{tag}.csv,
cola_scores{tag}.csv (raw per-package scores, every level and seed).

Usage
  python src/train_cola.py                        # all levels, 3 seeds
  python src/train_cola.py --levels 0 --seeds 0 --epochs 5 --rounds 8   # quick test
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from baseline_protocol import (DATA_DIR, EXTRA_FPRS, RESULTS_DIR, SELECT_MAX_FPR, SPLIT_FILE,  # noqa: E402
                               bootstrap_ci, combine, load_features, normal_groups,
                               point_metrics, population_metrics, subgroup_metrics, to_ranks)
from build_graph import load_bundle  # noqa: E402
from connectivity import PeerIndex, connectivity_table, load_core_deps  # noqa: E402
from losses import BilinearDiscriminator  # noqa: E402

LEVELS = [0.0, 0.01, 0.02, 0.05]
SEEDS = [0, 1, 2]
P2 = [0.0, 0.5, 1.0]
SUBGRAPH = 4
DIM = 64


# ---------------------------------------------------------------------------
# Graph with held-out packages attached (no links between held-out packages)
# ---------------------------------------------------------------------------
class ScoringGraph:
    def __init__(self, bundle):
        g = bundle["graph"]
        n0 = g.num_nodes
        ei = g.edge_index.numpy()
        nbrs = [[] for _ in range(n0)]
        for a, b in zip(ei[0], ei[1]):
            nbrs[a].append(int(b))
        h = bundle["heldout"]
        nh = len(h)
        ctx_row = {n: k for k, n in enumerate(bundle["context_names"])}
        x_rows, extra_nbrs = [], []
        held_nbrs = []
        nxt = n0 + nh
        for i in range(nh):
            lst = []
            for t in bundle["heldout_deps"][i]:
                if t in bundle["node_of_name"]:
                    lst.append(bundle["node_of_name"][t])
                else:  # private copy of a context node not in the training graph
                    x_rows.append(bundle["x_context"][ctx_row[t]].numpy())
                    extra_nbrs.append([n0 + i])
                    lst.append(nxt)
                    nxt += 1
            tgt = bundle["node_of_name"].get(h["name"].iloc[i])
            if tgt is not None:  # training packages that depend on this name
                lst.extend(bundle["dependents"].get(tgt, []))
            held_nbrs.append(lst)
        all_nbrs = nbrs + held_nbrs + extra_nbrs
        self.indptr = np.zeros(len(all_nbrs) + 1, dtype=np.int64)
        self.indptr[1:] = np.cumsum([len(x) for x in all_nbrs])
        self.indices = np.array([v for x in all_nbrs for v in x], dtype=np.int64)
        x_extra = np.stack(x_rows) if x_rows else np.zeros((0, g.x.shape[1]), dtype=np.float32)
        self.x = torch.cat([g.x, bundle["x_heldout"].float(), torch.from_numpy(x_extra).float()])
        self.n_train, self.n_held = n0, nh
        self.train_anchors = np.where(~g.is_context.numpy())[0]       # evaluated training packages
        self.held_node = n0 + np.arange(nh)
        self.held_paths = h["path"].values

    def degree(self, v):
        return self.indptr[v + 1] - self.indptr[v]

    def random_neighbour(self, v, rng):
        deg = self.degree(v)
        pick = self.indptr[v] + (rng.random(len(v)) * np.maximum(deg, 1)).astype(np.int64)
        out = np.where(deg > 0, self.indices[np.minimum(pick, len(self.indices) - 1)], -1)
        return out

    def sample(self, anchors, p2, rng, size=SUBGRAPH):
        """[B, size] node ids (column 0 = anchor) and [B, size] pad mask."""
        B = len(anchors)
        nodes = np.empty((B, size), dtype=np.int64)
        pad = np.zeros((B, size), dtype=bool)
        nodes[:, 0] = anchors
        for j in range(1, size):
            n1 = self.random_neighbour(anchors, rng)
            node = n1.copy()
            if p2 > 0:
                go = (rng.random(B) < p2) & (n1 >= 0)
                n2 = np.full(B, -1)
                if go.any():
                    n2[go] = self.random_neighbour(n1[go], rng)
                ok = go & (n2 >= 0) & (n2 != anchors)
                node[ok] = n2[ok]
            pad[:, j] = node < 0
            nodes[:, j] = np.where(node < 0, anchors, node)
        return nodes, pad


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CoLA(nn.Module):
    def __init__(self, in_dim, dim=DIM):
        super().__init__()
        self.lin = nn.Linear(in_dim, dim, bias=False)
        self.act = nn.PReLU()
        self.disc = BilinearDiscriminator(dim)

    def embed(self, X_sub, x_target):
        """X_sub [B, S, F] with the target (column 0) and pads already zeroed.
        Star GCN with self-loops, mean aggregation: the centre averages all
        S nodes, every leaf averages itself and the (anonymised) centre."""
        Z = self.lin(X_sub)
        centre = Z.mean(dim=1, keepdim=True)
        leaves = (Z[:, 1:] + Z[:, :1]) / 2.0
        H = self.act(torch.cat([centre, leaves], dim=1))
        r = H.mean(dim=1)
        h = self.act(self.lin(x_target))
        return h, r

    def forward(self, X_sub, x_target):
        h, r = self.embed(X_sub, x_target)
        pos = self.disc(h, r)
        neg = self.disc(h, torch.roll(r, shifts=1, dims=0))
        return pos, neg


def batch_tensors(graph, nodes, pad, device):
    X = graph.x[torch.from_numpy(nodes)].clone()
    X[:, 0] = 0.0                                  # anonymise the target
    X[torch.from_numpy(pad)] = 0.0                 # padding for missing neighbours
    return X.to(device), graph.x[torch.from_numpy(nodes[:, 0])].to(device)


def train(graph, p2, seed, epochs, batch, lr, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CoLA(graph.x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    anchors = graph.train_anchors
    for ep in range(epochs):
        perm = rng.permutation(anchors)
        tot = 0.0
        for a in range(0, len(perm), batch):
            b = perm[a:a + batch]
            if len(b) < 2:
                continue
            nodes, pad = graph.sample(b, p2, rng)
            X, xt = batch_tensors(graph, nodes, pad, device)
            pos, neg = model(X, xt)
            loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) +
                    F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      epoch {ep + 1:3d}/{epochs}  loss {tot / len(perm):.4f}", flush=True)
    return model


@torch.no_grad()
def score(model, graph, query_nodes, p2, seed, rounds, batch, device):
    """CoLA anomaly score: mean over rounds of s(v, r(other)) - s(v, r(v))."""
    model.eval()
    rng = np.random.default_rng(10_000 + seed)
    out = np.zeros(len(query_nodes))
    for _ in range(rounds):
        perm = rng.permutation(len(query_nodes))
        for a in range(0, len(perm), batch):
            idx = perm[a:a + batch]
            nodes, pad = graph.sample(query_nodes[idx], p2, rng)
            X, xt = batch_tensors(graph, nodes, pad, device)
            pos, neg = model(X, xt)
            out[idx] += (neg - pos).cpu().numpy()
    model.train()
    return out / rounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def bundle_path(level, seed):
    return f"{DATA_DIR}/graph_c{round(level * 100):02d}_s{seed if level > 0 else 0}.pt"


def main():
    ap = argparse.ArgumentParser(description="Plain CoLA under the baseline protocol")
    ap.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--p2", type=float, nargs="*", default=P2, help="2-hop probabilities to choose from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rounds", type=int, default=64, help="sampling rounds when scoring")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.levels = sorted(args.levels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | torch {torch.__version__}", flush=True)
    missing = [bundle_path(lv, s) for lv in args.levels for s in args.seeds if not os.path.exists(bundle_path(lv, s))]
    if missing:
        raise SystemExit("missing bundles (build level 0 with: python src/build_graph.py --contam 0 --seed 0):\n  "
                         + "\n  ".join(sorted(set(missing))))

    # --- evaluation setup: identical to baseline_protocol.py ---
    feats, _ = load_features()
    y_all = feats["label"].values
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    fam_all = feats["path"].map(dict(zip(splits["path"], splits["campaign_id"]))).fillna("").values
    w_all = feats["path"].map(dict(zip(splits["path"], splits["family_weight"].astype(float)))).values
    train_idx = np.where((role == "train") & (y_all == 0))[0]
    val_idx, test_idx = np.where(role == "val")[0], np.where(role == "test")[0]
    y_val, y = y_all[val_idx], y_all[test_idx]
    w, fam = w_all[test_idx], fam_all[test_idx]
    hard = (y == 1) & (feats["gd_flagged"].values[test_idx] == 0)
    gd_fpr = float(feats["gd_flagged"].values[test_idx][y == 0].mean())  # for point_metrics' GuardDog column
    pop_groups, _ = normal_groups(feats, test_idx, y)
    deps_of = load_core_deps(DATA_DIR)
    conn = connectivity_table(feats, train_idx, test_idx, deps_of)
    sg = {"conn": conn["connected"].values, "iso": ~conn["connected"].values,
          "peers": conn["has_peers"].values, "nopeers": ~conn["has_peers"].values}
    val_paths, test_paths = feats["path"].values[val_idx], feats["path"].values[test_idx]
    print(f"test {int((y == 0).sum())}+{int((y == 1).sum())} | val {int((y_val == 0).sum())}+{int((y_val == 1).sum())}"
          f" | connected test: {sg['conn'].mean():.1%}", flush=True)

    rows, chosen, score_rows = [], [], []

    def summarise(level, name, seed_scores, seed_rows, setting):
        s_ens = np.mean(seed_scores, axis=0)
        row = {"level": level, "method": name, **point_metrics(s_ens, y, w, hard, gd_fpr),
               **bootstrap_ci(s_ens, y, w, fam, args.boot), **population_metrics(s_ens, y, w, pop_groups),
               **subgroup_metrics(s_ens, y, w, sg)}
        per_seed = pd.DataFrame(seed_rows)
        for col in ["auc_fw", "tpr1_fw", "tpr1_flagged_fw", "tpr1_hard_fw", *EXTRA_FPRS]:
            v = per_seed[col]
            row[f"{col}_seed_mean"], row[f"{col}_seed_std"] = v.mean(), (v.std() if len(v) > 1 else 0.0)
            row[f"{col}_seed_min"], row[f"{col}_seed_max"] = v.min(), v.max()
        row["setting"] = setting
        print(f"level {level:.0%} | {name:34s} ROC-AUC (fw) {row['auc_fw']:.3f} [{row['auc_fw_lo']:.3f}, "
              f"{row['auc_fw_hi']:.3f}] | caught@1%FPR (fw, seed mean) {row['tpr1_fw_seed_mean']:.1%} | "
              f"@0.5% {row['tpr05_fw']:.1%} | connected: AUC {row.get('sg_auc_fw_conn', np.nan):.3f}, "
              f"caught {row.get('sg_tpr1_fw_conn', np.nan):.1%} | isolated: AUC "
              f"{row.get('sg_auc_fw_iso', np.nan):.3f}", flush=True)
        print(f"    per seed caught@1%FPR (fw): {', '.join(f'{v:.1%}' for v in per_seed['tpr1_fw'])}", flush=True)
        return row

    for level in args.levels:
        graphs = {}
        p2_best = None
        runs = {}  # p2 -> list of (val scores, test scores, fit paths) per seed
        for seed in args.seeds:
            path = bundle_path(level, seed)
            if path not in graphs:
                t0 = time.time()
                b = load_bundle(path)
                graphs[path] = (ScoringGraph(b), b)
                print(f"[level {level:.0%}] loaded {os.path.basename(path)} ({time.time() - t0:.0f}s)", flush=True)
            graph, b = graphs[path]
            held_pos = {p: i for i, p in enumerate(graph.held_paths)}
            vq = graph.held_node[[held_pos[p] for p in val_paths]]
            tq = graph.held_node[[held_pos[p] for p in test_paths]]
            nodes = b["nodes"]
            fit_paths = nodes.loc[nodes["kind"] == "evaluated", "path"].values

            todo = list(args.p2) if (seed == args.seeds[0]) else sorted({0.0, p2_best})
            for p2 in todo:
                print(f"  level {level:.0%} seed {seed} p2 {p2}: training", flush=True)
                t0 = time.time()
                model = train(graph, p2, seed, args.epochs, args.batch, args.lr, device)
                sv = score(model, graph, vq, p2, seed, args.rounds, args.batch * 4, device)
                st = score(model, graph, tq, p2, seed, args.rounds, args.batch * 4, device)
                pauc = roc_auc_score(y_val, sv, max_fpr=SELECT_MAX_FPR)
                print(f"    val pAUC(<=5% FPR) {pauc:.3f} | {time.time() - t0:.0f}s", flush=True)
                runs.setdefault(p2, []).append((sv, st, fit_paths))
                if seed == args.seeds[0]:
                    chosen.append({"level": level, "p2": p2, "val_pauc": pauc})
                for which, pths, s_ in [("val", val_paths, sv), ("test", test_paths, st)]:
                    score_rows.append(pd.DataFrame({"level": level, "seed": seed, "p2": p2, "split": which,
                                                    "path": pths, "score": s_}))
            if seed == args.seeds[0]:
                cands = [c for c in chosen if c["level"] == level]
                p2_best = max(cands, key=lambda c: c["val_pauc"])["p2"]
                print(f"  [level {level:.0%}] p2 chosen on validation: {p2_best}", flush=True)

        variants = [("CoLA 1-hop", 0.0)] + ([("CoLA", p2_best)] if p2_best != 0.0 else [])
        for name, p2 in variants:
            per_seed = runs[p2]
            seed_scores = [to_ranks(st) for _, st, _ in per_seed]
            seed_rows = [point_metrics(st, y, w, hard, gd_fpr) for _, st, _ in per_seed]
            label = f"{name} (p2={p2})"
            rows.append(summarise(level, label, seed_scores, seed_rows, f"p2={p2}"))
            # per-subgroup calibration (same function and subgroups as the baselines)
            cal_scores, cal_rows = [], []
            for sv, st, fit_paths in per_seed:
                fit_mask = feats["path"].isin(set(fit_paths)).values
                idx = PeerIndex(feats["path"].values[fit_mask], feats["Name"].values[fit_mask], deps_of)
                has_v = np.asarray((idx.query(val_paths, feats["Name"].values[val_idx], deps_of) > 0).sum(1)).ravel() > 0
                has_t = np.asarray((idx.query(test_paths, feats["Name"].values[test_idx], deps_of) > 0).sum(1)).ravel() > 0
                c = combine(np.where(has_v, sv, np.nan), np.where(has_t, st, np.nan), sv, st, y_val)
                cal_scores.append(to_ranks(c))
                cal_rows.append(point_metrics(c, y, w, hard, gd_fpr))
            rows.append(summarise(level, f"{label}, subgroup-calibrated", cal_scores, cal_rows, f"p2={p2}"))
            # ties among isolated test packages (diagnostic: constant summary for isolated nodes)
            st0 = per_seed[0][1]
            iso = sg["iso"]
            print(f"    isolated test packages: {len(np.unique(np.round(st0[iso], 6)))} distinct scores "
                  f"of {int(iso.sum())}", flush=True)

    results, chosen = pd.DataFrame(rows), pd.DataFrame(chosen)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/cola_results{args.tag}.csv", index=False)
    chosen.to_csv(f"{RESULTS_DIR}/cola_settings{args.tag}.csv", index=False)
    pd.concat(score_rows).to_csv(f"{RESULTS_DIR}/cola_scores{args.tag}.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== CoLA test results (family-weighted) ===")
    print(results[["level", "method", "auc_fw", "auc_fw_lo", "auc_fw_hi", "tpr05_fw", "tpr1_fw_seed_mean",
                   "tpr1_fw_seed_std", "tpr2_fw", "auc_hard"]].round(3).to_string(index=False))
    sg_cols = [c for g in ["conn", "iso", "peers"] for c in
               (f"sg_auc_fw_{g}", f"sg_tpr1_fw_{g}", f"sg_tpr1glob_fw_{g}") if c in results.columns]
    print("\n=== Connectivity subgroups: ROC-AUC fw | caught @ 1% FPR within subgroup | at the global threshold ===")
    print(results[["level", "method"] + sg_cols].round(3).to_string(index=False))
    print("\n=== Validation choice of p2 ===")
    print(chosen.round(3).to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/cola_*{args.tag}.csv")


if __name__ == "__main__":
    main()
