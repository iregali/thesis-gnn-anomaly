"""
train_ocgnn.py
Plain one-class GNN (Deep SVDD on GNN embeddings; OCGNN, Wang et al. 2021;
Deep SVDD, Ruff et al. 2018) on the package graph bundles, evaluated exactly
like the baselines and CoLA (gnn_eval.Evaluator). Reference model for any loss
contribution (Methodology §24).

Why one-class (§24): malware is mainly an ATTRIBUTE anomaly, and CoLA's
neighbourhood-fit score measured connectivity. Here each package is embedded
from its OWN features plus its sampled neighbourhood, and scored by its
distance to the centre of the normal training embeddings. Isolated packages are
scored by their own features (like the OCSVM); connected packages get context
on top, so any gain over the depth-0 model is attributable to structure.

Neighbourhood (GraphSAGE-style fanout sampling, with replacement)
  depth 0  own features only (MLP): Deep SVDD without the graph = the
           neural, structure-free reference
  depth 1  + FANOUT direct neighbours (mostly context nodes: dependency targets)
  depth 2  + FANOUT neighbours of each (co-dependents with full features)
  Packages without neighbours (--isolated): "self" (default) pads every
  neighbour slot with the package itself, so isolated and connected packages
  pass through the same aggregation at the same scale (otherwise a sum over 0
  vs FANOUT neighbours separates them by degree alone, the bias that made
  CoLA's score a connectivity detector, §24); "empty" leaves the slots masked.
  Held-out packages are attached without links to each other (as in
  train_cola.ScoringGraph), equivalent to one-at-a-time insertion.

Context nodes: by default (--context-features users) a dependency target is
described by the mean features of its training users if it has at least
--context-min-users (3) of them, else zero (Methodology §24); leave-one-out
for training anchors.

Encoders (--encoder; 1 or 2 message-passing layers, all without bias terms)
  gin   (1 + eps) * self + sum(neighbours) -> 2-layer MLP
  gat   single-head attention over self + neighbours
  tf    dot-product attention (query = self, keys/values = self + neighbours)

Own-feature skip (--skip concat, default): for depth >= 1 the embedding is
[MLP(own features) ; GNN output], GraphSAGE-style, so the package's own
features (where malware stands out, §24) are not diluted by aggregating
FANOUT neighbours; --skip none uses the GNN output alone.

Deep SVDD safeguards against the trivial solution (all embeddings = centre):
no bias terms anywhere, centre fixed after one forward pass of the initial
network (coordinates with |c| < 0.1 set to +/-0.1), weight decay.
Score = squared distance to the centre, averaged over --rounds sampling rounds.

Protocol (as train_cola.py): level 0 uses graph_c00_s0.pt for every seed,
level > 0 graph_cXX_s{seed}.pt. Every depth is trained for every seed and
reported; the depth with the best validation pAUC (<= 5% FPR, seed 0) is
marked in ocgnn_settings{tag}.csv. Plain and subgroup-calibrated rows.

Usage
  python src/train_ocgnn.py                               # GIN, all levels, 3 seeds
  python src/train_ocgnn.py --encoder gat --tag _gat
  python src/train_ocgnn.py --levels 0 --seeds 0 --epochs 2 --rounds 2 --boot 20   # quick test
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
from baseline_protocol import DATA_DIR, RESULTS_DIR, SELECT_MAX_FPR  # noqa: E402
from build_graph import load_bundle  # noqa: E402
from gnn_eval import Evaluator  # noqa: E402
from train_cola import ScoringGraph, bundle_path  # noqa: E402

LEVELS = [0.0, 0.01, 0.02, 0.05]
SEEDS = [0, 1, 2]
DEPTHS = [0, 1, 2]
FANOUT = 5
DIM = 64


# ---------------------------------------------------------------------------
# Neighbourhood sampling: fixed-size trees, with masks
# ---------------------------------------------------------------------------
def sample_tree(graph, anchors, depth, fanout, rng, isolated="self"):
    """Returns node ids n1 [B, f], n2 [B, f, f] and masks m1, m2 (False = padding).
    isolated="self": slots without a neighbour are filled with the node itself (mask True)."""
    B = len(anchors)
    if depth == 0:
        return None, None, None, None
    rep1 = np.repeat(anchors, fanout)
    n1 = graph.random_neighbour(rep1, rng).reshape(B, fanout)
    m1 = n1 >= 0
    n1 = np.where(m1, n1, anchors[:, None])
    if isolated == "self":
        m1 = np.ones_like(m1)
    if depth == 1:
        return n1, m1, None, None
    rep2 = np.repeat(n1.ravel(), fanout)
    n2 = graph.random_neighbour(rep2, rng).reshape(B, fanout, fanout)
    m2 = (n2 >= 0) & m1[:, :, None]
    n2 = np.where(n2 >= 0, n2, n1[:, :, None])
    if isolated == "self":
        m2 = np.ones_like(m2)
    return n1, m1, n2, m2


def gather(graph, anchors, nodes, training):
    """Feature rows for nodes (any shape [B, ...]); leave-one-out of the anchor's own
    contribution to context profiles during training (see train_cola.ScoringGraph)."""
    shape = nodes.shape
    flat = nodes.reshape(shape[0], -1)
    X = graph.x[torch.from_numpy(flat)].clone()
    if training and graph.loo_keys is not None:
        X = _loo(graph, X, anchors, flat)
    return X.reshape(*shape, X.shape[-1])


def _loo(graph, X, anchors, flat):
    N = graph.x.shape[0]
    hit = np.isin(anchors[:, None] * N + flat, graph.loo_keys)
    if not hit.any():
        return X
    b, j = np.nonzero(hit)
    c = torch.from_numpy(flat[b, j])
    n = graph.ctx_n[c]
    loo = (graph.ctx_sum[c] - graph.x0_archive[torch.from_numpy(anchors[b])]) / torch.clamp(n - 1, min=1)[:, None]
    loo[n <= 1] = 0.0
    X[torch.from_numpy(b)[:, None], torch.from_numpy(j)[:, None], graph.archive[None, :]] = loo
    return X


# ---------------------------------------------------------------------------
# Encoders (no bias terms: Deep SVDD)
# ---------------------------------------------------------------------------
class Layer(nn.Module):
    def __init__(self, kind, din, dout):
        super().__init__()
        self.kind = kind
        if kind == "gin":
            self.eps = nn.Parameter(torch.zeros(1))
            self.mlp = nn.Sequential(nn.Linear(din, dout, bias=False), nn.LeakyReLU(0.1),
                                     nn.Linear(dout, dout, bias=False))
        elif kind == "gat":
            self.W = nn.Linear(din, dout, bias=False)
            self.a_self = nn.Parameter(torch.randn(dout) * 0.1)
            self.a_nbr = nn.Parameter(torch.randn(dout) * 0.1)
        elif kind == "tf":
            self.q = nn.Linear(din, dout, bias=False)
            self.k = nn.Linear(din, dout, bias=False)
            self.v = nn.Linear(din, dout, bias=False)
            self.o = nn.Linear(dout, dout, bias=False)
        else:
            raise ValueError(kind)

    def forward(self, c, n, m):
        """c [..., din] centre, n [..., k, din] neighbours, m [..., k] mask -> [..., dout]."""
        mf = m.unsqueeze(-1).float()
        if self.kind == "gin":
            return F.leaky_relu(self.mlp((1 + self.eps) * c + (n * mf).sum(-2)), 0.1)
        if self.kind == "gat":
            wc, wn = self.W(c), self.W(n)
            e_self = F.leaky_relu((wc * self.a_self).sum(-1) + (wc * self.a_nbr).sum(-1), 0.2)
            e_n = F.leaky_relu((wc * self.a_self).sum(-1, keepdim=True) + (wn * self.a_nbr).sum(-1), 0.2)
            e_n = e_n.masked_fill(~m, float("-inf"))
            att = torch.softmax(torch.cat([e_self.unsqueeze(-1), e_n], dim=-1), dim=-1)
            vals = torch.cat([wc.unsqueeze(-2), wn], dim=-2)
            return F.leaky_relu((att.unsqueeze(-1) * vals).sum(-2), 0.1)
        # transformer-style attention
        q = self.q(c).unsqueeze(-2)
        kv_in = torch.cat([c.unsqueeze(-2), n], dim=-2)
        keys, vals = self.k(kv_in), self.v(kv_in)
        logits = (q * keys).sum(-1) / keys.shape[-1] ** 0.5
        mask = torch.cat([torch.ones_like(m[..., :1]), m], dim=-1)
        att = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        return F.leaky_relu(self.o((att.unsqueeze(-1) * vals).sum(-2)), 0.1)


class OCGNN(nn.Module):
    def __init__(self, in_dim, depth, kind, dim=DIM, skip="concat"):
        super().__init__()
        self.depth = depth
        self.skip = skip if depth > 0 else "none"
        if self.skip == "concat":
            self.own = nn.Sequential(nn.Linear(in_dim, dim, bias=False), nn.LeakyReLU(0.1),
                                     nn.Linear(dim, dim, bias=False))
        if depth == 0:
            self.mlp = nn.Sequential(nn.Linear(in_dim, dim, bias=False), nn.LeakyReLU(0.1),
                                     nn.Linear(dim, dim, bias=False))
        else:
            self.l1 = Layer(kind, in_dim, dim)
            if depth == 2:
                self.l2 = Layer(kind, dim, dim)

    def forward(self, x0, x1=None, m1=None, x2=None, m2=None):
        if self.depth == 0:
            return self.mlp(x0)
        h_t = self.l1(x0, x1, m1)                        # target after layer 1
        if self.depth == 2:
            h_mid = self.l1(x1, x2, m2)                  # each direct neighbour after layer 1
            h_t = self.l2(h_t, h_mid, m1)
        if self.skip == "concat":
            return torch.cat([self.own(x0), h_t], dim=-1)
        return h_t


def batch(graph, anchors, depth, fanout, rng, device, training, isolated="self"):
    n1, m1, n2, m2 = sample_tree(graph, anchors, depth, fanout, rng, isolated)
    x0 = graph.x[torch.from_numpy(anchors)].to(device)
    if depth == 0:
        return (x0,)
    x1 = gather(graph, anchors, n1, training).to(device)
    if depth == 1:
        return x0, x1, torch.from_numpy(m1).to(device)
    x2 = gather(graph, anchors, n2, training).to(device)
    return x0, x1, torch.from_numpy(m1).to(device), x2, torch.from_numpy(m2).to(device)


def train(graph, depth, kind, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = OCGNN(graph.x.shape[1], depth, kind, skip=args.skip).to(device)
    anchors = graph.train_anchors
    with torch.no_grad():  # fix the centre from the initial network
        z = torch.cat([model(*batch(graph, anchors[a:a + 2048], depth, args.fanout, rng, device, True, args.isolated))
                       for a in range(0, len(anchors), 2048)])
        c = z.mean(0)
        c[(c.abs() < 0.1) & (c < 0)] = -0.1
        c[(c.abs() < 0.1) & (c >= 0)] = 0.1
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for ep in range(args.epochs):
        perm = rng.permutation(anchors)
        tot = 0.0
        for a in range(0, len(perm), args.batch):
            b = perm[a:a + args.batch]
            z = model(*batch(graph, b, depth, args.fanout, rng, device, True, args.isolated))
            loss = ((z - c) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep == 0 or (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"      epoch {ep + 1:3d}/{args.epochs}  loss {tot / len(perm):.5f}", flush=True)
    return model, c


@torch.no_grad()
def score(model, c, graph, nodes, depth, seed, args, device):
    model.eval()
    rng = np.random.default_rng(10_000 + seed)
    rounds = 1 if depth == 0 else args.rounds
    out = np.zeros(len(nodes))
    for _ in range(rounds):
        for a in range(0, len(nodes), 4096):
            z = model(*batch(graph, nodes[a:a + 4096], depth, args.fanout, rng, device, False, args.isolated))
            out[a:a + 4096] += ((z - c) ** 2).sum(-1).cpu().numpy()
    model.train()
    return out / rounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="One-class GNN (Deep SVDD) under the baseline protocol")
    ap.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--depths", type=int, nargs="*", default=DEPTHS)
    ap.add_argument("--encoder", default="gin", choices=["gin", "gat", "tf"])
    ap.add_argument("--fanout", type=int, default=FANOUT)
    ap.add_argument("--isolated", default="self", choices=["self", "empty"])
    ap.add_argument("--skip", default="concat", choices=["concat", "none"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--context-features", default="users", choices=["none", "users"])
    ap.add_argument("--context-min-users", type=int, default=3)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.levels = sorted(args.levels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | encoder {args.encoder} | depths {args.depths} | context {args.context_features} "
          f"(min users {args.context_min_users}) | isolated: {args.isolated} | skip: {args.skip}", flush=True)
    missing = sorted({bundle_path(lv, s) for lv in args.levels for s in args.seeds
                      if not os.path.exists(bundle_path(lv, s))})
    if missing:
        raise SystemExit("missing bundles:\n  " + "\n  ".join(missing))

    ev = Evaluator(args.boot)
    rows, chosen, score_rows = [], [], []
    enc = {"gin": "GIN", "gat": "GAT", "tf": "Transformer"}[args.encoder]
    for level in args.levels:
        graphs, runs = {}, {}
        for seed in args.seeds:
            path = bundle_path(level, seed)
            if path not in graphs:
                b = load_bundle(path)
                graphs[path] = (ScoringGraph(b, args.context_features, args.context_min_users), b)
            graph, b = graphs[path]
            pos = {p: i for i, p in enumerate(graph.held_paths)}
            vq = graph.held_node[[pos[p] for p in ev.val_paths]]
            tq = graph.held_node[[pos[p] for p in ev.test_paths]]
            fit_paths = b["nodes"].loc[b["nodes"]["kind"] == "evaluated", "path"].values
            for depth in args.depths:
                print(f"  level {level:.0%} seed {seed} depth {depth}: training", flush=True)
                t0 = time.time()
                model, c = train(graph, depth, args.encoder, seed, args, device)
                sv = score(model, c, graph, vq, depth, seed, args, device)
                st = score(model, c, graph, tq, depth, seed, args, device)
                pauc = roc_auc_score(ev.y_val, sv, max_fpr=SELECT_MAX_FPR)
                print(f"    val pAUC(<=5% FPR) {pauc:.3f} | {time.time() - t0:.0f}s", flush=True)
                runs.setdefault(depth, []).append((sv, st, fit_paths))
                if seed == args.seeds[0]:
                    chosen.append({"level": level, "encoder": args.encoder, "depth": depth, "val_pauc": pauc})
                for which, pths, s_ in [("val", ev.val_paths, sv), ("test", ev.test_paths, st)]:
                    score_rows.append(pd.DataFrame({"level": level, "seed": seed, "depth": depth, "split": which,
                                                    "path": pths, "score": s_}))
        best = max([c_ for c_ in chosen if c_["level"] == level], key=lambda c_: c_["val_pauc"])["depth"]
        for c_ in chosen:
            if c_["level"] == level:
                c_["chosen"] = c_["depth"] == best
        print(f"  [level {level:.0%}] depth chosen on validation: {best}", flush=True)
        for depth in args.depths:
            name = "Deep SVDD (no graph)" if depth == 0 else f"OC-{enc}, depth {depth}"
            per_seed = runs[depth]
            rows.append(ev.row(level, name, [st for _, st, _ in per_seed], f"depth={depth}"))
            cal = [ev.calibrated(sv, st, fp) for sv, st, fp in per_seed]
            rows.append(ev.row(level, f"{name}, subgroup-calibrated", cal, f"depth={depth}"))

    results, chosen = pd.DataFrame(rows), pd.DataFrame(chosen)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/ocgnn_results{args.tag}.csv", index=False)
    chosen.to_csv(f"{RESULTS_DIR}/ocgnn_settings{args.tag}.csv", index=False)
    pd.concat(score_rows).to_csv(f"{RESULTS_DIR}/ocgnn_scores{args.tag}.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== One-class GNN test results (family-weighted) ===")
    print(results[["level", "method", "auc_fw", "auc_fw_lo", "auc_fw_hi", "tpr05_fw", "tpr1_fw_seed_mean",
                   "tpr1_fw_seed_std", "tpr2_fw", "auc_hard"]].round(3).to_string(index=False))
    sg_cols = [c_ for g in ["conn", "iso"] for c_ in
               (f"sg_auc_fw_{g}", f"sg_tpr1_fw_{g}", f"sg_tpr1glob_fw_{g}") if c_ in results.columns]
    print("\n=== Connectivity subgroups: ROC-AUC fw | caught @ 1% FPR within subgroup | at the global threshold ===")
    print(results[["level", "method"] + sg_cols].round(3).to_string(index=False))
    print("\n=== Validation (seed 0) ===")
    print(chosen.round(3).to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/ocgnn_*{args.tag}.csv")


if __name__ == "__main__":
    main()
