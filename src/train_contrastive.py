"""
train_contrastive.py
The contrastive framework of the thesis (Chapter 3, §3.4-3.6): a GNN encoder is
trained without labels with one of the existing contrastive losses analysed in
§3.5.4-3.5.7, then frozen; its embeddings are scored by the classical
detectors of §3.6 (Isolation Forest, LOF, One-Class SVM) fitted on the
training packages. Evaluated exactly like the baselines, CoLA and the
one-class GNN (gnn_eval.Evaluator). Methodology §30.

  encoder (--encoder: gin, gat, tf = Graph Transformer; also tconv, gps)
    x  loss (--loss: infonce, triplet, dgi)
    -> frozen embeddings  -> IF / LOF / OCSVM  (+ the DGI discriminator itself)

Encoders, neighbourhood sampling, context-node features and the own-feature
skip are those of train_ocgnn.py (same bundles, same flags), so the only thing
that changes between the two scripts is the training objective and the scorer.
Depth 0 (own features only, MLP) is the structure-free control: the same loss
without the graph.

Views (InfoNCE, triplet; §3.5.1-3.5.2). Each training package is encoded twice,
each time with
  - an independently sampled neighbourhood (fanout sampling, as CoLA's
    stochastic subgraphs),
  - feature masking: each feature column zeroed with probability --mask-p
    (one mask per view, shared by all nodes of the batch, as GRACE),
  - edge dropping: each sampled neighbour slot replaced by its parent with
    probability --drop-p (the same convention as packages without
    neighbours, --isolated self).
For an isolated package (most malware) only the feature mask differs between
its views: the loss then learns from its own features alone. This is reported,
not hidden (Methodology §30).

Losses (all on a 2-layer projection head for InfoNCE / triplet; the embedding
used for scoring is the encoder output before the head)
  infonce  NT-Xent / GRACE (Oord et al. 2018; Zhu et al. 2020): the two views of
           a package are the positive pair, all other packages of the batch in
           both views are negatives; cosine similarity, temperature --tau.
  triplet  (Schroff et al. 2015) anchor = view 1, positive = view 2 of the same
           package, negative = view 2 of another package: random (--mining
           random, as eq. 12 in §3.5.6) or the hardest one in the batch
           (--mining hard); L2 distance on normalised projections, margin
           --margin.
  dgi      Deep Graph Infomax (Velickovic et al. 2019): bilinear discriminator
           between a package embedding and the summary s = sigmoid(mean of the
           batch's embeddings); negatives = the same packages encoded with
           CORRUPTED features (every node's feature row replaced by that of a
           random training node, structure kept: row-wise shuffling).

Scoring (§3.6). Embeddings are averaged over --rounds neighbourhood samples
(no augmentation). Training packages are embedded with leave-one-out context
profiles (as in training), held-out packages without. The detectors are fitted
on the embeddings of the evaluated TRAINING packages (contamination included,
as for the baselines); embeddings are standardised for LOF and OCSVM on the
same packages. Each detector's setting (LOF k, OCSVM nu / gamma; same grids as
the baselines) is chosen on validation (pAUC <= 5% FPR, first seed) per depth
and reused for the other seeds. For DGI the discriminator's own score
(1 - agreement with the training summary) is also reported (scorer "DGI-D").
Plain and subgroup-calibrated rows, as for every other method.

Outputs (RESULTS_DIR): contrastive_results{tag}.csv, contrastive_settings{tag}.csv
(validation pAUC per depth / scorer / setting, chosen row marked),
contrastive_scores{tag}.csv (level, seed, depth, scorer, split, path, score).
Default tag: _{loss}_{encoder}.

Usage
  python src/train_contrastive.py --loss dgi --encoder gin --rescale all --agg mean
  python src/train_contrastive.py --loss infonce --encoder gat --rescale all
  python src/train_contrastive.py --loss triplet --levels 0 --seeds 0 --epochs 2 --rounds 2 --boot 20  # quick test
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import LOF_NEIGHBORS, OCSVM_GAMMA, OCSVM_NU, RESULTS_DIR, SELECT_MAX_FPR  # noqa: E402
from build_graph import load_bundle  # noqa: E402
from gnn_eval import Evaluator  # noqa: E402
from train_cola import ScoringGraph, bundle_path  # noqa: E402
from train_ocgnn import DIM, FANOUT, OCGNN, gather, memory, sample_tree  # noqa: E402

LEVELS = [0.0, 0.01]
SEEDS = [0, 1, 2]
DEPTHS = [0, 1, 2]
SCORERS = ["IF", "LOF", "OCSVM"]


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def view(graph, anchors, depth, rng, device, args, training=True, feat_mask=None, drop_p=0.0, corrupt=False):
    """Model inputs for one view of the anchors. feat_mask [F] (1 = keep) or None;
    drop_p = neighbour-slot dropping; corrupt = DGI row-wise feature shuffling."""
    n1, m1, n2, m2 = sample_tree(graph, anchors, depth, args.fanout, rng, args.isolated)
    if depth >= 1 and drop_p > 0:
        d1 = rng.random(n1.shape) < drop_p
        n1 = np.where(d1, anchors[:, None], n1)
        if args.isolated == "empty":
            m1 = m1 & ~d1
        if depth == 2:
            d2 = rng.random(n2.shape) < drop_p
            n2 = np.where(d2, n1[:, :, None], n2)
            if args.isolated == "empty":
                m2 = m2 & ~d2
    if corrupt:
        # every node's feature row replaced by that of a random training-graph node (structure kept);
        # context profiles are not recomputed (they belong to other nodes anyway)
        perm = rng.integers(0, graph.n_train, size=graph.x.shape[0])
        x0 = graph.x[torch.from_numpy(perm[anchors])]
        x1 = graph.x[torch.from_numpy(perm[n1])] if depth >= 1 else None
        x2 = graph.x[torch.from_numpy(perm[n2])] if depth == 2 else None
    else:
        x0 = graph.x[torch.from_numpy(anchors)]
        x1 = gather(graph, anchors, n1, training) if depth >= 1 else None
        x2 = gather(graph, anchors, n2, training) if depth == 2 else None
    if feat_mask is not None:
        x0 = x0 * feat_mask
        x1 = x1 * feat_mask if x1 is not None else None
        x2 = x2 * feat_mask if x2 is not None else None
    x0 = x0.to(device)
    if depth == 0:
        return (x0,)
    mem = memory(graph, anchors, depth, args.fanout, rng, device, args.isolated, args.mem) if args.mem else None
    m1t = torch.from_numpy(m1).to(device)
    if depth == 1:
        return x0, x1.to(device), m1t, None, None, mem
    return x0, x1.to(device), m1t, x2.to(device), torch.from_numpy(m2).to(device), mem


def feature_mask(n_feat, p, rng):
    return torch.from_numpy((rng.random(n_feat) >= p).astype(np.float32))


# ---------------------------------------------------------------------------
# Model: encoder (from train_ocgnn) + loss-specific head
# ---------------------------------------------------------------------------
class Contrastive(nn.Module):
    def __init__(self, in_dim, depth, args):
        super().__init__()
        self.enc = OCGNN(in_dim, depth, args.encoder, skip=args.skip, agg=args.agg)
        out = DIM * 2 if (depth > 0 and args.skip == "concat") else DIM
        self.out_dim = out
        if args.loss in ("infonce", "triplet"):
            self.proj = nn.Sequential(nn.Linear(out, DIM), nn.ELU(), nn.Linear(DIM, DIM))
        else:
            self.W = nn.Parameter(torch.empty(out, out))
            nn.init.xavier_uniform_(self.W)

    def forward(self, *inp):
        return self.enc(*inp)

    def disc(self, h, s):
        """DGI bilinear discriminator logits: h [B, d], s [d]."""
        return h @ (self.W @ s)


def infonce(z1, z2, tau):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    n = len(z1)
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def triplet(z1, z2, margin, mining, rng):
    a, p = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    n = len(a)
    d_pos = (a - p).norm(dim=-1)
    if mining == "hard":
        d = torch.cdist(a, p)
        d = d + torch.eye(n, device=a.device) * 1e6
        d_neg = d.min(dim=1).values
    else:
        shift = int(rng.integers(1, n)) if n > 1 else 0
        d_neg = (a - p.roll(shift, dims=0)).norm(dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


def train(graph, depth, seed, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Contrastive(graph.x.shape[1], depth, args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    anchors = graph.train_anchors
    n_feat = graph.x.shape[1]
    for ep in range(args.epochs):
        perm = rng.permutation(anchors)
        tot = 0.0
        for a in range(0, len(perm), args.batch):
            b = perm[a:a + args.batch]
            if len(b) < 2:
                continue
            if args.loss == "dgi":
                h = model(*view(graph, b, depth, rng, device, args))
                h_c = model(*view(graph, b, depth, rng, device, args, corrupt=True))
                s = torch.sigmoid(h.mean(0))
                logits = torch.cat([model.disc(h, s), model.disc(h_c, s)])
                target = torch.cat([torch.ones(len(b)), torch.zeros(len(b))]).to(device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            else:
                z1 = model.proj(model(*view(graph, b, depth, rng, device, args,
                                            feat_mask=feature_mask(n_feat, args.mask_p, rng), drop_p=args.drop_p)))
                z2 = model.proj(model(*view(graph, b, depth, rng, device, args,
                                            feat_mask=feature_mask(n_feat, args.mask_p, rng), drop_p=args.drop_p)))
                loss = infonce(z1, z2, args.tau) if args.loss == "infonce" else \
                    triplet(z1, z2, args.margin, args.mining, rng)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep == 0 or (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"      epoch {ep + 1:3d}/{args.epochs}  loss {tot / len(perm):.5f}", flush=True)
    return model


@torch.no_grad()
def embed(model, graph, nodes, depth, seed, args, device, training):
    """Embeddings averaged over sampling rounds (no augmentation)."""
    model.eval()
    rng = np.random.default_rng(10_000 + seed)
    rounds = 1 if depth == 0 else args.rounds
    out = np.zeros((len(nodes), model.out_dim))
    for _ in range(rounds):
        for a in range(0, len(nodes), 4096):
            h = model(*view(graph, nodes[a:a + 4096], depth, rng, device, args, training=training))
            out[a:a + 4096] += h.cpu().numpy()
    model.train()
    return out / rounds


# ---------------------------------------------------------------------------
# Scorers on frozen embeddings (§3.6)
# ---------------------------------------------------------------------------
def scorer_grid(kind):
    if kind == "IF":
        return [{}]
    if kind == "LOF":
        return [{"k": k} for k in LOF_NEIGHBORS]
    return [{"nu": nu, "g": g} for nu in OCSVM_NU for g in OCSVM_GAMMA]


def fit_scorer(kind, setting, E_fit, seed):
    if kind == "IF":
        return None, IsolationForest(n_estimators=300, random_state=seed, n_jobs=2).fit(E_fit)
    sc = StandardScaler().fit(E_fit)
    Z = sc.transform(E_fit)
    if kind == "LOF":
        return sc, LocalOutlierFactor(n_neighbors=setting["k"], novelty=True).fit(Z)
    return sc, OneClassSVM(kernel="rbf", gamma=setting["g"] / Z.shape[1], nu=setting["nu"]).fit(Z)


def apply_scorer(fitted, E):
    sc, mdl = fitted
    return -mdl.decision_function(sc.transform(E) if sc is not None else E)


def dgi_scores(model, E_fit, E, device):
    with torch.no_grad():
        s = torch.sigmoid(torch.from_numpy(E_fit).float().to(device).mean(0))
        return -model.disc(torch.from_numpy(E).float().to(device), s).cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Contrastive GNN encoder + classical detectors (thesis §3.4-3.6)")
    ap.add_argument("--loss", required=True, choices=["infonce", "triplet", "dgi"])
    ap.add_argument("--encoder", default="gin", choices=["gin", "gat", "tf", "tconv", "gps"])
    ap.add_argument("--levels", type=float, nargs="*", default=LEVELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--depths", type=int, nargs="*", default=DEPTHS)
    ap.add_argument("--gps-memory", type=int, default=512)
    ap.add_argument("--fanout", type=int, default=FANOUT)
    ap.add_argument("--isolated", default="self", choices=["self", "empty"])
    ap.add_argument("--skip", default="concat", choices=["concat", "none"])
    ap.add_argument("--rescale", default="all", choices=["bundle", "all"])
    ap.add_argument("--agg", default="mean", choices=["sum", "mean"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=0.5, help="InfoNCE temperature")
    ap.add_argument("--margin", type=float, default=0.5, help="triplet margin")
    ap.add_argument("--mining", default="random", choices=["random", "hard"], help="triplet negatives")
    ap.add_argument("--mask-p", type=float, default=0.2, help="feature masking probability (views)")
    ap.add_argument("--drop-p", type=float, default=0.2, help="neighbour dropping probability (views)")
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--context-features", default="users", choices=["none", "users"])
    ap.add_argument("--context-min-users", type=int, default=3)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    args.levels = sorted(args.levels)
    args.mem = args.gps_memory if args.encoder == "gps" else 0
    tag = args.tag if args.tag is not None else f"_{args.loss}_{args.encoder}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | loss {args.loss} | encoder {args.encoder} | depths {args.depths} | rescale "
          f"{args.rescale} | agg {args.agg} | context {args.context_features} | views: mask {args.mask_p}, "
          f"drop {args.drop_p}", flush=True)
    missing = sorted({bundle_path(lv, s) for lv in args.levels for s in args.seeds
                      if not os.path.exists(bundle_path(lv, s))})
    if missing:
        raise SystemExit("missing bundles:\n  " + "\n  ".join(missing))

    ev = Evaluator(args.boot)
    enc = {"gin": "GIN", "gat": "GAT", "tf": "Graph Transformer", "tconv": "TransformerConv",
           "gps": "GraphGPS"}[args.encoder]
    lname = {"infonce": "InfoNCE", "triplet": "Triplet", "dgi": "DGI"}[args.loss]
    scorers = SCORERS + (["DGI-D"] if args.loss == "dgi" else [])
    rows, settings, score_rows = [], [], []
    for level in args.levels:
        graphs, runs, chosen_setting = {}, {}, {}
        for seed in args.seeds:
            path = bundle_path(level, seed)
            if path not in graphs:
                b = load_bundle(path)
                graphs[path] = (ScoringGraph(b, args.context_features, args.context_min_users, args.rescale), b)
            graph, b = graphs[path]
            pos = {p: i for i, p in enumerate(graph.held_paths)}
            vq = graph.held_node[[pos[p] for p in ev.val_paths]]
            tq = graph.held_node[[pos[p] for p in ev.test_paths]]
            fit_paths = b["nodes"].loc[b["nodes"]["kind"] == "evaluated", "path"].values
            for depth in args.depths:
                print(f"  level {level:.0%} seed {seed} depth {depth}: training", flush=True)
                t0 = time.time()
                model = train(graph, depth, seed, args, device)
                E_fit = embed(model, graph, graph.train_anchors, depth, seed, args, device, True)
                E_v = embed(model, graph, vq, depth, seed, args, device, False)
                E_t = embed(model, graph, tq, depth, seed, args, device, False)
                for kind in scorers:
                    key = (depth, kind)
                    if kind == "DGI-D":
                        sv, st = dgi_scores(model, E_fit, E_v, device), dgi_scores(model, E_fit, E_t, device)
                        setting = {}
                        pauc = roc_auc_score(ev.y_val, sv, max_fpr=SELECT_MAX_FPR)
                    elif key not in chosen_setting:   # first seed: choose the setting on validation
                        best = None
                        for cand in scorer_grid(kind):
                            fitted = fit_scorer(kind, cand, E_fit, seed)
                            sv_c = apply_scorer(fitted, E_v)
                            p = roc_auc_score(ev.y_val, sv_c, max_fpr=SELECT_MAX_FPR)
                            settings.append({"level": level, "depth": depth, "scorer": kind, "setting": str(cand),
                                             "val_pauc": p})
                            if best is None or p > best[0]:
                                best = (p, cand, fitted, sv_c)
                        pauc, setting, fitted, sv = best
                        chosen_setting[key] = setting
                        st = apply_scorer(fitted, E_t)
                    else:
                        setting = chosen_setting[key]
                        fitted = fit_scorer(kind, setting, E_fit, seed)
                        sv, st = apply_scorer(fitted, E_v), apply_scorer(fitted, E_t)
                        pauc = roc_auc_score(ev.y_val, sv, max_fpr=SELECT_MAX_FPR)
                    if kind == "DGI-D" and seed == args.seeds[0]:
                        settings.append({"level": level, "depth": depth, "scorer": kind, "setting": "{}",
                                         "val_pauc": pauc})
                    runs.setdefault(key, []).append((sv, st, fit_paths))
                    print(f"    {kind:6s} val pAUC(<=5% FPR) {pauc:.3f}  {setting}", flush=True)
                    for which, pths, s_ in [("val", ev.val_paths, sv), ("test", ev.test_paths, st)]:
                        score_rows.append(pd.DataFrame({"level": level, "seed": seed, "depth": depth, "scorer": kind,
                                                        "split": which, "path": pths, "score": s_}))
                print(f"    {time.time() - t0:.0f}s", flush=True)
        for (depth, kind), per_seed in runs.items():
            name = f"{lname} + {kind}, " + ("no graph (MLP)" if depth == 0 else f"{enc} depth {depth}")
            st_str = f"depth={depth}; {kind} {chosen_setting.get((depth, kind), {})}"
            rows.append(ev.row(level, name, [st for _, st, _ in per_seed], st_str))
            cal = [ev.calibrated(sv, st, fp) for sv, st, fp in per_seed]
            rows.append(ev.row(level, f"{name}, subgroup-calibrated", cal, st_str))

    results, settings = pd.DataFrame(rows), pd.DataFrame(settings)
    # chosen = best validation pAUC per level over depth x scorer (selected settings only)
    best_rows = settings.loc[settings.groupby(["level", "depth", "scorer"])["val_pauc"].idxmax()]
    settings["selected_setting"] = settings.index.isin(best_rows.index)
    top = best_rows.loc[best_rows.groupby("level")["val_pauc"].idxmax()]
    settings["chosen"] = settings.index.isin(top.index)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/contrastive_results{tag}.csv", index=False)
    settings.to_csv(f"{RESULTS_DIR}/contrastive_settings{tag}.csv", index=False)
    pd.concat(score_rows).to_csv(f"{RESULTS_DIR}/contrastive_scores{tag}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n=== {lname} + {enc}: test results (family-weighted) ===")
    print(results[["level", "method", "auc_fw", "auc_fw_lo", "auc_fw_hi", "tpr05_fw", "tpr1_fw_seed_mean",
                   "tpr1_fw_seed_std", "tpr2_fw", "auc_hard"]].round(3).to_string(index=False))
    sg_cols = [c_ for g in ["conn", "iso"] for c_ in
               (f"sg_auc_fw_{g}", f"sg_tpr1_fw_{g}", f"sg_tpr1glob_fw_{g}") if c_ in results.columns]
    print("\n=== Connectivity subgroups: ROC-AUC fw | caught @ 1% FPR within subgroup | at the global threshold ===")
    print(results[["level", "method"] + sg_cols].round(3).to_string(index=False))
    print("\n=== Validation (first seed): selected setting per depth x scorer; * = chosen per level ===")
    sel = settings[settings["selected_setting"]].copy()
    sel["chosen"] = np.where(sel["chosen"], "*", "")
    print(sel.drop(columns="selected_setting").round(3).to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}/contrastive_*{tag}.csv")


if __name__ == "__main__":
    main()
