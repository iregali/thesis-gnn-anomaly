"""
build_graph.py
Builds the training graph for the GNN and everything needed to score held-out
packages by inserting them one at a time (Methodology §13, §18, §20).

Graph (homogeneous)
  - One node type: package.
      * evaluated nodes: releases with full features (one archive each)
      * context nodes: dependency targets that are not evaluated packages
        (never downloaded, never scored). Archive features are 0 (after
        scaling), name features are real, flag is_context = 1.
  - One edge type: core dependency ("A depends on B"), from the archives'
    own declarations (extract_dependencies.py). Stored directed in
    `dep_edge_index` and undirected in `edge_index` (what the encoder uses).
    Typosquat pairs are NOT edges: only name features (and, later, a lookup
    for the negative sampler).

Training graph (split file from make_campaign_splits.py --contam-share)
  - all TRAIN normal packages
  - plus unlabeled contamination: a random draw (by --seed) from the
    "contam" pool, so that malware makes up --contam of the evaluated nodes
    (1%, 2%, 5%)
  - plus the context nodes their dependencies point to
  A dependency on a name that is an evaluated node in the graph links to
  that node (one node per name: the first release by path), otherwise to the
  context node of that name.

Held-out packages (validation and test, both classes)
  Not in the graph. Their features and dependency names are saved so that
  insert_package() can add one package at a time: edges to the targets it
  depends on (new context nodes are created for unknown names), plus edges
  from graph nodes that already depend on its name.

Features (same columns for every node)
  static + GuardDog features (identical filtering to the baselines: load()
  from baseline_experiment.py), dependency counts (n_core, n_optional), name
  features (min_dist_to_popular, n_typosquat_targets, has_typosquat_edge),
  and is_context. `is_popular` is never a feature (§19).
  log1p + standardisation fitted on the evaluated nodes of the training graph
  only (normal + contamination, i.e. what is available without labels).

Labels are kept in the metadata tables for evaluation and analysis only;
they are not attributes of the graph.

Output: one bundle per (contamination level, seed), load with load_bundle().

Usage
  python src/build_graph.py --contam 0.05 --seed 0
  python src/build_graph.py --contam 0.01 --seed 2 --optional
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from baseline_experiment import DATA_DIR, DROP_COLS, GD_GROUPS, ID_COLS, load

NAME_FEATURES = ["min_dist_to_popular", "n_typosquat_targets", "has_typosquat_edge"]
DEP_FEATURES = ["n_core", "n_optional"]
GD_COLS = ["code_issue_count"] + GD_GROUPS
NOT_FEATURES = set(ID_COLS) | set(DROP_COLS) | {"label", "scan_failed"}


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_inputs(data_dir, splits_file):
    """Package features, split assignment, dependency edges, name features."""
    feats = pd.concat([load("malicious", 1), load("normal", 0)], ignore_index=True)
    splits = pd.read_csv(f"{data_dir}/{splits_file}", dtype={"Version": str})
    deps = pd.concat([pd.read_csv(f"{data_dir}/deps_{c}_edges.csv", dtype={"Version": str})
                      for c in ["malicious", "normal"]], ignore_index=True)
    dep_pkgs = pd.concat([pd.read_csv(f"{data_dir}/deps_{c}_packages.csv", dtype={"Version": str})
                          for c in ["malicious", "normal"]], ignore_index=True)
    names = pd.read_csv(f"{data_dir}/name_features.csv", keep_default_na=False)
    feats = feats.merge(dep_pkgs[["path"] + DEP_FEATURES].drop_duplicates("path"), on="path", how="left")
    # as in the baselines: failed GuardDog scans count as "no findings"
    failed = feats["scan_failed"].fillna(1) == 1
    feats.loc[failed, GD_COLS] = 0
    return feats, splits, deps, names


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class Scaler:
    """log1p on non-negative continuous columns, then standardise.
    Fitted on the given rows only (the evaluated nodes of the training graph)."""

    def fit(self, X):
        self.cont = ~np.all(np.isin(X, [0, 1]), axis=0)
        self.nonneg = self.cont & (X.min(axis=0) >= 0)
        Z = self._log(X)
        self.mean = np.where(self.cont, Z.mean(axis=0), 0.0)
        std = Z.std(axis=0)
        self.std = np.where(self.cont & (std > 0), std, 1.0)
        return self

    def _log(self, X):
        X = X.astype(np.float64).copy()
        X[:, self.nonneg] = np.log1p(np.clip(X[:, self.nonneg], 0, None))
        return X

    def transform(self, X):
        return ((self._log(X) - self.mean) / self.std).astype(np.float32)


def feature_columns(feats):
    archive = [c for c in feats.columns
               if c not in NOT_FEATURES and c not in DEP_FEATURES
               and pd.api.types.is_numeric_dtype(feats[c])]
    return archive + DEP_FEATURES


def name_feature_table(names):
    t = names.copy()
    t["name"] = t["name"].map(pep503)
    return t.drop_duplicates("name").set_index("name")[NAME_FEATURES].astype(float)


# ---------------------------------------------------------------------------
# Contamination draw
# ---------------------------------------------------------------------------
def n_contamination(level, n_normal):
    """Malicious packages needed so they make up `level` of the evaluated nodes."""
    return int(round(level * n_normal / (1 - level)))


def draw_contamination(pool_paths, level, n_normal, seed):
    n = n_contamination(level, n_normal)
    if n > len(pool_paths):
        raise SystemExit(f"[error] contamination {level:.0%} needs {n} packages, pool has {len(pool_paths)}")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(pool_paths), size=n, replace=False))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_bundle(feats, splits, deps, names, level, seed, optional=False):
    """Training graph + held-out packages, as a dict (see module docstring)."""
    feats = feats.copy()
    feats["name"] = feats["Name"].map(pep503)
    split_of = dict(zip(splits["path"], splits["split"]))
    feats["split"] = feats["path"].map(split_of)
    meta_cols = ["path", "Name", "Version", "name", "label", "split"]
    for col in ["campaign_id", "family_weight"]:
        if col in splits.columns:
            feats[col] = feats["path"].map(dict(zip(splits["path"], splits[col])))
            meta_cols.append(col)

    train_normal = feats[(feats["split"] == "train") & (feats["label"] == 0)]
    pool = feats.loc[feats["split"] == "contam", "path"]
    if level > 0 and pool.empty:
        raise SystemExit("[error] split file has no contamination pool; "
                         "run make_campaign_splits.py --contam-share 0.25")
    contam_paths = draw_contamination(pool, level, len(train_normal), seed) if level > 0 else []
    graph_pkgs = pd.concat([train_normal, feats[feats["path"].isin(contam_paths)]])
    graph_pkgs = graph_pkgs.sort_values("path").reset_index(drop=True)
    heldout = feats[feats["split"].isin(["val", "test"])].sort_values("path").reset_index(drop=True)

    # dependency lists per archive (core, optionally extras)
    d = deps if optional else deps[deps["optional"] == 0]
    deps_of = d.groupby("path")["dep"].apply(lambda s: sorted(set(s))).to_dict()

    # features
    cols = feature_columns(feats)
    name_tab = name_feature_table(names)
    zero_name = np.zeros(len(NAME_FEATURES))

    def name_feats(name_list):
        return np.stack([name_tab.loc[n].values if n in name_tab.index else zero_name
                         for n in name_list]) if name_list else np.zeros((0, len(NAME_FEATURES)))

    def raw(df):
        return np.hstack([df[cols].fillna(0).values.astype(np.float64),
                          name_feats(list(df["name"])), np.zeros((len(df), 1))])

    feature_names = cols + NAME_FEATURES + ["is_context"]
    archive_idx = np.arange(len(cols))
    scaler = Scaler().fit(raw(graph_pkgs))

    def context_x(name_list):
        X = np.zeros((len(name_list), len(feature_names)))
        X[:, len(cols):len(cols) + len(NAME_FEATURES)] = name_feats(name_list)
        Z = scaler.transform(X)
        Z[:, archive_idx] = 0.0          # not downloaded: neutral archive features
        Z[:, -1] = 1.0                   # is_context
        return Z

    # nodes: evaluated first, then context
    node_of_name = {}
    for i, n in enumerate(graph_pkgs["name"]):
        node_of_name.setdefault(n, i)
    targets = sorted({t for p in graph_pkgs["path"] for t in deps_of.get(p, [])})
    context_names = [t for t in targets if t not in node_of_name]
    for j, n in enumerate(context_names):
        node_of_name[n] = len(graph_pkgs) + j

    src, dst = [], []
    for i, p in enumerate(graph_pkgs["path"]):
        for t in deps_of.get(p, []):
            src.append(i)
            dst.append(node_of_name[t])
    dep_edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)

    x = np.vstack([scaler.transform(raw(graph_pkgs)), context_x(context_names)])
    num_nodes = x.shape[0]
    graph = Data(x=torch.from_numpy(x.astype(np.float32)),
                 edge_index=to_undirected(dep_edge_index, num_nodes=num_nodes),
                 dep_edge_index=dep_edge_index, num_nodes=num_nodes)
    graph.is_context = torch.zeros(num_nodes, dtype=torch.bool)
    graph.is_context[len(graph_pkgs):] = True

    nodes = pd.concat([
        graph_pkgs[meta_cols].assign(kind="evaluated"),
        pd.DataFrame({"name": context_names, "kind": "context"}),
    ], ignore_index=True)
    nodes["is_contamination"] = nodes["path"].isin(contam_paths)

    # held-out: features + dependency names; context rows for names not in the graph
    heldout_deps = [deps_of.get(p, []) for p in heldout["path"]]
    unseen = sorted({t for ts in heldout_deps for t in ts if t not in node_of_name})
    dependents = {}
    for s_, t_ in zip(src, dst):
        dependents.setdefault(int(t_), []).append(int(s_))

    return {
        "graph": graph,
        "nodes": nodes,
        "node_of_name": node_of_name,
        "dependents": dependents,
        "heldout": heldout[meta_cols].reset_index(drop=True),
        "x_heldout": torch.from_numpy(scaler.transform(raw(heldout))),
        "heldout_deps": heldout_deps,
        "context_names": unseen,
        "x_context": torch.from_numpy(context_x(unseen)),
        "feature_names": feature_names,
        "scaler": scaler,
        "config": {"contam": level, "seed": seed, "optional": optional,
                   "n_contamination": len(contam_paths)},
    }


def insert_package(bundle, i):
    """Graph with held-out package i added (and any new context nodes).
    Returns (Data, index of the inserted node). The bundle is not changed."""
    g = bundle["graph"]
    n0 = g.num_nodes
    new = n0
    x_rows = [bundle["x_heldout"][i:i + 1]]
    extra = {}
    ctx_row = {n: k for k, n in enumerate(bundle["context_names"])}

    def node(name):
        if name in bundle["node_of_name"]:
            return bundle["node_of_name"][name]
        if name not in extra:
            extra[name] = n0 + 1 + len(extra)
            k = ctx_row[name]
            x_rows.append(bundle["x_context"][k:k + 1])
        return extra[name]

    src = [new] * len(bundle["heldout_deps"][i])
    dst = [node(t) for t in bundle["heldout_deps"][i]]
    # graph nodes that already depend on this package's name
    name = bundle["heldout"]["name"].iloc[i]
    target = bundle["node_of_name"].get(name)
    if target is not None:
        for s_ in bundle["dependents"].get(target, []):
            src.append(s_)
            dst.append(new)

    num_nodes = n0 + len(x_rows)
    add = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    data = Data(x=torch.cat([g.x] + x_rows),
                edge_index=to_undirected(torch.cat([g.dep_edge_index, add], dim=1), num_nodes=num_nodes),
                num_nodes=num_nodes)
    data.is_context = torch.cat([g.is_context, torch.tensor([False] + [True] * len(extra))])
    return data, new


def load_bundle(path):
    return torch.load(path, weights_only=False)


def summary(bundle):
    g, nodes, h = bundle["graph"], bundle["nodes"], bundle["heldout"]
    ev = nodes["kind"] == "evaluated"
    deg = torch.bincount(g.dep_edge_index[0], minlength=g.num_nodes)[:int(ev.sum())]
    print("\n=== Training graph ===")
    print(f"nodes: {g.num_nodes} (evaluated {int(ev.sum())}, context {int((~ev).sum())}) | "
          f"dependency edges: {g.dep_edge_index.size(1)} | features: {g.num_features}")
    print(f"contamination: {int(nodes['is_contamination'].sum())} malicious packages "
          f"({nodes.loc[ev, 'is_contamination'].mean():.1%} of evaluated nodes)")
    print(f"evaluated nodes without dependencies: {(deg == 0).float().mean():.1%}")
    print("\n=== Held-out packages (inserted one at a time) ===")
    print(pd.crosstab(h["split"], h["label"].map({0: "normal", 1: "malicious"})))
    no_deps = np.mean([len(d) == 0 for d in bundle["heldout_deps"]])
    print(f"without dependencies: {no_deps:.1%} | new context names: {len(bundle['context_names'])}")


def main():
    ap = argparse.ArgumentParser(description="Training graph + held-out packages for the GNN")
    ap.add_argument("--splits", default="splits_campaign_v2.csv")
    ap.add_argument("--contam", type=float, default=0.05, help="malicious share of evaluated nodes")
    ap.add_argument("--seed", type=int, default=0, help="contamination draw")
    ap.add_argument("--optional", action="store_true", help="also use optional (extras) dependencies")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feats, splits, deps, names = load_inputs(DATA_DIR, args.splits)
    bundle = build_bundle(feats, splits, deps, names, args.contam, args.seed, args.optional)
    out = args.out or f"{DATA_DIR}/graph_c{round(args.contam * 100):02d}_s{args.seed}.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(bundle, out)
    summary(bundle)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
