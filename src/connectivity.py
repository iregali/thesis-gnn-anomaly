"""
connectivity.py
One label-blind definition of graph connectivity and dependency peers, shared
by baseline_protocol.py (subgroup metrics, peer-deviation detector), the GNN
evaluation and export (heldout_connectivity.csv). No torch needed.

Graph (as in build_graph.py): a package P has an edge to every name in its
CORE dependencies D(P); a training package Q that depends on P's name gives P
an incoming edge. Write S(P) = D(P) + {name(P)}. Then two packages are
neighbours or share a neighbour exactly when S(P) and S(Q) intersect:
  - both depend on the same name            (co-dependents, 2 hops)
  - one depends on the other's name         (direct dependency, 1 hop)

Definitions for a held-out package P, relative to a reference set R
(the training normal packages, the same for every contamination level):
  out_deg    |D(P)|
  in_deg     number of packages in R that depend on name(P)
  connected  out_deg + in_deg > 0           (= not isolated after insertion)
  n_peers    number of packages in R whose S intersects S(P)
  has_peers  n_peers > 0
A package can be connected without peers: it depends only on names that no
training package uses (its neighbours are context nodes with no features).

Usage (export for the record; baseline_protocol computes the same on the fly)
  python src/connectivity.py
"""

import re

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = "/home/igalimi1/thesis/data"


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def load_core_deps(data_dir=DATA_DIR):
    """path -> set of core dependency names (PEP 503), both classes.
    Names are read literally (PyPI has packages called "null", "nan", ...)."""
    parts = []
    for cls in ["malicious", "normal"]:
        d = pd.read_csv(f"{data_dir}/deps_{cls}_edges.csv", dtype=str, keep_default_na=False,
                        usecols=["path", "dep", "optional"])
        parts.append(d[(d["optional"] == "0") & (d["dep"] != "")])
    d = pd.concat(parts, ignore_index=True)
    return d.groupby("path")["dep"].apply(lambda s: set(s)).to_dict()


class PeerIndex:
    """Sparse incidence of S(P) for a set of packages, for fast peer queries."""

    def __init__(self, paths, names, deps_of):
        self.vocab = {}
        rows, cols = [], []
        for i, (p, n) in enumerate(zip(paths, names)):
            for t in self.node_set(p, n, deps_of):
                rows.append(i)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
        self.A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(paths), max(len(self.vocab), 1)))
        # dependents: how many packages in the set depend on each name
        self.n_dependents = {}
        for p in paths:
            for t in deps_of.get(p, ()):
                self.n_dependents[t] = self.n_dependents.get(t, 0) + 1

    @staticmethod
    def node_set(path, name, deps_of):
        return set(deps_of.get(path, ())) | {pep503(name)}

    def query(self, paths, names, deps_of):
        """Sparse [len(paths), len(set)] matrix: > 0 where the packages are peers."""
        rows, cols = [], []
        for i, (p, n) in enumerate(zip(paths, names)):
            for t in self.node_set(p, n, deps_of):
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        Q = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(paths), self.A.shape[1]))
        return (Q @ self.A.T).tocsr()


def connectivity_table(feats, ref_idx, query_idx, deps_of):
    """Connectivity of the packages feats.iloc[query_idx] relative to the
    reference packages feats.iloc[ref_idx]. Returns a DataFrame (query order)."""
    ref = PeerIndex(feats["path"].values[ref_idx], feats["Name"].values[ref_idx], deps_of)
    q_paths, q_names = feats["path"].values[query_idx], feats["Name"].values[query_idx]
    M = ref.query(q_paths, q_names, deps_of)
    # a package never counts as its own peer (only relevant if query and ref overlap)
    pos = {p: j for j, p in enumerate(feats["path"].values[ref_idx])}
    n_peers = np.asarray((M > 0).sum(axis=1)).ravel()
    for i, p in enumerate(q_paths):
        if p in pos and M[i, pos[p]] > 0:
            n_peers[i] -= 1
    out_deg = np.array([len(deps_of.get(p, ())) for p in q_paths])
    in_deg = np.array([ref.n_dependents.get(pep503(n), 0) for n in q_names])
    return pd.DataFrame({"path": q_paths, "out_deg": out_deg, "in_deg": in_deg,
                         "connected": (out_deg + in_deg) > 0, "n_peers": n_peers,
                         "has_peers": n_peers > 0})


def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from baseline_protocol import SPLIT_FILE, load_features  # noqa: E402

    feats, _ = load_features()
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    y = feats["label"].values
    ref_idx = np.where((role == "train") & (y == 0))[0]
    q_idx = np.where(role.isin(["val", "test"]))[0]
    deps_of = load_core_deps()
    t = connectivity_table(feats, ref_idx, q_idx, deps_of)
    t["split"], t["label"] = role.values[q_idx], y[q_idx]
    t.to_csv(f"{DATA_DIR}/heldout_connectivity.csv", index=False)
    print(t.groupby(["split", "label"])[["connected", "has_peers"]].mean().round(3))
    print(f"median peers (connected test packages): "
          f"{t.loc[t['connected'] & (t['split'] == 'test'), 'n_peers'].median():.0f}")
    print(f"Saved {DATA_DIR}/heldout_connectivity.csv")


if __name__ == "__main__":
    main()
