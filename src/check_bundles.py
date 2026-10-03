"""
check_bundles.py
Sanity checks on the GNN graph bundles before any training (Methodology §13, §20).
Fails loudly (exit code 1) instead of passing silently when something is missing.

Checks per bundle
  1. exists          every expected (level, seed) bundle is on disk
  2. forbidden       no label / split / campaign / popularity column among the features
  3. parity_cols     node feature columns = the baseline feature columns (+ is_context)
  4. contamination   contamination share matches the level, and the contaminated
                     packages are EXACTLY the ones baseline_protocol.py draws
  5. heldout         no validation/test package is a node of the training graph;
                     held-out counts match the split file
  6. name_coverage   every evaluated / context name has a row in name_features.csv
                     (otherwise build_graph fills min_dist_to_popular with 0 =
                     "identical to a popular name", while the baselines use 3)

Usage
  python src/check_bundles.py
  python src/check_bundles.py --levels 0.01 0.05 --seeds 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from baseline_protocol import DATA_DIR, SPLIT_FILE, load_features  # noqa: E402
from build_graph import draw_contamination, load_bundle, n_contamination, pep503  # noqa: E402

FORBIDDEN = {"family_weight", "campaign_id", "split", "label", "is_popular",
             "scan_failed", "read_failed", "archive_type", "gd_flagged"}


def bundle_path(level, seed):
    return f"{DATA_DIR}/graph_c{round(level * 100):02d}_s{seed}.pt"


def main():
    ap = argparse.ArgumentParser(description="Sanity checks on the graph bundles")
    ap.add_argument("--levels", type=float, nargs="*", default=[0.01, 0.02, 0.05])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    # 1. existence first, so a missing build is never mistaken for a pass
    expected = [(lv, s, bundle_path(lv, s)) for lv in args.levels for s in args.seeds]
    missing = [p for _, _, p in expected if not os.path.exists(p)]
    if missing:
        print(f"[FAIL] exists: {len(missing)}/{len(expected)} bundles missing:")
        for p in missing:
            print(f"         {p}")
        print("       → run slurm/build_graph.sh (and check logs/build_graph_*.out)")
        sys.exit(1)
    print(f"[ok]   exists: all {len(expected)} bundles found")

    # baseline view of the same data
    feats, groups = load_features()
    base_cols = groups["static"] + groups["deps"] + groups["names"] + groups["guarddog"]
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    n_train = int(((role == "train") & (feats["label"] == 0)).sum())
    pool = feats.loc[role == "contam", "path"].tolist()
    n_heldout = int(role.isin(["val", "test"]).sum())
    names = pd.read_csv(f"{DATA_DIR}/name_features.csv", keep_default_na=False)
    known_names = set(names["name"].map(pep503))

    failures = 0
    rows = []
    for level, seed, path in expected:
        b = load_bundle(path)
        g, nodes, h = b["graph"], b["nodes"], b["heldout"]
        fn = b["feature_names"]
        ev = nodes["kind"] == "evaluated"
        res = {"bundle": os.path.basename(path), "nodes": g.num_nodes, "features": len(fn)}

        # 2. forbidden
        bad = sorted(set(fn) & FORBIDDEN)
        res["forbidden"] = ",".join(bad) or "none"

        # 3. parity of feature columns (order does not matter, content does)
        only_graph = sorted(set(fn) - set(base_cols) - {"is_context"})
        only_base = sorted(set(base_cols) - set(fn))
        res["parity_cols"] = "ok" if not (only_graph or only_base) else \
            f"graph-only {only_graph} | baseline-only {only_base}"

        # 4. contamination: share and identity
        contam = set(nodes.loc[nodes["is_contamination"], "path"])
        drawn = set(draw_contamination(pool, level, n_train, seed))
        share = len(contam) / max(int(ev.sum()), 1)
        res["n_contam"] = len(contam)
        res["contam_share"] = round(share, 4)
        ok_n = len(contam) == n_contamination(level, n_train)
        res["contamination"] = "ok" if (ok_n and contam == drawn) else \
            f"count {len(contam)} vs {n_contamination(level, n_train)}, " \
            f"{len(contam ^ drawn)} paths differ from baseline draw"

        # 5. held-out separation
        graph_paths = set(nodes.loc[ev, "path"])
        leaked = graph_paths & set(h["path"])
        res["heldout"] = "ok" if (not leaked and len(h) == n_heldout) else \
            f"{len(leaked)} held-out paths in graph; {len(h)} held-out vs {n_heldout} in split"

        # 6. name coverage (evaluated + context + held-out + unseen context names)
        all_names = set(nodes["name"].dropna()) | set(h["name"]) | set(b["context_names"])
        uncovered = sorted(all_names - known_names)
        res["name_coverage"] = "ok" if not uncovered else \
            f"{len(uncovered)} names without name features, e.g. {uncovered[:5]}"

        checks = ["forbidden", "parity_cols", "contamination", "heldout", "name_coverage"]
        status = [res[c] in ("ok", "none") for c in checks]
        failures += (not all(status))
        rows.append(res)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(out.to_string(index=False))

    if failures:
        print(f"\n[FAIL] {failures} bundle(s) with at least one failed check")
        sys.exit(1)
    print(f"\n[ok]   all checks passed for {len(rows)} bundles")


if __name__ == "__main__":
    main()
