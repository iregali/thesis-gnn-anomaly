"""
learning_curve_raw.py
Learning curve of the feature baselines (Methodology §33): the OCSVM and the
Isolation Forest on the raw features (static+deps+names+GuardDog), trained on a
fraction of the clean training normals. Exactly the same packages are kept as
in `train_contrastive.py --train-frac` (gnn_eval.train_subset), so the curves
can be put side by side. Evaluated like everything else (gnn_eval.Evaluator:
test, family-weighted, bootstrap, subgroups). CPU only.

Question: would MORE training data (a larger graph / sample) change the
conclusions? If the gap between the contrastive GNN, its no-graph control and
the raw-feature detectors stays the same from 10% to 100% of the training data,
scale is not what limits the graph models.

Usage
  python src/learning_curve_raw.py                      # fractions 0.1 0.25 0.5 1.0, seeds 0 1 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import RESULTS_DIR, fit_model, load_features, preprocess  # noqa: E402
from complementarity_check import ocsvm_setting  # noqa: E402
from gnn_eval import Evaluator, train_subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Raw-feature learning curve (clean training)")
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    ev = Evaluator(args.boot)
    f, train_idx = ev.feats, ev.train_idx
    _, g = load_features()
    cols = g["static"] + g["deps"] + g["names"] + g["guarddog"]
    X_raw = f[cols].fillna(0).values.astype(float)
    paths = f["path"].values
    oc_set = {**ocsvm_setting(0.0), "trim": 0.0}
    rows = []
    for frac in args.fractions:
        per = {"OCSVM": [], "Isolation Forest": []}
        cal = {"OCSVM": []}
        for seed in args.seeds:
            keep = train_subset(paths[train_idx], frac, seed)
            fit_idx = np.array([i for i in train_idx if paths[i] in keep])
            Xm = preprocess(X_raw, fit_idx, oc_set["mode"])
            oc = fit_model("OCSVM", oc_set, Xm[fit_idx], seed)
            sv, st = -oc.decision_function(Xm[ev.val_idx]), -oc.decision_function(Xm[ev.test_idx])
            per["OCSVM"].append(st)
            cal["OCSVM"].append(ev.calibrated(sv, st, paths[fit_idx]))
            iso = fit_model("IF", {"trim": 0.0}, X_raw[fit_idx], seed)
            per["Isolation Forest"].append(-iso.decision_function(X_raw[ev.test_idx]))
            print(f"  fraction {frac:.0%} seed {seed}: {len(fit_idx)} training packages", flush=True)
        for name, sc in per.items():
            r = ev.row(0.0, f"{name}, raw features", sc, f"train_frac={frac}")
            rows.append({"train_frac": frac, **r})
        r = ev.row(0.0, "OCSVM, raw features, subgroup-calibrated", cal["OCSVM"], f"train_frac={frac}")
        rows.append({"train_frac": frac, **r})

    res = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = f"{RESULTS_DIR}/learning_curve_raw.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print("\n=== Raw-feature learning curve (clean, test, family-weighted) ===")
    print(res[["train_frac", "method", "auc_fw", "tpr1_fw_seed_mean", "tpr1_fw_seed_std", "sg_auc_fw_conn",
               "sg_tpr1_fw_conn", "sg_auc_fw_iso"]].round(3).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
