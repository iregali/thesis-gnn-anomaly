"""
complementarity_check.py
Does a graph-based score add information to the best feature detector?
(Methodology §26-§27, option C of §24.) Decisions are taken on VALIDATION; the
test split is only evaluated with --split test, once the decision is made.

For every saved one-class GNN configuration (ocgnn_scores{tag}.csv, written by
train_ocgnn.py) and every seed, the OCSVM (static+deps+names+GuardDog, the
setting chosen on validation for that level in baseline_protocol_settings)
is refitted on the same training packages (same contamination draw), and the
two scores are compared and combined:

  OCSVM          the feature detector alone
  model          the GNN (or depth-0 Deep SVDD) score alone
  avg50, avg25   rank average with weight 0.5 / 0.25 on the model
  conn_only      OCSVM for isolated packages; rank average (0.5) of OCSVM and
                 model for connected ones; both parts calibrated against
                 validation normals of their own subgroup (same function as
                 the baselines' calibrated rows)

Control: the depth-0 Deep SVDD (no graph) goes through exactly the same
combinations. A combination that helps only as much with depth 0 as with a GNN
is an ensembling effect, not a structural one. Structure adds information only
if a GNN combination beats BOTH the OCSVM alone AND the depth-0 combination,
above all on connected packages.

Metrics (family-weighted, mean over seeds): ROC-AUC and detection at 1% FPR,
overall and per subgroup (connected / isolated, relative to the training
normals, connectivity.py).

Usage
  python src/complementarity_check.py                      # validation, levels 0 and 0.01
  python src/complementarity_check.py --levels 0           # clean only
  python src/complementarity_check.py --split test         # only after the decision
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from baseline_protocol import (DATA_DIR, RESULTS_DIR, SPLIT_FILE, combine, fit_model, load_features,  # noqa: E402
                               preprocess, tpr_at_fpr)
from build_graph import draw_contamination  # noqa: E402
from connectivity import connectivity_table, load_core_deps  # noqa: E402

TAGS = ["_gin_all", "_ginmean_all", "_gat_all", "_tf_all", "_tconv_all", "_gps_all"]
FEATURES = "static+deps+names+GuardDog"


def ranks(s):
    return pd.Series(s).rank(pct=True).values


def metrics(s, y, w, mask):
    out = {}
    for g, m in mask.items():
        if (y[m] == 1).sum() < 10 or (y[m] == 0).sum() < 50:
            continue
        out[f"auc_{g}"] = roc_auc_score(y[m], s[m], sample_weight=w[m])
        out[f"tpr1_{g}"] = tpr_at_fpr(s[m], y[m], w[m], 0.01)[1]
    return out


def ocsvm_setting(level):
    for tag in ["_conn", ""]:
        p = f"{RESULTS_DIR}/baseline_protocol_settings{tag}.csv"
        if not os.path.exists(p):
            continue
        st = pd.read_csv(p)
        st = st[(st["level"].astype(float) == level) & (st["features"] == FEATURES) & (st["method"] == "OCSVM")]
        if "variant" in st.columns:
            st = st[st["variant"].fillna("naive") == "naive"]
        if len(st):
            return ast.literal_eval(st["setting"].iloc[0])
    raise SystemExit(f"no OCSVM setting for level {level} in baseline_protocol_settings*.csv")


def main():
    ap = argparse.ArgumentParser(description="Does a GNN score add information to the OCSVM?")
    ap.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.01])
    ap.add_argument("--tags", nargs="*", default=TAGS)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    feats, groups = load_features()
    y_all = feats["label"].values
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    w_all = feats["path"].map(dict(zip(splits["path"], splits["family_weight"].astype(float)))).values
    train_idx = np.where((role == "train") & (y_all == 0))[0]
    val_idx = np.where(role == "val")[0]
    ev_idx = val_idx if args.split == "val" else np.where(role == "test")[0]
    y, w = y_all[ev_idx], w_all[ev_idx]
    y_val = y_all[val_idx]
    ev_paths = feats["path"].values[ev_idx]
    deps_of = load_core_deps(DATA_DIR)
    conn_ev = connectivity_table(feats, train_idx, ev_idx, deps_of)["connected"].values
    conn_val = connectivity_table(feats, train_idx, val_idx, deps_of)["connected"].values
    mask = {"all": np.ones(len(ev_idx), bool), "conn": conn_ev, "iso": ~conn_ev}
    pool = feats.loc[role == "contam", "path"].tolist()
    path_to_idx = {p: i for i, p in enumerate(feats["path"].values)}
    cols = groups["static"] + groups["deps"] + groups["names"] + groups["guarddog"]
    X_raw = feats[cols].fillna(0).values.astype(float)
    print(f"{args.split}: {int((y == 0).sum())} normal + {int((y == 1).sum())} malicious | connected "
          f"{conn_ev.mean():.1%} (malicious {conn_ev[y == 1].mean():.1%})", flush=True)

    rows = []
    for level in args.levels:
        setting = {**ocsvm_setting(level), "trim": 0.0}
        print(f"\n[level {level:.0%}] OCSVM setting {setting}", flush=True)
        oc_cache = {}
        for tag in args.tags:
            p = f"{RESULTS_DIR}/ocgnn_scores{tag}.csv"
            if not os.path.exists(p):
                print(f"  (skip {tag}: no scores file)")
                continue
            sc = pd.read_csv(p, dtype={"path": str})
            sc = sc[sc["level"].astype(float) == level]
            for depth, d in sc.groupby("depth"):
                name = "Deep SVDD (depth 0)" if depth == 0 else f"{tag.strip('_').replace('_all', '')} d{depth}"
                per = []
                for seed, ds in d.groupby("seed"):
                    if seed not in oc_cache:   # OCSVM on the same training packages as this seed's bundle
                        drawn = draw_contamination(pool, level, len(train_idx), int(seed)) if level > 0 else []
                        fit_idx = np.concatenate([train_idx, np.array([path_to_idx[q] for q in drawn], dtype=int)])
                        Xm = preprocess(X_raw, fit_idx, setting["mode"])
                        mdl = fit_model("OCSVM", setting, Xm[fit_idx], int(seed))
                        oc_cache[seed] = (-mdl.decision_function(Xm[val_idx]), -mdl.decision_function(Xm[ev_idx]))
                    oc_v, oc_e = oc_cache[seed]
                    sv = ds[ds["split"] == "val"].set_index("path")["score"]
                    se = ds[ds["split"] == args.split].set_index("path")["score"]
                    g_v = sv.reindex(feats["path"].values[val_idx]).values
                    g_e = se.reindex(ev_paths).values
                    if np.isnan(g_e).any() or np.isnan(g_v).any():
                        raise SystemExit(f"{tag} depth {depth} seed {seed}: scores missing for some packages")
                    combos = {
                        "OCSVM": oc_e,
                        "model": g_e,
                        "avg50": 0.5 * ranks(oc_e) + 0.5 * ranks(g_e),
                        "avg25": 0.75 * ranks(oc_e) + 0.25 * ranks(g_e),
                    }
                    # connected: rank average of the two (ranks taken on validation + evaluation jointly,
                    # so validation calibration stays meaningful); isolated: OCSVM alone
                    both_v = 0.5 * ranks(np.r_[oc_v, oc_e])[:len(oc_v)] + 0.5 * ranks(np.r_[g_v, g_e])[:len(g_v)]
                    both_e = 0.5 * ranks(np.r_[oc_v, oc_e])[len(oc_v):] + 0.5 * ranks(np.r_[g_v, g_e])[len(g_v):]
                    combos["conn_only"] = combine(np.where(conn_val, both_v, np.nan), np.where(conn_ev, both_e, np.nan),
                                                  oc_v, oc_e, y_val)
                    per.append({k: metrics(s, y, w, mask) for k, s in combos.items()})
                for combo in per[0]:
                    m = pd.DataFrame([pp[combo] for pp in per]).mean()
                    rows.append({"level": level, "model": name, "combo": combo, **m.round(4).to_dict()})
                print(f"  {name}: done ({len(per)} seeds)", flush=True)

    res = pd.DataFrame(rows)
    res = res[~((res["combo"] == "OCSVM") & res.duplicated(["level", "combo"]))]   # OCSVM row once per level
    res.loc[res["combo"] == "OCSVM", "model"] = "-"
    out = f"{RESULTS_DIR}/complementarity_{args.split}.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols_show = ["level", "model", "combo"] + [c for c in ["auc_all", "tpr1_all", "auc_conn", "tpr1_conn",
                                                           "auc_iso", "tpr1_iso"] if c in res.columns]
    print(f"\n=== Complementarity on {args.split} (family-weighted, mean over seeds) ===")
    print(res[cols_show].round(3).to_string(index=False))
    print("\nReading: structure adds information only if a GNN combination beats BOTH 'OCSVM' and the same "
          "combination with 'Deep SVDD (depth 0)', above all on connected packages.")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
