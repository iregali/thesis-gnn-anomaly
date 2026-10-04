"""
dep_combination_check.py
Quick test (VALIDATION set only; the test set stays untouched): do measures of
how unusual a package's COMBINATION of dependencies is separate malicious from
normal packages, beyond what the best feature detector already sees?

Measures (label-blind; computed against the core dependency lists of the
training normal packages, the reference R; higher = more unusual)
  n_unknown        dependencies that no package in R uses
  rarest           -log(1 + number of R packages using the rarest dependency)
  support          -log(1 + number of R packages whose dependencies include ALL
                   of this package's dependencies)
  max_jaccard      -(highest Jaccard similarity between this package's
                   dependency set and any R package's set)
  unseen_pairs     share of this package's dependency pairs that never occur
                   together in any R package (needs >= 2 dependencies)
Validation packages are not in R, so no leave-one-out is needed here (it WOULD
be needed for training packages if these became model features).

Reported on connected validation packages (>= 1 dependency) and on those with
>= 2 dependencies: family-weighted ROC-AUC of each measure alone, of the OCSVM
(static+deps+names+GuardDog, the 0% setting), and of their rank average.

Usage
  python src/dep_combination_check.py
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from baseline_protocol import DATA_DIR, SPLIT_FILE, fit_model, load_features, preprocess  # noqa: E402
from connectivity import load_core_deps  # noqa: E402

OCSVM_SETTING = {"mode": "scale_all", "nu": 0.05, "g": 1.0, "trim": 0.0}  # chosen on validation at 0% (§21)
MAX_PAIRS_DEPS = 60  # cap for pair enumeration of very long dependency lists


def measures(ref_sets, query_sets):
    vocab = {}
    for d in ref_sets:
        for t in d:
            vocab.setdefault(t, len(vocab))
    df = np.zeros(len(vocab))
    users = {}
    for i, d in enumerate(ref_sets):
        for t in d:
            df[vocab[t]] += 1
            users.setdefault(t, set()).add(i)
    pairs = set()
    for d in ref_sets:
        dd = sorted(d)[:MAX_PAIRS_DEPS]
        pairs.update(combinations(dd, 2))
    rows, cols = [], []
    for i, d in enumerate(ref_sets):
        for t in d:
            rows.append(i)
            cols.append(vocab[t])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ref_sets), max(len(vocab), 1)))
    ref_size = np.array([len(d) for d in ref_sets])

    out = []
    for d in query_sets:
        known = [t for t in d if t in vocab]
        r = {"n_deps": len(d), "n_unknown": len(d) - len(known)}
        r["rarest"] = -np.log1p(min(df[vocab[t]] for t in known)) if known else 0.0
        if d:
            common = None
            for t in d:
                u = users.get(t, set())
                common = u if common is None else (common & u)
                if not common:
                    break
            r["support"] = -np.log1p(len(common) if common else 0)
        else:
            r["support"] = np.nan
        if known:
            q = np.zeros(A.shape[1])
            q[[vocab[t] for t in known]] = 1
            inter = A @ q
            jac = inter / (len(d) + ref_size - inter)
            r["max_jaccard"] = -float(jac.max())
        else:
            r["max_jaccard"] = 0.0 if d else np.nan
        if len(d) >= 2:
            dd = sorted(d)[:MAX_PAIRS_DEPS]
            ps = list(combinations(dd, 2))
            r["unseen_pairs"] = float(np.mean([p not in pairs for p in ps]))
        else:
            r["unseen_pairs"] = np.nan
        out.append(r)
    return pd.DataFrame(out)


def ranks(v):
    return pd.Series(v).rank(pct=True).values


def main():
    feats, groups = load_features()
    y_all = feats["label"].values
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
    w_all = feats["path"].map(dict(zip(splits["path"], splits["family_weight"].astype(float)))).values
    train_idx = np.where((role == "train") & (y_all == 0))[0]
    val_idx = np.where(role == "val")[0]
    deps_of = load_core_deps(DATA_DIR)

    ref = [deps_of.get(p, set()) for p in feats["path"].values[train_idx]]
    qry = [deps_of.get(p, set()) for p in feats["path"].values[val_idx]]
    m = measures(ref, qry)
    y, w = y_all[val_idx], w_all[val_idx]

    cols = groups["static"] + groups["deps"] + groups["names"] + groups["guarddog"]
    X = preprocess(feats[cols].fillna(0).values.astype(float), train_idx, OCSVM_SETTING["mode"])
    oc = fit_model("OCSVM", OCSVM_SETTING, X[train_idx], 0)
    m["ocsvm"] = -oc.decision_function(X[val_idx])

    print("=== Validation packages by number of core dependencies ===")
    nd = m["n_deps"].clip(upper=2).map({0: "0", 1: "1", 2: ">=2"})
    print(pd.crosstab(nd, np.where(y == 1, "malicious", "normal"), margins=True).to_string())
    for lab, cls in [(1, "malicious"), (0, "normal")]:
        sel = (y == lab) & (m["n_deps"] >= 1)
        print(f"  connected {cls}: {sel.sum()} packages, {int((sel & (m['n_deps'] >= 2)).sum())} with >= 2 dependencies")

    names = ["n_unknown", "rarest", "support", "max_jaccard", "unseen_pairs"]
    rows = []
    for subset, mask in [("connected (>=1 dep)", m["n_deps"].values >= 1), (">=2 deps", m["n_deps"].values >= 2)]:
        yy, ww = y[mask], w[mask]
        if len(set(yy)) < 2:
            continue
        base = roc_auc_score(yy, m["ocsvm"].values[mask], sample_weight=ww)
        r = {"subset": subset, "normal": int((yy == 0).sum()), "malicious": int((yy == 1).sum()),
             "OCSVM": round(base, 3)}
        for c in names:
            v = m[c].values[mask]
            ok = ~np.isnan(v)
            if len(set(yy[ok])) < 2:
                continue
            r[c] = round(roc_auc_score(yy[ok], v[ok], sample_weight=ww[ok]), 3)
            combo = (ranks(m["ocsvm"].values[mask][ok]) + ranks(v[ok])) / 2
            r[f"OCSVM+{c}"] = round(roc_auc_score(yy[ok], combo, sample_weight=ww[ok]), 3)
        rows.append(r)
    res = pd.DataFrame(rows).set_index("subset").T
    print("\n=== Family-weighted ROC-AUC on validation (measure alone | rank average with the OCSVM) ===")
    print(res.to_string())
    print("\nReading: a measure helps if 'OCSVM+measure' beats 'OCSVM' on the same subset.")


if __name__ == "__main__":
    main()
