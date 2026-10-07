"""
twins_vs_detector_check.py
Do near-copies help where the best feature detector FAILS? (Methodology §29.)
Analysis only, on VALIDATION: nothing here becomes a feature or an edge, and
campaign labels are only used to weight families.

artifact_twins_check.py showed that malware has near-identical-code twins
(code_near, estimated Jaccard >= 0.8) much more often than normal packages,
mostly inside large campaigns. That matters for detection only if the twins
sit where the One-Class SVM (static+deps+names+GuardDog, setting chosen on
validation for --level) misses malware. Here the OCSVM is fitted as in
complementarity_check.py, its threshold is the 99th percentile of the
validation normals (1% false alarms), and every validation malware package is
either CAUGHT or MISSED. Then, with the same density matching as the twin
check (malware: one release per name, subsampled to --fractions; all normals;
--reps repetitions; scenario no_largest drops the largest malware family):

  twin          has a near-copy in the pooled sample (any split, any class)
  flagged_twin  has a near-copy whose OCSVM score is above the threshold
  normal_twin   has a near-copy that is a normal package
  rescued       missed, but caught after the simplest use of the twins:
                score := max(own score, scores of its near-copies), with the
                threshold re-set on the validation normals' new scores (so
                false alarms stay at 1%)

Shares are reported per package and family-weighted (every malware family in
the evaluated subsample counts equally). Validation normals get the same
columns: a normal with a flagged twin is a false alarm that propagation would
create.

Reading
  missed malware has twins about as often as caught malware, and flagged twins
  -> near-copy edges could add detections the OCSVM cannot make (worth a
     shared-artifact graph, with a temporal protocol)
  twins concentrated among CAUGHT malware, or caught_after ~ caught_before
  -> the copies are where the detector already succeeds: redundant

Caveat: partners can come from any split and any time; a deployed system only
sees packages published earlier, so 'rescued' is an upper bound.

Needs the artifact cache written by artifact_twins_check.py (run it first).

Usage
  python src/twins_vs_detector_check.py                       # clean training, validation
  python src/twins_vs_detector_check.py --level 0.01 --seeds 0 1 2
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import DATA_DIR, RESULTS_DIR, SPLIT_FILE, fit_model, load_features, preprocess  # noqa: E402
from build_graph import draw_contamination  # noqa: E402
from complementarity_check import ocsvm_setting  # noqa: E402
from make_campaigns import near_duplicate_pairs  # noqa: E402

CACHE = f"{DATA_DIR}/artifacts_cache.pkl"
FPR = 0.01


def ocsvm_scores(feats, groups, role, level, seed):
    """OCSVM (as in complementarity_check) fitted on the training normals (+ contamination draw);
    returns scores for every row of feats."""
    y = feats["label"].values
    train_idx = np.where((role == "train") & (y == 0))[0]
    setting = {**ocsvm_setting(level), "trim": 0.0}
    pool = feats.loc[role == "contam", "path"].tolist()
    drawn = set(draw_contamination(pool, level, len(train_idx), seed)) if level > 0 else set()
    fit_idx = np.concatenate([train_idx, np.where(feats["path"].isin(drawn).values)[0]]).astype(int)
    cols = groups["static"] + groups["deps"] + groups["names"] + groups["guarddog"]
    X = preprocess(feats[cols].fillna(0).values.astype(float), fit_idx, setting["mode"])
    mdl = fit_model("OCSVM", setting, X[fit_idx], seed)
    print(f"[level {level:.0%}, seed {seed}] OCSVM {setting}, fitted on {len(fit_idx)} packages", flush=True)
    return -mdl.decision_function(X)


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w)) if np.sum(w) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description="Near-copies among malware the OCSVM misses vs catches")
    ap.add_argument("--level", type=float, default=0.0, help="contamination level of the OCSVM training data")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.02, 0.05, 1.0])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--near", type=float, default=0.8, help="near-duplicate threshold (as in the twin check)")
    args = ap.parse_args()

    feats, groups = load_features()
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role_of = dict(zip(splits["path"], splits["split"]))
    fam_of = dict(zip(splits["path"], splits["campaign_id"]))
    role = feats["path"].map(role_of).values

    # same package set as artifact_twins_check.py (so every archive is in the cache)
    shuf = feats.sample(frac=1.0, random_state=0)
    pkgs = pd.concat([shuf[shuf["label"] == 0], shuf[shuf["label"] == 1].drop_duplicates("Name")], ignore_index=True)
    if not os.path.exists(CACHE):
        raise SystemExit(f"{CACHE} not found: run src/artifact_twins_check.py first")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    paths = pkgs["path"].values
    missing = [p for p in paths if p not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} archives not in the cache (e.g. {missing[0]}): rerun artifact_twins_check.py")

    n = len(pkgs)
    label = pkgs["label"].values
    prole = pkgs["path"].map(role_of).fillna("").values
    family = np.array([(fam_of.get(p, "") or f"single:{p}") if l == 1 else "" for p, l in zip(paths, label)],
                      dtype=object)
    near_mh = {i: cache[p][1] for i, p in enumerate(paths) if cache[p][1] is not None}
    print(f"normal {int((label == 0).sum())} | malicious (one release per name) {int(label.sum())} | "
          f"near-duplicate matching over {len(near_mh)} packages (threshold {args.near})", flush=True)
    pairs = np.array(sorted(near_duplicate_pairs(near_mh, args.near)), dtype=int).reshape(-1, 2)
    print(f"[info] {len(pairs)} near-duplicate pairs", flush=True)

    all_mal = np.where(label == 1)[0]
    fam_sizes = pd.Series(family[all_mal]).value_counts()
    largest = fam_sizes.index[0]
    print(f"[info] largest malware family: {largest} ({fam_sizes.iloc[0]} of {len(all_mal)} names)", flush=True)
    val_nor = np.where((label == 0) & (prole == "val"))[0]
    scenarios = {"all": all_mal, "no_largest": all_mal[family[all_mal] != largest]}

    mal_rows, nor_rows = [], []
    for seed in args.seeds:
        s_all = pd.Series(ocsvm_scores(feats, groups, role, args.level, seed), index=feats["path"].values)
        s = s_all.reindex(paths).values
        thr = np.quantile(s[val_nor], 1 - FPR)
        val_mal_all = np.where((label == 1) & (prole == "val"))[0]
        print(f"  validation: {len(val_nor)} normal, {len(val_mal_all)} malicious names; "
              f"caught at {FPR:.0%} FPR: {(s[val_mal_all] > thr).mean():.1%} (per package, all names)", flush=True)
        for scen, idx_mal in scenarios.items():
            for frac in args.fractions:
                k = max(1, int(round(frac * len(idx_mal))))
                for rep in range(args.reps if frac < 1.0 else 1):
                    rng = np.random.default_rng(1000 + rep)
                    pick = rng.choice(idx_mal, size=k, replace=False) if frac < 1.0 else idx_mal
                    pool = label == 0
                    pool[pick] = True
                    ev = pick[prole[pick] == "val"]
                    if len(ev) == 0:
                        continue
                    p = pairs[pool[pairs[:, 0]] & pool[pairs[:, 1]]]
                    a, b = p[:, 0], p[:, 1]
                    twin = np.zeros(n, bool)
                    twin[a] = twin[b] = True
                    flag = s > thr
                    ftwin = np.zeros(n, bool)
                    ftwin[a[flag[b]]] = True
                    ftwin[b[flag[a]]] = True
                    ntwin = np.zeros(n, bool)
                    ntwin[a[label[b] == 0]] = True
                    ntwin[b[label[a] == 0]] = True
                    # simplest propagation: max over own score and near-copies' scores
                    s2 = s.copy()
                    np.maximum.at(s2, a, s[b])
                    np.maximum.at(s2, b, s[a])
                    thr2 = np.quantile(s2[val_nor], 1 - FPR)
                    fam_ev = pd.Series(family[ev])
                    w = (1.0 / fam_ev.map(fam_ev.value_counts())).values
                    for i, wi in zip(ev, w):
                        mal_rows.append({"seed": seed, "scenario": scen, "fraction": frac, "rep": rep, "w": wi,
                                         "caught": s[i] > thr, "twin": twin[i], "flagged_twin": ftwin[i],
                                         "normal_twin": ntwin[i], "caught_after": s2[i] > thr2})
                    nor_rows.append({"seed": seed, "scenario": scen, "fraction": frac, "rep": rep,
                                     "twin": twin[val_nor].mean(), "flagged_twin": ftwin[val_nor].mean(),
                                     "raised_above_thr": ((s2[val_nor] > thr) & ~flag[val_nor]).mean()})
                print(f"  seed {seed}, {scen}, fraction {frac}: done", flush=True)

    M, N = pd.DataFrame(mal_rows), pd.DataFrame(nor_rows)
    out = []
    for (scen, frac), d in M.groupby(["scenario", "fraction"]):
        c, m = d[d["caught"]], d[~d["caught"]]
        nr = N[(N["scenario"] == scen) & (N["fraction"] == frac)]
        out.append({
            "scenario": scen, "fraction": frac,
            "val_malicious_per_rep": len(d) / d.groupby(["seed", "rep"]).ngroups,
            "caught_fw": wmean(d["caught"], d["w"]), "caught_after_fw": wmean(d["caught_after"], d["w"]),
            "missed_twin": m["twin"].mean(), "missed_twin_fw": wmean(m["twin"], m["w"]),
            "caught_twin": c["twin"].mean(), "caught_twin_fw": wmean(c["twin"], c["w"]),
            "missed_flagged_twin_fw": wmean(m["flagged_twin"], m["w"]),
            "missed_normal_twin_fw": wmean(m["normal_twin"], m["w"]),
            "rescued_of_missed_fw": wmean(m["caught_after"], m["w"]),
            "lost_of_caught_fw": wmean(~c["caught_after"], c["w"]),
            "normal_twin": nr["twin"].mean(), "normal_flagged_twin": nr["flagged_twin"].mean(),
            "normal_raised_above_thr": nr["raised_above_thr"].mean()})
    res = pd.DataFrame(out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"_c{int(round(args.level * 100)):02d}"
    out_csv = f"{RESULTS_DIR}/twins_vs_detector{tag}.csv"
    res.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    print(f"\n=== Near-copies (code_near >= {args.near}) vs OCSVM decisions at {FPR:.0%} FPR, validation, "
          f"level {args.level:.0%}, seeds {args.seeds} ===")
    print("missed_* / caught_* = share of missed / caught validation malware with a near-copy in the pool; "
          "_fw = every family counts equally")
    print("rescued_of_missed_fw = missed malware caught after score := max(own, near-copies'), threshold re-set "
          "on validation normals (still 1% FPR); lost_of_caught_fw = the price on caught malware")
    print("fraction 1.0 = all malware, NOT density-matched (contrast only)")
    for scen in res["scenario"].unique():
        print(f"\n--- scenario: {scen} ---")
        r = res[res["scenario"] == scen].drop(columns="scenario").set_index("fraction")
        print(r.round(3).T.to_string())
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
