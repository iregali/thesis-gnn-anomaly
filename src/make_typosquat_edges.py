"""
make_typosquat_edges.py
Creates typosquat edges: a link from a package name to a POPULAR package
name it closely resembles (e.g. `reqeusts` -> `requests`). Also writes
name-based features for every node.

Label-blind and parity-safe: only package names are used, computed the same
way for every node (malicious, normal, and dependency targets), without
looking at labels.

Nodes considered: all evaluation packages (static_*.csv) and all dependency
targets (deps_*_edges.csv), by normalized name (PEP 503).

Reference: the most-downloaded PyPI packages (hugovk/top-pypi-packages, 30
days), top N. Popular packages are targets, never sources. Note: the list
is recent, while the malware spans 2021-2026; the most popular packages are
stable over such periods, but this is a limitation to mention.

Rules (each edge records which rule produced it):
  edit       Damerau-Levenshtein distance (insert, delete, substitute, swap
             adjacent letters) <= 1 for targets of 4-8 characters, <= 2 for
             9+ characters. Targets of <= 3 characters are skipped (far too
             many harmless collisions).
  separator  identical after removing - _ . (e.g. `python-dateutil` vs
             `pythondateutil`), but not the same name
  affix      popular name plus a generic affix: python-, py-, py, -python,
             -py, -dev, -lib, -sdk, -api, 2, 3 (e.g. `python-requests`).
             Noisier (some legitimate packages do this) -> kept as its own
             rule so it can be dropped in the ablation.

Outputs (DATA_DIR)
  typosquat_edges.csv   name, target, rule, distance, target_rank
  name_features.csv     name, is_popular, min_dist_to_popular (capped at 3),
                        nearest_popular, n_typosquat_targets, has_typosquat_edge

Usage
  python src/make_typosquat_edges.py
  python src/make_typosquat_edges.py --top 5000 --popular-file top-pypi-packages.min.json
"""

import argparse
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein
from rapidfuzz.process import cdist

DATA_DIR = "/home/igalimi1/thesis/data"
POPULAR_URL = ("https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/"
               "top-pypi-packages.min.json")
AFFIX_PRE = ["python-", "py-", "py"]
AFFIX_SUF = ["-python", "-py", "-dev", "-lib", "-sdk", "-api", "2", "3"]


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def squash(name):
    return re.sub(r"[-_.]", "", name)


def max_dist(target_len):
    if target_len <= 3:
        return 0
    return 1 if target_len <= 8 else 2


def load_popular(path, top):
    if not os.path.exists(path):
        print(f"[info] downloading popular package list -> {path}", flush=True)
        urllib.request.urlretrieve(POPULAR_URL, path)
    data = json.load(open(path))
    names = [pep503(r["project"]) for r in data["rows"][:top]]
    print(f"[info] popular list: top {len(names)} (updated {data.get('last_update')})")
    return names


def load_nodes(data_dir):
    """Normalized names of evaluation packages (with class) and dependency targets."""
    frames = []
    for cls in ["malicious", "normal"]:
        s = pd.read_csv(f"{data_dir}/static_{cls}.csv", usecols=["Name"], dtype=str)
        frames.append(pd.DataFrame({"name": s["Name"].map(pep503), "role": cls}))
        edges = f"{data_dir}/deps_{cls}_edges.csv"
        if os.path.exists(edges):
            e = pd.read_csv(edges, usecols=["dep"], dtype=str, keep_default_na=False)
            frames.append(pd.DataFrame({"name": e["dep"].map(pep503), "role": "dependency_target"}))
    nodes = pd.concat(frames).drop_duplicates()
    roles = nodes.groupby("name")["role"].agg(lambda r: ";".join(sorted(set(r))))
    return roles  # name -> roles


def main():
    ap = argparse.ArgumentParser(description="Label-blind typosquat edges to popular packages")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--top", type=int, default=5000, help="number of popular packages used as targets")
    ap.add_argument("--popular-file", default=None,
                    help="local copy of top-pypi-packages.min.json (downloaded if missing)")
    args = ap.parse_args()
    data_dir = args.data_dir
    popular_file = args.popular_file or f"{data_dir}/top-pypi-packages.min.json"

    popular = load_popular(popular_file, args.top)
    rank = {p: i + 1 for i, p in enumerate(popular)}
    popular_set = set(popular)
    roles = load_nodes(data_dir)
    names = [n for n in roles.index if n]
    sources = [n for n in names if n not in popular_set]
    print(f"[info] nodes: {len(names)} names | candidate sources (not popular): {len(sources)}")

    edges = []
    # --- rule: edit distance (vectorized, in chunks) ---
    pop_arr = np.array(popular)
    pop_len = np.array([len(p) for p in popular])
    pop_max = np.array([max_dist(l) for l in pop_len])
    min_dist = {}
    nearest = {}
    chunk = 2000
    for start in range(0, len(sources), chunk):
        batch = sources[start:start + chunk]
        dist = cdist(batch, popular, scorer=DamerauLevenshtein.distance,
                     score_cutoff=3, dtype=np.int32, workers=-1)
        for i, name in enumerate(batch):
            row = dist[i]
            j = int(row.argmin())
            min_dist[name] = int(min(row[j], 3))
            nearest[name] = popular[j] if row[j] <= 3 else ""
            hits = np.where((row <= pop_max) & (row > 0))[0]
            for k in hits:
                edges.append((name, popular[k], "edit", int(row[k])))

    # --- rule: separator variants ---
    squashed_pop = {}
    for p in popular:
        squashed_pop.setdefault(squash(p), []).append(p)
    for name in sources:
        for p in squashed_pop.get(squash(name), []):
            if p != name:
                edges.append((name, p, "separator", 0))

    # --- rule: generic affixes ---
    for name in sources:
        for pre in AFFIX_PRE:
            if name.startswith(pre) and name[len(pre):] in popular_set:
                edges.append((name, name[len(pre):], "affix", 0))
        for suf in AFFIX_SUF:
            if name.endswith(suf) and name[:-len(suf)] in popular_set:
                edges.append((name, name[:-len(suf)], "affix", 0))

    e = pd.DataFrame(edges, columns=["name", "target", "rule", "distance"]).drop_duplicates(["name", "target", "rule"])
    e["target_rank"] = e["target"].map(rank)
    e.to_csv(f"{data_dir}/typosquat_edges.csv", index=False)

    # --- name features for every node ---
    feats = pd.DataFrame({"name": names})
    feats["is_popular"] = feats["name"].isin(popular_set).astype(int)
    feats["min_dist_to_popular"] = feats["name"].map(min_dist).fillna(0).astype(int)
    feats["nearest_popular"] = feats["name"].map(nearest).fillna("")
    n_targets = e.groupby("name")["target"].nunique()
    feats["n_typosquat_targets"] = feats["name"].map(n_targets).fillna(0).astype(int)
    feats["has_typosquat_edge"] = (feats["n_typosquat_targets"] > 0).astype(int)
    feats["roles"] = feats["name"].map(roles)
    feats.to_csv(f"{data_dir}/name_features.csv", index=False)

    # ---- Summary ----
    print(f"\n=== Typosquat edges ===")
    print(f"edges: {len(e)} | by rule: {e['rule'].value_counts().to_dict()}")
    for cls in ["malicious", "normal", "dependency_target"]:
        sel = feats[feats["roles"].str.contains(cls)]
        if sel.empty:
            continue
        by_rule = {r: sel["name"].isin(e.loc[e["rule"] == r, "name"]).mean() for r in ["edit", "separator", "affix"]}
        print(f"  {cls:18s} names: {len(sel):6d} | with any typosquat edge: {sel['has_typosquat_edge'].mean():.1%} "
              f"| edit {by_rule['edit']:.1%}, separator {by_rule['separator']:.1%}, affix {by_rule['affix']:.1%}")

    mal = set(feats.loc[feats["roles"].str.contains("malicious"), "name"])
    print("\nmost targeted popular packages (malicious sources):")
    for t, n in e[e["name"].isin(mal)]["target"].value_counts().head(15).items():
        print(f"  {t:25s} {n:5d}   (rank {rank[t]})")
    norm = set(feats.loc[feats["roles"].str.contains("normal"), "name"])
    ex = e[e["name"].isin(norm)].sample(min(10, int(e["name"].isin(norm).sum())), random_state=0)
    print("\nrandom examples among NORMAL packages (possible false positives):")
    for _, r in ex.iterrows():
        print(f"  {r['name']:30s} -> {r['target']:25s} ({r['rule']}, distance {r['distance']})")
    print(f"\nSaved {data_dir}/typosquat_edges.csv and {data_dir}/name_features.csv")


if __name__ == "__main__":
    main()
