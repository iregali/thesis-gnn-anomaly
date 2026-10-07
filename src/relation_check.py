"""
relation_check.py
Are there RELATIONS that carry malware information the package's own features
do not, above all for isolated malware? (Methodology §31.) Analysis only,
validation only: nothing here becomes an edge, and the checks decide whether a
new edge type is worth building.

Relations checked (archives read as text with the extractors' reader; nothing
is executed):
  chain              depends on another MALICIOUS name (two-stage attack:
                     clean-looking wrapper, payload in the dependency). Uses
                     labels of OTHER packages, so it is an analysis count, never
                     a feature.
  imitates           name has a typosquat edge to a popular package
                     (typosquat_edges.csv; "strict" = edit / separator rules,
                     without the noisier affix rule)
  imitates_no_use    imitates a popular package but neither declares it as a
                     dependency nor imports it (a real extension of X usually
                     uses X; a typosquat of X usually does not)
  repo_mismatch      a repository URL in the metadata (Home-page, Project-URL,
                     Download-URL; GitHub / GitLab / Bitbucket) whose repository
                     name does not match the package name
  starjack           that repository is named after a POPULAR package other
                     than this one (borrowing a famous project's identity)
  undeclared_import  imports the module of a popular package without declaring
                     any dependency (relies on it being installed already)

Reported for: all normals, all malware (one release per name; per package and
family-weighted), and VALIDATION malware split into caught / missed by the
OCSVM (static+deps+names+GuardDog, clean setting, 1% FPR on validation
normals), separately for isolated malware (no core dependencies).
Ceiling per flag: if every validation package with the flag were alarmed,
how much recall would it add on MISSED malware, and what share of validation
normals would it alarm (budget: 1%).
Finally the label-free flags are appended to the OCSVM features and the
validation result is compared with the plain OCSVM (same setting).

Known limitations (reported): module names are derived from distribution
names (requests -> requests, python-dateutil -> dateutil) with a small alias
table, so some imports are missed; repository matching is name-based.

Usage
  python src/relation_check.py --workers 16
"""
import argparse
import os
import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import (DATA_DIR, RESULTS_DIR, SPLIT_FILE, fit_model, load_features, pep503,  # noqa: E402
                               preprocess, tpr_at_fpr)
from complementarity_check import ocsvm_setting  # noqa: E402
from connectivity import load_core_deps  # noqa: E402
from extract_static_features import read_archive  # noqa: E402

CACHE = f"{DATA_DIR}/relations_cache.pkl"
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
FPR = 0.01
IMPORT_RE = re.compile(r"^[ \t]*(?:from[ \t]+([A-Za-z_]\w*)|import[ \t]+([A-Za-z_][\w \t,.]*))", re.MULTILINE)
DYN_IMPORT_RE = re.compile(r"""(?:__import__|import_module)\(\s*['"]([A-Za-z_]\w*)""")
REPO_RE = re.compile(r"https?://(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org)/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
META_KEYS = ("home-page", "project-url", "download-url")
ALIASES = {"beautifulsoup4": "bs4", "pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn",
           "python-dateutil": "dateutil", "opencv-python": "cv2", "opencv-python-headless": "cv2",
           "pyjwt": "jwt", "attrs": "attr", "pycryptodome": "Crypto", "pycryptodomex": "Cryptodome",
           "typing-extensions": "typing_extensions", "pyopenssl": "OpenSSL", "python-dotenv": "dotenv",
           "discord-py": "discord", "psycopg2-binary": "psycopg2", "pymupdf": "fitz", "pywin32": "win32api",
           "scikit-image": "skimage", "protobuf": "google", "msgpack-python": "msgpack",
           "python-telegram-bot": "telegram", "websocket-client": "websocket", "pyserial": "serial",
           "pyusb": "usb", "pygobject": "gi", "faiss-cpu": "faiss", "tensorflow-gpu": "tensorflow"}
NOT_A_SIGNAL = {"setuptools", "pip", "wheel", "distutils", "pkg-resources"}


def module_of(dist):
    """Importable top-level module of a distribution name (best effort)."""
    if dist in ALIASES:
        return ALIASES[dist]
    m = dist[len("python-"):] if dist.startswith("python-") else dist
    return m.replace("-", "_").replace(".", "_")


def relations_of(path):
    """(path, imported top-level modules, metadata repository names); never raises."""
    try:
        _, texts, _ = read_archive(path)
    except Exception:  # noqa: BLE001
        return path, None, None
    mods, repos = set(), set()
    for k, v in texts.items():
        if k.endswith(".py"):
            for a, b in IMPORT_RE.findall(v):
                if a:
                    mods.add(a)
                else:
                    for part in b.split(","):
                        part = part.strip().split(" ")[0].split(".")[0]
                        if part.isidentifier():
                            mods.add(part)
            mods.update(DYN_IMPORT_RE.findall(v))
        elif os.path.basename(k) in ("PKG-INFO", "METADATA") and k.count("/") <= 2:
            for line in v.splitlines():
                key, _, val = line.partition(":")
                if key.strip().lower() in META_KEYS:
                    for _, _, repo in REPO_RE.findall(val):
                        repo = re.sub(r"\.git$", "", repo)
                        repos.add(pep503(repo))
    return path, mods, repos


def extract(paths, workers):
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    todo = [p for p in paths if p not in cache]
    if todo:
        print(f"[info] reading {len(todo)} archives with {workers} workers (cached: {len(cache)})", flush=True)
        with Pool(workers) as pool:
            for i, (p, mods, repos) in enumerate(pool.imap_unordered(relations_of, todo, chunksize=16), 1):
                cache[p] = (mods, repos)
                if i % 2000 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
    return cache


def strip_affix(n):
    for pre in ("python-", "py-", "py"):
        if n.startswith(pre) and len(n) > len(pre) + 2:
            n = n[len(pre):]
    for suf in ("-python", "-py"):
        if n.endswith(suf):
            n = n[:-len(suf)]
    return n


def names_match(a, b):
    a, b = strip_affix(a), strip_affix(b)
    ca, cb = a.replace("-", ""), b.replace("-", "")
    return ca == cb or ca in cb or cb in ca


def fw_rate(flag, fam):
    """Family-weighted share: every family counts once."""
    if len(flag) == 0:
        return np.nan
    s = pd.Series(fam)
    w = (1.0 / s.map(s.value_counts())).values
    return float(np.sum(w * flag) / np.sum(w))


def main():
    ap = argparse.ArgumentParser(description="Relations for isolated malware: chains and imitation")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feats, groups = load_features()
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role_of = dict(zip(splits["path"], splits["split"]))
    fam_of = dict(zip(splits["path"], splits["campaign_id"]))
    w_of = dict(zip(splits["path"], splits["family_weight"].astype(float)))
    deps = load_core_deps(DATA_DIR)

    # same package set as the twin checks: all normals + one release per malicious name
    shuf = feats.sample(frac=1.0, random_state=0)
    pkgs = pd.concat([shuf[shuf["label"] == 0], shuf[shuf["label"] == 1].drop_duplicates("Name")], ignore_index=True)
    paths = pkgs["path"].values
    label = pkgs["label"].values
    name = pkgs["Name"].map(pep503).values
    family = np.array([(fam_of.get(p, "") or f"single:{p}") if l == 1 else "" for p, l in zip(paths, label)],
                      dtype=object)
    cache = extract(list(paths), args.workers)
    read_ok = np.array([cache[p][0] is not None for p in paths])
    print(f"normal {int((label == 0).sum())} | malicious names {int(label.sum())} | archives read "
          f"{read_ok.mean():.1%}", flush=True)

    # popular names and typosquat targets
    edges = pd.read_csv(f"{DATA_DIR}/typosquat_edges.csv", dtype=str, keep_default_na=False)
    tgt_all = edges.groupby("name")["target"].apply(set).to_dict()
    tgt_strict = edges[edges["rule"] != "affix"].groupby("name")["target"].apply(set).to_dict()
    nf = pd.read_csv(f"{DATA_DIR}/name_features.csv", dtype=str, keep_default_na=False)
    popular = set(nf.loc[nf["is_popular"] == "1", "name"]) - NOT_A_SIGNAL
    pop_module = {module_of(p): p for p in popular}
    mal_names = set(name[label == 1])

    rows = []
    for p, n in zip(paths, name):
        mods, repos = cache[p]
        mods, repos = mods or set(), repos or set()
        d = deps.get(p, set())
        tg = tgt_all.get(n, set())
        uses = {t for t in tg if t in d or module_of(t) in mods}
        undeclared = {pop_module[m] for m in mods if m in pop_module} - d - {n}
        rows.append({
            "isolated": len(d) == 0,
            "chain": len(d & (mal_names - {n})) > 0,
            "imitates": len(tg) > 0,
            "imitates_strict": len(tgt_strict.get(n, set())) > 0,
            "imitates_no_use": len(tg) > 0 and len(uses) == 0,
            "has_repo": len(repos) > 0,
            "repo_mismatch": any(not names_match(r, n) for r in repos),
            "starjack": any(r in popular and r != n and not names_match(r, n) for r in repos),
            "undeclared_import": len(d) == 0 and len(undeclared) > 0,
        })
    R = pd.DataFrame(rows)
    flags = ["chain", "imitates", "imitates_strict", "imitates_no_use", "has_repo", "repo_mismatch", "starjack",
             "undeclared_import"]

    # OCSVM decisions (clean training, as in complementarity_check / twins_vs_detector_check)
    y_all = feats["label"].values
    role_all = feats["path"].map(role_of).values
    train_idx = np.where((role_all == "train") & (y_all == 0))[0]
    val_idx = np.where(role_all == "val")[0]
    setting = {**ocsvm_setting(0.0), "trim": 0.0}
    cols = groups["static"] + groups["deps"] + groups["names"] + groups["guarddog"]
    X_raw = feats[cols].fillna(0).values.astype(float)

    def ocsvm(X):
        Xm = preprocess(X, train_idx, setting["mode"])
        return -fit_model("OCSVM", setting, Xm[train_idx], args.seed).decision_function(Xm)

    s_all = pd.Series(ocsvm(X_raw), index=feats["path"].values)
    s = s_all.reindex(paths).values
    prole = np.array([role_of.get(p, "") for p in paths])
    v_nor = (prole == "val") & (label == 0)
    v_mal = (prole == "val") & (label == 1)
    thr = np.quantile(s[v_nor], 1 - FPR)
    caught = s > thr
    print(f"[info] OCSVM {setting}: validation malware names caught at 1% FPR: {caught[v_mal].mean():.1%}",
          flush=True)

    def block(mask):
        out = {"n": int(mask.sum())}
        for f in flags:
            v = R[f].values[mask]
            out[f] = v.mean() if mask.sum() else np.nan
            if label[mask].all() and mask.sum():
                out[f + "_fw"] = fw_rate(v, family[mask])
        return out

    groups_def = {
        "normal (all)": label == 0,
        "malicious (all names)": label == 1,
        "  isolated malicious": (label == 1) & R["isolated"].values,
        "normal, validation": v_nor,
        "val malware CAUGHT": v_mal & caught,
        "val malware MISSED": v_mal & ~caught,
        "  missed & isolated": v_mal & ~caught & R["isolated"].values,
        "  missed & connected": v_mal & ~caught & ~R["isolated"].values,
    }
    tab = pd.DataFrame({g: block(m) for g, m in groups_def.items()}).T
    pd.set_option("display.width", 250)
    print("\n=== Share of packages with each relation (per package; *_fw = every malware family counts once) ===")
    print(tab[["n"] + flags].round(3).to_string())
    print("\n--- family-weighted (malware groups only) ---")
    print(tab[[f + "_fw" for f in flags]].dropna(how="all").round(3).to_string())

    # ceiling: alarm every validation package with the flag
    missed_share = float((v_mal & ~caught).sum() / v_mal.sum())
    ceil = []
    for f in flags:
        v = R[f].values
        ceil.append({"flag": f, "missed_with_flag": v[v_mal & ~caught].mean(),
                     "recall_gain_ceiling": v[v_mal & ~caught].mean() * missed_share,
                     "normals_alarmed": v[v_nor].mean(),
                     "normals_alarmed_not_already": (v & ~caught)[v_nor].mean()})
    print(f"\n=== Ceiling (validation): alarm every package with the flag; missed share {missed_share:.1%}; "
          f"false-alarm budget {FPR:.0%} ===")
    print(pd.DataFrame(ceil).round(3).to_string(index=False))

    # label-free flags as extra OCSVM features
    extra = ["imitates_no_use", "repo_mismatch", "starjack", "undeclared_import"]
    E = R.set_index(pd.Index(paths))[extra].astype(float).reindex(feats["path"].values).fillna(0).values
    s_aug = ocsvm(np.hstack([X_raw, E]))
    in_set = set(paths)
    ev = np.array([i for i in val_idx if feats["path"].values[i] in in_set])   # validation rows with flags
    yv, wv = y_all[ev], feats["path"].map(w_of).values[ev]
    res = []
    for nm, sc in [("OCSVM", s_all.values), ("OCSVM + relation flags", s_aug)]:
        sv = sc[ev]
        res.append({"model": nm, "val_auc_fw": roc_auc_score(yv, sv, sample_weight=wv),
                    "val_caught1_fw": tpr_at_fpr(sv, yv, wv, FPR)[1]})
    print("\n=== Validation: OCSVM with the label-free flags appended (same setting) ===")
    print(pd.DataFrame(res).round(3).to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tab.to_csv(f"{RESULTS_DIR}/relation_check_groups.csv")
    pd.DataFrame(ceil).to_csv(f"{RESULTS_DIR}/relation_check_ceiling.csv", index=False)
    print("\nchain examples (wrapper -> malicious dependency):")
    ex = [(name[i], sorted(deps.get(paths[i], set()) & (mal_names - {name[i]}))[:3])
          for i in np.where(R["chain"].values & (label == 1))[0][:20]]
    for a, b in ex:
        print(f"  {a} -> {', '.join(b)}")
    print(f"\nSaved {RESULTS_DIR}/relation_check_*.csv")


if __name__ == "__main__":
    main()
