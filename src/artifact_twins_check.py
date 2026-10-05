"""
artifact_twins_check.py
Would SHARED-ARTIFACT edges carry a real malware signal, or only a sampling
artifact? (Methodology §28.) Analysis only: nothing here becomes a feature or
an edge, and campaign labels are not used.

Artifacts, read from each archive as text (same reader as the extractors;
nothing is executed), identically for both classes:
  email        author / maintainer e-mail in PKG-INFO
  author       normalised author / maintainer name in PKG-INFO
  url          URL in the code, normalised to host + path (no query)
  domain       host of a URL in the code
  blob         hash of a long base64 / hex string literal (>= 200 characters)
  code_exact   fingerprint of all normalised .py code (as in make_campaigns.py)
  code_near    estimated Jaccard >= 0.8 of the normalised code (MinHash + LSH)
Two packages are TWINS for an artifact type if they share an artifact of that
type. Artifacts held by more than --max-share of the packages are ignored
(placeholders such as author@example.com, github.com).

The sampling problem: malregistry is close to the COMPLETE set of known
PyPI malware, while the normal packages are a small random sample of PyPI
projects. Siblings of a malicious package are therefore present, siblings of
a normal package mostly are not, and twins would look like a malware signal
even if both classes had them equally often.

Density matching: malware is reduced to one release per name (normals have one
release per project) and then subsampled to the SAME sampling fraction as the
normals. The normal fraction is unknown exactly (15,000 sampled projects out of
the PyPI projects active in the time window), so several fractions are
reported (--fractions). Twin rates are computed in the pooled sample
(all normals + the malware subsample), repeated --reps times.

Reading
  malware twin rate >> normal twin rate at the realistic fractions
      -> a real collective signal: shared-artifact edges are worth building
         (with a temporal protocol)
  similar rates once matched -> the signal was a sampling artifact

Usage
  python src/artifact_twins_check.py                     # extract (cached) + analyse
  python src/artifact_twins_check.py --fractions 0.01 0.02 0.05 --reps 50
"""

import argparse
import hashlib
import os
import pickle
import re
import sys
from multiprocessing import Pool
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from baseline_protocol import DATA_DIR, RESULTS_DIR, load_features  # noqa: E402
from extract_static_features import read_archive  # noqa: E402
from make_campaigns import BLOB_RE, minhash_of, name_variants, near_duplicate_pairs  # noqa: E402

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
CACHE = f"{DATA_DIR}/artifacts_cache.pkl"
TYPES = ["email", "author", "url", "domain", "blob", "code_exact", "code_near"]
URL_RE = re.compile(r"""https?://[^\s'"<>()\\]+""", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def pkg_info_fields(texts):
    for k, v in texts.items():
        if os.path.basename(k) == "PKG-INFO" and k.count("/") <= 1:
            out = {}
            for line in v.splitlines():
                if not line or line[0].isspace():
                    continue
                key, _, val = line.partition(":")
                key = key.strip().lower()
                if key in ("author", "author-email", "maintainer", "maintainer-email") and val.strip():
                    out.setdefault(key, []).append(val.strip())
            return out
    return {}


def artifacts_of(item):
    """(path, {type: set(artifacts)}, minhash or None); never raises."""
    name, version, path = item
    arts = {t: set() for t in TYPES if t != "code_near"}
    try:
        _, texts, _ = read_archive(path)
    except Exception:  # noqa: BLE001
        return path, arts, None
    meta = pkg_info_fields(texts)
    for key in ("author-email", "maintainer-email", "author", "maintainer"):
        for val in meta.get(key, []):
            arts["email"].update(e.lower() for e in EMAIL_RE.findall(val))
            if not key.endswith("email"):
                n = norm_name(EMAIL_RE.sub(" ", val))
                if n and n not in ("unknown", "none", "author", "your name"):
                    arts["author"].add(n)
    py = {k: v for k, v in texts.items() if k.endswith(".py")}
    for code in py.values():
        for u in URL_RE.findall(code):
            try:
                parts = urlsplit(u.rstrip(".,;:'\")"))
            except ValueError:
                continue
            host = (parts.hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if not host or host in ("localhost", "127.0.0.1"):
                continue
            arts["domain"].add(host)
            arts["url"].add(host + parts.path.rstrip("/"))
        for m in BLOB_RE.finditer(code):
            arts["blob"].add(hashlib.sha1(m.group(0).encode("utf-8", "replace")).hexdigest())
    mh = None
    if py:
        patterns = [re.compile(re.escape(v), re.IGNORECASE) for v in name_variants(name)]
        version_re = re.compile(re.escape(str(version))) if str(version) not in ("", "unknown") else None

        def normalize(text):
            text = BLOB_RE.sub("'<BLOB>'", text)
            for p in patterns:
                text = p.sub("<NAME>", text)
            if version_re:
                text = version_re.sub("<VERSION>", text)
            return re.sub(r"\s+", "", text)

        pairs = sorted((normalize("/".join(k.split("/")[1:]) or k), normalize(v)) for k, v in py.items())
        h = hashlib.sha256()
        for rel, code in pairs:
            h.update(rel.encode("utf-8", "replace") + b"\x00" + code.encode("utf-8", "replace") + b"\x01")
        arts["code_exact"].add(h.hexdigest())
        mh = minhash_of("".join(rel + code for rel, code in pairs))
    return path, arts, mh


def extract(items, workers):
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
        todo = [it for it in items if it[2] not in cache]
    else:
        cache, todo = {}, items
    if todo:
        print(f"[info] reading {len(todo)} archives with {workers} workers (cached: {len(cache)})", flush=True)
        with Pool(workers) as pool:
            for i, (path, arts, mh) in enumerate(pool.imap_unordered(artifacts_of, todo, chunksize=16), 1):
                cache[path] = (arts, mh)
                if i % 2000 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
    return cache


def main():
    ap = argparse.ArgumentParser(description="Density-matched artifact-twin check")
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.01, 0.02, 0.05, 0.10, 1.0])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--max-share", type=float, default=0.01,
                    help="ignore artifacts held by more than this share of all packages")
    ap.add_argument("--near", type=float, default=0.8, help="near-duplicate threshold (code_near)")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    feats, _ = load_features()
    bad = [p for p in feats["path"].values[:20] if not os.path.exists(p)]
    if bad:
        raise SystemExit(f"archive paths not found, e.g. {bad[:2]}: the 'path' column must point to archives")
    # one release per name for malware (normals already have one per project)
    feats = feats.sample(frac=1.0, random_state=0)
    mal = feats[feats["label"] == 1].drop_duplicates("Name")
    nor = feats[feats["label"] == 0]
    pkgs = pd.concat([nor, mal], ignore_index=True)
    print(f"normal {len(nor)} | malicious (one release per name) {len(mal)}", flush=True)
    cache = extract(list(zip(pkgs["Name"], pkgs["Version"], pkgs["path"])), args.workers)

    n = len(pkgs)
    label = pkgs["label"].values
    paths = pkgs["path"].values
    # artifact -> member indices, per type (only artifacts held by >= 2 packages can create twins)
    groups = {}
    for t in TYPES[:-1]:
        mem = {}
        for i, p in enumerate(paths):
            for a in cache[p][0][t]:
                mem.setdefault(a, []).append(i)
        groups[t] = [np.array(v) for v in mem.values() if len(v) >= 2]
    near_mh = {i: cache[p][1] for i, p in enumerate(paths) if cache[p][1] is not None}
    print(f"[info] near-duplicate matching over {len(near_mh)} packages (threshold {args.near})", flush=True)
    near_pairs = np.array(sorted(near_duplicate_pairs(near_mh, args.near)), dtype=int).reshape(-1, 2)
    print("[info] artifact coverage (share of packages with >= 1 artifact of the type):")
    for t in TYPES[:-2]:
        cov = np.array([len(cache[p][0][t]) > 0 for p in paths])
        print(f"  {t:10s} normal {cov[label == 0].mean():.1%} | malicious {cov[label == 1].mean():.1%}")

    def twin_flags(pool):
        """Per type: (has a twin in the pool, has a NORMAL twin in the pool) for every package.
        Artifacts held by more than max_share of the POOL are ignored (too common)."""
        cap = max(2, int(args.max_share * pool.sum()))
        out = {}
        for t in TYPES:
            has = np.zeros(n, dtype=bool)
            has_norm = np.zeros(n, dtype=bool)
            if t == "code_near":
                ok = pool[near_pairs[:, 0]] & pool[near_pairs[:, 1]] if len(near_pairs) else []
                for a, b in near_pairs[ok] if len(near_pairs) else []:
                    has[a] = has[b] = True
                    has_norm[a] |= label[b] == 0
                    has_norm[b] |= label[a] == 0
            else:
                for g in groups[t]:
                    m = g[pool[g]]
                    if len(m) < 2 or len(m) > cap:
                        continue
                    has[m] = True
                    n_norm = int((label[m] == 0).sum())
                    # a package has a normal twin if another normal member exists
                    has_norm[m] |= (n_norm - (label[m] == 0)) > 0
            out[t] = (has, has_norm)
        return out

    idx_nor = np.where(label == 0)[0]
    idx_mal = np.where(label == 1)[0]
    rows = []
    for f in args.fractions:
        k = max(1, int(round(f * len(idx_mal))))
        for rep in range(args.reps if f < 1.0 else 1):
            rng = np.random.default_rng(1000 + rep)
            pick = rng.choice(idx_mal, size=k, replace=False) if f < 1.0 else idx_mal
            pool = np.zeros(n, dtype=bool)
            pool[idx_nor] = True
            pool[pick] = True
            flags = twin_flags(pool)
            for t in TYPES:
                has, has_norm = flags[t]
                rows.append({"fraction": f, "rep": rep, "type": t, "n_malicious": k,
                             "twin_rate_malicious": has[pick].mean(), "twin_rate_normal": has[idx_nor].mean(),
                             "malicious_with_normal_twin": has_norm[pick].mean()})
        print(f"  fraction {f}: done", flush=True)
    res = pd.DataFrame(rows).groupby(["type", "fraction"]).agg(
        n_malicious=("n_malicious", "first"),
        malicious=("twin_rate_malicious", "mean"), malicious_sd=("twin_rate_malicious", "std"),
        normal=("twin_rate_normal", "mean"),
        malicious_with_normal_twin=("malicious_with_normal_twin", "mean")).reset_index()
    res["ratio"] = res["malicious"] / res["normal"].replace(0, np.nan)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(f"{RESULTS_DIR}/artifact_twins.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Twin rates in the pooled sample (all normals + malware subsampled to the fraction) ===")
    print("malicious / normal = share of packages with >= 1 twin of that type in the pool;")
    print("fraction 1.0 = all malware (one release per name): NOT density-matched, shown for contrast")
    print(res.round(3).to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/artifact_twins.csv")


if __name__ == "__main__":
    main()
