"""
make_campaigns.py
Groups malicious packages into campaigns and saves the result to
DATA_DIR/campaigns.csv.

Each archive is read in memory; nothing is executed (same reader as the
static feature extractor). All .py files are used, not only setup.py: many
packages share template setup.py files while the payload sits elsewhere.

Normalization, so that copies of the same payload match:
  - file paths made relative to the package root (top folder removed)
  - the package's own name (all spelling variants) and version replaced
  - long encoded string literals (base64 / hex, >= 200 characters) replaced
    by <BLOB>: many campaigns ship the same wrapper with a regenerated payload
  - all whitespace removed

Campaigns = groups of archives connected by ANY of:
  A. the same package name (all releases of a name)
  B. identical normalized code (same fingerprint)
  C. near-identical normalized code: estimated Jaccard similarity of
     9-character shingles >= THRESHOLD (MinHash + LSH, 128 permutations).
     Only for code of at least MIN_LEN characters: tiny wrappers look alike
     across unrelated actors. The threshold is deliberately high: malicious
     forks of the same legitimate library share much of its code without
     being from the same actor.

The output reports campaign counts for several thresholds (sensitivity) and
the similarity of chosen name pairs (to calibrate on known cases).

Known limitation: different actors copying the same public template merge
(they share tooling, which is what a detector sees).

Campaign labels are for data selection, splitting and analysis ONLY.
Never use them as a model feature or to build graph edges.

Usage
  python src/make_campaigns.py
  python src/make_campaigns.py --threshold 0.8 --sensitivity 0.7 0.8 0.9
"""

import argparse
import hashlib
import os
import re
from multiprocessing import Pool

import pandas as pd
from datasketch import MinHash, MinHashLSH

from extract_guarddog import load_local_items
from extract_static_features import read_archive

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
NUM_PERM = 128
SHINGLE = 9
MIN_LEN = 500          # near-duplicate matching only for code at least this long
MAX_CHARS = 300_000    # cap on normalized code used for shingling
DEFAULT_CHECKS = ["aiohttp-sock", "aiohttp-proxies-connector", "yper", "selnium"]
DEFAULT_PAIRS = ["aiohttp-proxies-connector:aiohttp-proxy2",
                 "aiohttp-proxies-connector:aiohttp-sock",
                 "aiohttp-proxies-connector:aiohttp-proxy5",
                 "aiohttp-proxies-connector:aiohttp-proxies"]

BLOB_RE = re.compile(r"""(['"])(?:[A-Za-z0-9+/=]{200,}|[0-9a-fA-F]{200,})\1""")


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def name_variants(name):
    """All common spellings of a package name, longest first."""
    base = str(name)
    parts = re.split(r"[-_.]+", base)
    variants = {base, base.lower(), "-".join(parts), "_".join(parts), ".".join(parts),
                "".join(parts)}
    variants |= {v.lower() for v in variants}
    return sorted((v for v in variants if len(v) >= 2), key=len, reverse=True)


def minhash_of(text):
    text = text[:MAX_CHARS]
    if len(text) < MIN_LEN:
        return None
    shingles = {text[i:i + SHINGLE] for i in range(len(text) - SHINGLE + 1)}
    m = MinHash(num_perm=NUM_PERM, seed=1)
    m.update_batch([s.encode("utf-8", "replace") for s in shingles])
    return m


def process(item):
    """Return (name, version, path, fingerprint, n_py_files, norm_len, minhash)."""
    name, version, path = item
    try:
        _, texts, _ = read_archive(path)
    except Exception:  # noqa: BLE001 - unreadable: own campaign
        return name, version, path, f"unreadable:{path}", 0, 0, None

    py = {k: v for k, v in texts.items() if k.endswith(".py")}
    if not py:
        return name, version, path, f"nopy:{path}", 0, 0, None

    patterns = [re.compile(re.escape(v), re.IGNORECASE) for v in name_variants(name)]
    version_re = re.compile(re.escape(str(version))) if str(version) not in ("", "unknown") else None

    def normalize(text):
        text = BLOB_RE.sub("'<BLOB>'", text)
        for p in patterns:
            text = p.sub("<NAME>", text)
        if version_re:
            text = version_re.sub("<VERSION>", text)
        return re.sub(r"\s+", "", text)

    pairs = []
    for file_path, code in py.items():
        parts = file_path.split("/")
        rel = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
        pairs.append((normalize(rel), normalize(code)))
    pairs.sort()

    h = hashlib.sha256()
    for rel, code in pairs:
        h.update(rel.encode("utf-8", "replace") + b"\x00" + code.encode("utf-8", "replace") + b"\x01")
    joined = "".join(rel + code for rel, code in pairs)
    return name, version, path, h.hexdigest(), len(py), len(joined), minhash_of(joined)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def components(n, groups_of_keys, extra_pairs=()):
    """Union-find over n items; items sharing a key (per key list) or listed
    as a pair are merged. Returns the root of every item."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for keys in groups_of_keys:
        first = {}
        for i, k in enumerate(keys):
            if k in first:
                union(i, first[k])
            else:
                first[k] = i
    for i, j in extra_pairs:
        union(i, j)
    return [find(i) for i in range(n)]


def near_duplicate_pairs(minhashes, threshold):
    """Pairs of representative indices with estimated Jaccard >= threshold."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for i, m in minhashes.items():
        lsh.insert(str(i), m)
    pairs = set()
    for i, m in minhashes.items():
        for j in map(int, lsh.query(m)):
            if j != i and m.jaccard(minhashes[j]) >= threshold:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Group malicious packages into campaigns")
    ap.add_argument("--root", default="~/thesis/pypi_malregistry")
    ap.add_argument("--out", default="~/thesis/data/campaigns.csv")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="near-duplicate threshold used for the saved campaigns")
    ap.add_argument("--sensitivity", type=float, nargs="*", default=[0.7, 0.8, 0.9, 0.95],
                    help="thresholds for which only the campaign count is reported")
    ap.add_argument("--check", nargs="*", default=DEFAULT_CHECKS,
                    help="package names whose campaigns to print")
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS,
                    help="name1:name2 pairs whose estimated similarity to print")
    args = ap.parse_args()
    root, out = os.path.expanduser(args.root), os.path.expanduser(args.out)

    items = load_local_items(root)
    print(f"[info] processing {len(items)} archives with {args.workers} workers...", flush=True)
    with Pool(args.workers) as pool:
        rows = pool.map(process, items, chunksize=16)

    df = pd.DataFrame(rows, columns=["Name", "Version", "path", "fingerprint", "n_py_files",
                                     "norm_len", "minhash"])
    n = len(df)
    name_keys = df["Name"].map(pep503).tolist()
    fp_keys = df["fingerprint"].tolist()

    # Near-duplicate matching on one representative per exact fingerprint
    reps = df.drop_duplicates("fingerprint")
    minhashes = {i: m for i, m in zip(reps.index, reps["minhash"]) if m is not None}
    print(f"[info] near-duplicate candidates: {len(minhashes)} distinct code groups "
          f"(>= {MIN_LEN} characters)", flush=True)

    # Sensitivity: campaign counts per threshold
    counts = {"exact code only": len(set(components(n, [fp_keys]))),
              "same name only": len(set(components(n, [name_keys]))),
              "same name + exact code": len(set(components(n, [name_keys, fp_keys])))}
    chosen_pairs = None
    for t in sorted(set(args.sensitivity) | {args.threshold}):
        pairs = near_duplicate_pairs(minhashes, t)
        counts[f"+ near-duplicates >= {t}"] = len(set(components(n, [name_keys, fp_keys], pairs)))
        if t == args.threshold:
            chosen_pairs = pairs

    roots = components(n, [name_keys, fp_keys], chosen_pairs)
    comp_key = pd.Series(df["path"].values).groupby(roots).transform("min")
    df["campaign_id"] = "c_" + comp_key.map(lambda k: hashlib.sha256(k.encode()).hexdigest()[:10]).values
    df["campaign_size"] = df["campaign_id"].map(df["campaign_id"].value_counts())
    df.drop(columns=["minhash"]).to_csv(out, index=False)

    # ---- Summary ----
    print("\n=== Grouping rules (number of campaigns) ===")
    for label, c in counts.items():
        marker = "   <- saved" if label == f"+ near-duplicates >= {args.threshold}" else ""
        print(f"  {label:32s} {c:6d}{marker}")

    if args.pairs:
        print("\n=== Similarity of chosen pairs (estimated Jaccard, 9-char shingles) ===")
        by_name = df.drop_duplicates("Name").set_index("Name")
        for pair in args.pairs:
            a, b = pair.split(":")
            if a not in by_name.index or b not in by_name.index:
                print(f"  {a} vs {b}: not found")
                continue
            ma, mb = by_name.at[a, "minhash"], by_name.at[b, "minhash"]
            if ma is None or mb is None:
                print(f"  {a} vs {b}: too short for near-duplicate matching")
                continue
            same = by_name.at[a, "campaign_id"] == by_name.at[b, "campaign_id"]
            print(f"  {a} vs {b}: {ma.jaccard(mb):.3f}   same campaign: {same}")

    campaigns = df.groupby("campaign_id").agg(size=("path", "size"),
                                              examples=("Name", lambda s: ", ".join(sorted(set(s))[:4])))
    n_single = int((campaigns["size"] == 1).sum())
    print(f"\n=== Campaigns (threshold {args.threshold}) ===")
    print(f"archives: {n} | campaigns: {len(campaigns)} | singletons: {n_single} "
          f"({100 * n_single / n:.1f}% of archives)")
    print(f"without .py files: {int(df['fingerprint'].str.startswith('nopy:').sum())} | "
          f"unreadable: {int(df['fingerprint'].str.startswith('unreadable:').sum())}")

    buckets = pd.cut(campaigns["size"], bins=[0, 1, 5, 20, 100, 10**9],
                     labels=["1", "2-5", "6-20", "21-100", ">100"])
    print("\ncampaign sizes (number of campaigns, number of archives):")
    for label in buckets.cat.categories:
        sel = campaigns[buckets == label]
        print(f"  {label:>7}: {len(sel):6d} campaigns, {int(sel['size'].sum()):6d} archives")

    print("\nlargest 15 campaigns:")
    for cid, r in campaigns.sort_values("size", ascending=False).head(15).iterrows():
        n_names = df.loc[df["campaign_id"] == cid, "Name"].nunique()
        print(f"  {cid}  size {r['size']:4d}  ({n_names} names)  e.g. {r['examples']}")

    if args.check:
        print("\nmanual check (campaign of each name):")
        for name in args.check:
            sel = df[df["Name"].str.lower() == name.lower()]
            if sel.empty:
                print(f"  {name}: not found")
                continue
            cid = sel["campaign_id"].iloc[0]
            members = df.loc[df["campaign_id"] == cid, "Name"].unique()
            print(f"  {name}: {cid}, {int((df['campaign_id'] == cid).sum())} archives, "
                  f"{len(members)} names, e.g. {', '.join(sorted(members)[:6])}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()