"""
make_campaigns.py
Groups malicious packages into campaigns by payload identity, and saves the
result to DATA_DIR/campaigns.csv.

Method (exact match after normalization, plus name grouping)
  - Each archive is read in memory, nothing is executed (same reader as the
    static feature extractor).
  - All .py files are used, not only setup.py: many packages share template
    setup.py files while the payload sits elsewhere, so hashing setup.py alone
    would merge unrelated packages.
  - Normalization, so that copies of the same payload match:
      * file paths made relative to the package root (top folder removed)
      * the package's own name (all spelling variants) and version replaced
        by placeholders, in paths and in code
      * long encoded string literals (base64 / hex, >= 200 characters)
        replaced by <BLOB>: many campaigns ship the same wrapper code with a
        regenerated encoded payload per upload
      * all whitespace removed
  - Fingerprint = SHA-256 of the sorted (path, normalized code) pairs.
  - Campaigns = groups of archives connected by EITHER
      A. the same package name (all releases of a name), OR
      B. the same fingerprint (same normalized code under any name).
  - Packages without .py files, or unreadable archives, get their own
    campaign of size 1 (they are not evidence of shared code).

Known limitations:
  - Different actors copying the same public template merge into one
    campaign (they share tooling, which is what a detector sees).
  - Copies that change the wrapper code itself (not just the blob) under
    different names stay apart; near-duplicate matching would be the next step.

Campaign labels are for data selection, splitting and analysis ONLY.
Never use them as a model feature or to build graph edges.

Usage
  python src/make_campaigns.py
  python src/make_campaigns.py --root ~/thesis/pypi_malregistry --out ~/thesis/data/campaigns.csv
"""

import argparse
import hashlib
import os
import re
from collections import Counter
from multiprocessing import Pool

import pandas as pd

from extract_guarddog import load_local_items
from extract_static_features import read_archive

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
DEFAULT_CHECKS = ["aiohttp-sock", "aiohttp-proxies-connector", "aiohttp-socks5-connector",
                  "yper", "vyer", "selnium"]


def name_variants(name):
    """All common spellings of a package name, longest first."""
    base = str(name)
    parts = re.split(r"[-_.]+", base)
    variants = {base, base.lower(), "-".join(parts), "_".join(parts), ".".join(parts),
                "".join(parts)}
    variants |= {v.lower() for v in variants}
    return sorted((v for v in variants if len(v) >= 2), key=len, reverse=True)


BLOB_RE = re.compile(r"""(['"])(?:[A-Za-z0-9+/=]{200,}|[0-9a-fA-F]{200,})\1""")


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def fingerprint(item):
    """Return (name, version, path, fingerprint, n_py_files) for one archive."""
    name, version, path = item
    try:
        files, texts, _ = read_archive(path)
    except Exception:  # noqa: BLE001 - unreadable: own campaign
        return name, version, path, f"unreadable:{path}", 0

    py = {k: v for k, v in texts.items() if k.endswith(".py")}
    if not py:
        return name, version, path, f"nopy:{path}", 0

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
    return name, version, path, h.hexdigest(), len(py)


def main():
    ap = argparse.ArgumentParser(description="Group malicious packages into campaigns")
    ap.add_argument("--root", default="~/thesis/pypi_malregistry")
    ap.add_argument("--out", default="~/thesis/data/campaigns.csv")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--check", nargs="*", default=DEFAULT_CHECKS,
                    help="package names whose campaigns to print for a manual check")
    args = ap.parse_args()
    root, out = os.path.expanduser(args.root), os.path.expanduser(args.out)

    items = load_local_items(root)
    print(f"[info] fingerprinting {len(items)} archives with {args.workers} workers...", flush=True)
    with Pool(args.workers) as pool:
        rows = pool.map(fingerprint, items, chunksize=16)

    df = pd.DataFrame(rows, columns=["Name", "Version", "path", "fingerprint", "n_py_files"])
    n_exact = df["fingerprint"].nunique()

    # Union-find over archives: rule A (same name) + rule B (same fingerprint)
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_groups(keys):
        first = {}
        for i, k in enumerate(keys):
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i

    union_groups(df["Name"].map(pep503))  # rule A
    after_a = len({find(i) for i in range(len(df))})
    union_groups(df["fingerprint"])        # rule B
    roots = [find(i) for i in range(len(df))]

    # stable, readable id: hash of the component's smallest path
    comp_key = pd.Series(df["path"].values).groupby(roots).transform("min")
    df["campaign_id"] = "c_" + comp_key.map(lambda k: hashlib.sha256(k.encode()).hexdigest()[:10]).values
    df["campaign_size"] = df["campaign_id"].map(df["campaign_id"].value_counts())
    df.to_csv(out, index=False)
    print(f"[info] exact code (with <BLOB>) alone: {n_exact} groups | "
          f"same name alone: {after_a} groups | combined: {df['campaign_id'].nunique()} campaigns")

    # ---- Summary ----
    campaigns = df.groupby("campaign_id").agg(size=("path", "size"),
                                              examples=("Name", lambda s: ", ".join(sorted(set(s))[:4])))
    n_single = int((campaigns["size"] == 1).sum())
    print(f"\n=== Campaigns ===")
    print(f"archives: {len(df)} | campaigns: {len(campaigns)} | "
          f"singletons: {n_single} ({100 * n_single / len(campaigns):.1f}% of campaigns, "
          f"{100 * n_single / len(df):.1f}% of archives)")
    print(f"without .py files: {int(df['fingerprint'].str.startswith('nopy:').sum())} | "
          f"unreadable: {int(df['fingerprint'].str.startswith('unreadable:').sum())}")

    buckets = pd.cut(campaigns["size"], bins=[0, 1, 5, 20, 100, 10**9],
                     labels=["1", "2-5", "6-20", "21-100", ">100"])
    print("\ncampaign sizes (number of campaigns, number of archives):")
    for label in buckets.cat.categories:
        sel = campaigns[buckets == label]
        print(f"  {label:>7}: {len(sel):6d} campaigns, {int(sel['size'].sum()):6d} archives")

    print("\nlargest 15 campaigns:")
    top = campaigns.sort_values("size", ascending=False).head(15)
    for cid, r in top.iterrows():
        n_names = df.loc[df["campaign_id"] == cid, "Name"].nunique()
        print(f"  {cid}  size {r['size']:4d}  ({n_names} names)  e.g. {r['examples']}")

    if args.check:
        print("\nmanual check (campaign of each name):")
        for name in args.check:
            sel = df[df["Name"].str.lower() == name.lower()]
            if sel.empty:
                print(f"  {name}: not found")
                continue
            for cid in sel["campaign_id"].unique():
                members = df.loc[df["campaign_id"] == cid, "Name"].unique()
                print(f"  {name}: {cid}, {len(df[df['campaign_id'] == cid])} archives, "
                      f"{len(members)} names, e.g. {', '.join(sorted(members)[:5])}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()