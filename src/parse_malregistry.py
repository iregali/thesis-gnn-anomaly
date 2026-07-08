#!/usr/bin/env python3
"""
Parse the pypi_malregistry repo into a clean (name, version) CSV
ready for `bq load` into BigQuery.

Repo structure (confirmed):
    <package_name>/<version>/<package-version>.tar.gz

Usage:
    python parse_malregistry.py --repo /path/to/pypi_malregistry --out malregistry_list.csv

If --repo is omitted, the script clones the repo into ./pypi_malregistry first.
"""

import argparse
import csv
import os
import subprocess
import sys

REPO_URL = "https://github.com/lxyeternal/pypi_malregistry.git"

# Directories at the top level that are NOT packages
IGNORE_TOP = {".git"}


def clone_repo(dest: str) -> str:
    if os.path.isdir(dest):
        print(f"[info] repo already present at {dest}, skipping clone")
        return dest
    print(f"[info] cloning {REPO_URL} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
    return dest


def parse(repo: str):
    """Walk <repo>/<package>/<version>/ and yield (name, version)."""
    rows = []
    skipped = []
    for pkg in sorted(os.listdir(repo)):
        if pkg in IGNORE_TOP or pkg.startswith("."):
            continue
        pkg_path = os.path.join(repo, pkg)
        if not os.path.isdir(pkg_path):
            continue  # stray files like .gitignore, README
        versions = [
            v for v in os.listdir(pkg_path)
            if os.path.isdir(os.path.join(pkg_path, v))
        ]
        if not versions:
            # package dir with no version subdir - record with empty version, flag it
            skipped.append(pkg)
            continue
        for ver in sorted(versions):
            rows.append((pkg, ver))
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None,
                    help="Path to an existing clone. If omitted, clones fresh.")
    ap.add_argument("--out", default="malregistry_list.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    repo = args.repo or clone_repo(os.path.join(os.getcwd(), "pypi_malregistry"))
    if not os.path.isdir(repo):
        print(f"[error] repo path not found: {repo}", file=sys.stderr)
        sys.exit(1)

    rows, skipped = parse(repo)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Version"])          # header matches your BQ convention
        w.writerows(rows)

    print(f"[done] wrote {len(rows)} (name, version) rows -> {args.out}")
    print(f"[info] distinct packages: {len({r[0] for r in rows})}")
    if skipped:
        print(f"[warn] {len(skipped)} package dirs had no version subdir "
              f"(e.g. {skipped[:5]}) - not included")


if __name__ == "__main__":
    main()
