"""
fetch_popular_snapshots.py
Retrieves yearly snapshots (early July, 2021-2026) of the most-downloaded PyPI
packages from the public hugovk/top-pypi-packages repository history.

Why yearly snapshots: popularity shifts a lot (only ~61% of the top 5,000
overlap between July 2023 and 2026), and the malware spans 2021-2026. A single
list misses targets that were popular only in some years (e.g.
`aiohappyeyeballs` became popular from 2025). make_typosquat_edges.py uses the
UNION of the snapshots' top N.

Uses a partial git clone (history without file contents), then downloads only
the needed versions. Needs internet: run on the login node.

Usage
  python src/fetch_popular_snapshots.py --out-dir ~/thesis/data/popular
"""

import argparse
import json
import os
import subprocess
import tempfile

REPO = "https://github.com/hugovk/top-pypi-packages.git"
FILE = "top-pypi-packages-30-days.min.json"


def main():
    ap = argparse.ArgumentParser(description="Yearly PyPI popularity snapshots")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/thesis/data/popular"))
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2021, 2027)))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO, tmp], check=True)
        for year in args.years:
            log = subprocess.run(["git", "-C", tmp, "log", "-1", f"--before={year}-07-02",
                                  "--format=%H %cs", "--", FILE], capture_output=True, text=True, check=True)
            if not log.stdout.strip():
                print(f"{year}: no snapshot found")
                continue
            sha, date = log.stdout.split()
            raw = subprocess.run(["git", "-C", tmp, "show", f"{sha}:{FILE}"],
                                 capture_output=True, text=True, check=True).stdout
            data = json.loads(raw)
            out = os.path.join(args.out_dir, f"top-pypi-{year}-07.min.json")
            with open(out, "w") as f:
                json.dump(data, f)
            print(f"{year}: snapshot {date}, {len(data['rows'])} packages -> {out}")


if __name__ == "__main__":
    main()
