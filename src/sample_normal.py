"""
sample_normal.py

Time-matched random sampling of normal PyPI packages, one release per project,
downloaded as sdists into <out-root>/<name>/<version>/<file> (the same layout
as malregistry, so every extractor treats both classes identically).

Algorithm
  1. Population: all PyPI projects, from PyPI's public project list.
     The list is saved on first run and reused, so the sample is reproducible.
  2. Shuffle the list with a fixed seed.
  3. For each project, in that order:
       - skip if its name appears in malregistry (never sample known malware)
       - keep only releases first uploaded inside the malware's time window
       - pick ONE of those releases uniformly at random (seeded per project)
       - skip if that year's quota is already full
       - skip, and record it, if that release has no sdist (never substitute a
         wheel: sdist availability is reported for parity, Warning 2)
       - download and checksum-verify the sdist
  4. Stop when every year's quota is full.

Year quotas are proportional to the malware's per-year counts (from the build
dates stored inside each malregistry archive), so both classes share the same
time distribution. Time is matched because packaging conventions change over
the years; nothing else is matched, since other properties are the signal.

Resumable: every inspected project is written to sample_manifest.csv, and a
rerun skips projects already in it.

Needs internet: run on the login node (small tests) or the `staging` partition.

Usage
  python src/sample_normal.py --n 5000 --out-root ~/thesis/data/normal_archives
  python src/sample_normal.py --quotas 2023:3,2024:2 --out-root /tmp/test_normal   # quick test
"""

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import random
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter

import pandas as pd

ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".zip", ".whl")
HEADERS = {"User-Agent": "thesis-research-sampler (academic research)"}
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MANIFEST_COLS = ["Name", "Version", "year", "status", "filename"]
DEFAULT_YEARS = range(2021, 2027)


def pep503(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def http_json(url, extra_headers=None, timeout=30):
    req = urllib.request.Request(url, headers={**HEADERS, **(extra_headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---------------------------------------------------------------------------
# Malware side: time distribution and names to exclude
# ---------------------------------------------------------------------------
def malware_years_and_names(root):
    """Per-year counts (latest file date inside each archive) and normalized names."""
    years, names = Counter(), set()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if not f.endswith(ARCHIVE_EXTS):
                continue
            path = os.path.join(dirpath, f)
            parts = os.path.relpath(path, root).split(os.sep)
            if len(parts) >= 2:
                names.add(pep503(parts[0]))
            try:
                if f.endswith((".zip", ".whl")):
                    with zipfile.ZipFile(path) as z:
                        stamps = [datetime.datetime(*i.date_time).timestamp() for i in z.infolist()]
                else:
                    with tarfile.open(path) as t:
                        stamps = [m.mtime for m in t.getmembers() if m.isfile()]
                if stamps:
                    ts = datetime.datetime.fromtimestamp(max(stamps), datetime.timezone.utc)
                    years[ts.year] += 1
            except Exception:  # noqa: BLE001 - unreadable archives just don't count
                pass
    return years, names


def make_quotas(year_counts, years, n):
    """Split n over the years proportionally (largest-remainder rounding)."""
    counts = {y: year_counts.get(y, 0) for y in years}
    total = sum(counts.values())
    if total == 0:
        raise SystemExit("[error] no malware found in the chosen years")
    raw = {y: n * c / total for y, c in counts.items()}
    quotas = {y: int(v) for y, v in raw.items()}
    for y in sorted(raw, key=lambda y: raw[y] - quotas[y], reverse=True)[: n - sum(quotas.values())]:
        quotas[y] += 1
    return quotas


# ---------------------------------------------------------------------------
# Normal side
# ---------------------------------------------------------------------------
def load_projects(path):
    """PyPI project list; downloaded once, then reused for reproducibility."""
    if os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print("[info] downloading PyPI project list (one-time snapshot)...", flush=True)
    data = http_json("https://pypi.org/simple/",
                     {"Accept": "application/vnd.pypi.simple.v1+json"}, timeout=300)
    names = [p["name"] for p in data["projects"]]
    with open(path, "w") as f:
        f.write(f"# PyPI project list snapshot, {datetime.date.today().isoformat()}\n")
        f.write("\n".join(names) + "\n")
    return names


def inspect_project(name, seed, years):
    """Network step: pick one in-window release at random. No quota logic here."""
    try:
        data = http_json(f"https://pypi.org/pypi/{name}/json")
    except urllib.error.HTTPError as e:
        return {"Name": name, "status": "not_found" if e.code == 404 else f"http_{e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"Name": name, "status": f"error: {type(e).__name__}"}

    eligible = {}
    for version, files in (data.get("releases") or {}).items():
        files = [f for f in files if not f.get("yanked")]
        stamps = [f.get("upload_time_iso_8601") or f.get("upload_time") for f in files]
        stamps = [s for s in stamps if s]
        if not stamps:
            continue
        year = int(min(stamps)[:4])  # first upload of this release
        if year in years:
            eligible[version] = (year, files)
    if not eligible:
        return {"Name": name, "status": "no_release_in_window"}

    version = random.Random(f"{seed}:{name}").choice(sorted(eligible))
    year, files = eligible[version]
    row = {"Name": name, "Version": version, "year": year, "status": "candidate", "filename": ""}
    sdists = [f for f in files if f.get("packagetype") == "sdist"]
    if sdists:
        row["_sdist"] = sdists[0]
    else:
        row["status"] = "no_sdist"
    return row


def download(row, out_root):
    sdist = row.pop("_sdist")
    if sdist.get("size", 0) > MAX_DOWNLOAD_BYTES:
        row["status"] = "too_large"
        return row
    dest_dir = os.path.join(out_root, row["Name"], str(row["Version"]))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, sdist["filename"])
    try:
        sha = hashlib.sha256()
        req = urllib.request.Request(sdist["url"], headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
            for chunk in iter(lambda: r.read(1 << 16), b""):
                sha.update(chunk)
                out.write(chunk)
        expected = (sdist.get("digests") or {}).get("sha256")
        if expected and sha.hexdigest() != expected:
            os.remove(dest)
            row["status"] = "checksum_mismatch"
            return row
    except Exception as e:  # noqa: BLE001
        if os.path.exists(dest):
            os.remove(dest)
        row["status"] = f"download_failed: {type(e).__name__}"
        return row
    row["status"], row["filename"] = "ok", sdist["filename"]
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_quotas(text):
    return {int(y): int(q) for y, q in (part.split(":") for part in text.split(","))}


def main():
    ap = argparse.ArgumentParser(description="Time-matched random sample of normal PyPI packages")
    ap.add_argument("--n", type=int, default=5000, help="total normal packages to sample")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--malregistry-root", default=os.path.expanduser("~/thesis/pypi_malregistry"))
    ap.add_argument("--projects-file", default=None,
                    help="PyPI project list snapshot (default: <out-root>/pypi_projects_snapshot.txt)")
    ap.add_argument("--quotas", default=None, help="override, e.g. 2023:3100,2024:930 (for tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    manifest = os.path.join(out_root, "sample_manifest.csv")

    print("[info] reading malregistry (time distribution + names to exclude)...", flush=True)
    mal_years, mal_names = malware_years_and_names(os.path.expanduser(args.malregistry_root))
    if args.quotas:
        quotas = parse_quotas(args.quotas)
    else:
        quotas = make_quotas(mal_years, DEFAULT_YEARS, args.n)
    print(f"[info] malware per year: {dict(sorted(mal_years.items()))}")
    print(f"[info] quotas: {quotas} (total {sum(quotas.values())})")
    print(f"[info] malregistry names excluded: {len(mal_names)}", flush=True)

    projects = load_projects(args.projects_file or os.path.join(out_root, "pypi_projects_snapshot.txt"))
    order = sorted(projects)
    random.Random(args.seed).shuffle(order)

    filled, seen = Counter(), set()
    if os.path.exists(manifest):
        prev = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        seen = set(prev["Name"])
        ok = prev[prev["status"] == "ok"]
        filled = Counter(ok["year"].astype(float).astype(int))  # stored as e.g. "2025.0"
    todo = [p for p in order if p not in seen]
    print(f"[info] projects: {len(projects)} | already inspected: {len(seen)} | "
          f"filled so far: {dict(filled)}", flush=True)

    write_header = not os.path.exists(manifest)
    years = set(quotas)

    def inspect(name):
        if pep503(name) in mal_names:
            return {"Name": name, "status": "excluded_malregistry"}
        return inspect_project(name, args.seed, years)

    with cf.ThreadPoolExecutor(args.threads) as pool:
        for start in range(0, len(todo), args.batch):
            if all(filled[y] >= q for y, q in quotas.items()):
                break
            rows = list(pool.map(inspect, todo[start:start + args.batch]))

            # Quota decisions in shuffled order -> deterministic for a given seed.
            accepted = []
            for row in rows:
                if row["status"] != "candidate":
                    continue
                pending = sum(1 for r in accepted if r["year"] == row["year"])
                if filled[row["year"]] + pending >= quotas[row["year"]]:
                    row["status"] = "quota_full"
                    row.pop("_sdist", None)
                else:
                    accepted.append(row)

            for row in pool.map(lambda r: download(r, out_root), accepted):
                if row["status"] == "ok":
                    filled[row["year"]] += 1

            pd.DataFrame(rows).reindex(columns=MANIFEST_COLS).to_csv(
                manifest, mode="a", header=write_header, index=False)
            write_header = False
            done = sum(min(filled[y], q) for y, q in quotas.items())
            print(f"[progress] inspected {start + len(rows)} | sampled {done}/{sum(quotas.values())} "
                  f"| {dict(sorted(filled.items()))}", flush=True)

    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    print("\n=== Sampling summary ===")
    print(df["status"].value_counts().to_string())
    print("\nsampled per year vs quota:")
    for y, q in sorted(quotas.items()):
        print(f"  {y}: {filled[y]}/{q}")
    reached = df["status"].isin(["ok", "no_sdist", "too_large", "checksum_mismatch"]).sum()
    if reached:
        share = (df["status"] == "no_sdist").sum() / reached
        print(f"\nno sdist available: {100 * share:.1f}% of eligible releases (Warning 2 number)")


if __name__ == "__main__":
    main()
