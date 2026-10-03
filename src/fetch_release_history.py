"""
fetch_release_history.py
Release history of every sampled normal package, from PyPI's JSON API.
Used ONLY to define evaluation populations (is the sampled release the
project's first release? how old was the project?), never as a model feature:
release history does not exist in the same form for removed malware (parity).

Question it answers (Methodology §20, "is the normal sample realistic?"):
the sampler picks a random release of a project active in the time window, so
the normal set mixes first releases of new projects with later releases of
established ones. A registry screening new projects only ever sees the former.

Output: ~/thesis/data/normal_release_history.csv, one row per normal archive
  path, Name, Version
  status                 ok | not_found | version_missing | error: ...
  n_releases             releases with at least one uploaded file
  n_releases_no_files    releases listed without files (cannot be dated)
  project_first_upload   earliest upload of any file of the project (incl. yanked)
  release_first_upload   earliest non-yanked upload of the sampled release
                         (same definition as sample_normal.py; all files if all yanked)
  release_rank           number of the project's releases first uploaded earlier
  is_first_release       release_rank == 0
  project_age_days       release_first_upload - project_first_upload, in days

Caveat: releases deleted from PyPI vanish from the API, so "first release" means
"first release still listed". Releases listed without files are counted.

Resumable: rows already in the output are skipped. Needs internet: run on the
login node or the `staging` partition.

Usage
  python src/fetch_release_history.py
  python src/fetch_release_history.py --limit 50          # quick test
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

DATA_DIR = "/home/igalimi1/thesis/data"
OUT = f"{DATA_DIR}/normal_release_history.csv"
HEADERS = {"User-Agent": "thesis-research-sampler (academic research)"}
COLS = ["path", "Name", "Version", "status", "n_releases", "n_releases_no_files",
        "project_first_upload", "release_first_upload", "release_rank",
        "is_first_release", "project_age_days"]
INT_COLS = ["n_releases", "n_releases_no_files", "release_rank", "is_first_release"]


def norm_version(v):
    """Light version normalisation for matching (PEP 440 if `packaging` exists)."""
    try:
        from packaging.version import Version
        return str(Version(str(v)))
    except Exception:  # noqa: BLE001
        return re.sub(r"[-_.]+", ".", str(v)).lower().lstrip("v")


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def http_json(name, tries=4):
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == tries - 1:
                raise
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
        time.sleep(2 ** k)


def stamps(files, skip_yanked):
    out = []
    for f in files:
        if skip_yanked and f.get("yanked"):
            continue
        s = f.get("upload_time_iso_8601") or f.get("upload_time")
        if s:
            out.append(ts(s if ("+" in s or s.endswith("Z")) else s + "+00:00"))
    return out


def history(row):
    base = {"path": row["path"], "Name": row["Name"], "Version": row["Version"]}
    try:
        data = http_json(row["Name"])
    except urllib.error.HTTPError as e:
        return {**base, "status": "not_found" if e.code == 404 else f"http_{e.code}"}
    except Exception as e:  # noqa: BLE001
        return {**base, "status": f"error: {type(e).__name__}"}

    releases = data.get("releases") or {}
    first_of = {}            # version -> first upload (any file, incl. yanked)
    no_files = 0
    for v, files in releases.items():
        st = stamps(files, skip_yanked=False)
        if st:
            first_of[v] = min(st)
        else:
            no_files += 1
    if not first_of:
        return {**base, "status": "no_dated_releases", "n_releases_no_files": no_files}

    # find the sampled release: exact key first, then normalised match
    key = row["Version"] if row["Version"] in releases else None
    if key is None:
        nv = norm_version(row["Version"])
        key = next((v for v in releases if norm_version(v) == nv), None)
    if key is None or key not in first_of:
        return {**base, "status": "version_missing", "n_releases": len(first_of),
                "n_releases_no_files": no_files}

    rel_st = stamps(releases[key], skip_yanked=True) or stamps(releases[key], skip_yanked=False)
    rel_first = min(rel_st)
    proj_first = min(first_of.values())
    rank = sum(1 for v, t in first_of.items() if v != key and t < rel_first)
    return {**base, "status": "ok", "n_releases": len(first_of), "n_releases_no_files": no_files,
            "project_first_upload": proj_first.isoformat(), "release_first_upload": rel_first.isoformat(),
            "release_rank": rank, "is_first_release": int(rank == 0),
            "project_age_days": round((rel_first - proj_first).total_seconds() / 86400, 2)}


def main():
    ap = argparse.ArgumentParser(description="Release history of the sampled normal packages")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N packages (test)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    st = pd.read_csv(f"{DATA_DIR}/static_normal.csv", usecols=["Name", "Version", "path"],
                     dtype=str, keep_default_na=False).drop_duplicates("path")
    if args.limit:
        st = st.head(args.limit)
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, dtype=str, keep_default_na=False)
        done = set(prev.loc[~prev["status"].str.startswith(("error", "http_")), "path"])
    todo = st[~st["path"].isin(done)].to_dict("records")
    print(f"{len(st)} normal packages | already done {len(done)} | to fetch {len(todo)}", flush=True)

    write_header = not os.path.exists(args.out)
    buf = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(history, todo), 1):
            buf.append(res)
            if len(buf) >= 200 or i == len(todo):
                out = pd.DataFrame(buf, columns=COLS)
                for c in INT_COLS:
                    out[c] = pd.to_numeric(out[c]).astype("Int64")
                out.to_csv(args.out, mode="a", header=write_header, index=False)
                write_header, buf = False, []
                print(f"  {i}/{len(todo)}", flush=True)

    # keep the latest row per path (retried errors are appended later)
    df = pd.read_csv(args.out, dtype=str, keep_default_na=False).drop_duplicates("path", keep="last")
    df.to_csv(args.out, index=False)
    ok = df[df["status"] == "ok"]
    age = pd.to_numeric(ok["project_age_days"])
    print(f"\nstatus: {df['status'].value_counts().to_dict()}")
    print(f"first release: {pd.to_numeric(ok['is_first_release']).mean():.1%} of {len(ok)} | "
          f"project age at release: median {age.median():.0f} days, "
          f"<= 30 days {(age <= 30).mean():.1%}, > 365 days {(age > 365).mean():.1%}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
