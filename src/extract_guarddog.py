"""
extract_guarddog.py

Runs GuardDog's SOURCE-CODE rules over package versions and writes one row
of binary rule flags per package.

Two modes, same rules and same parser, so normal and malicious packages go
through identical feature extraction (parity):

  --mode pypi    scan (Name, Version) pairs from the deps.dev node CSVs,
                 downloaded live from PyPI                 [normal packages]
  --mode local   scan archive files found under a directory
                 (e.g. ~/thesis/pypi_malregistry)          [malicious packages]

FIXES vs. the previous version
  1. The version is passed with --version. The old code passed "name==version",
     which GuardDog treats as a package NAME -> PyPI 404 for every package.
  2. GuardDog's "errors" field now sets scan_failed=1 and is recorded in an
     `error` column. Previously a reported error produced an all-zero row with
     scan_failed=0, so 136,380 failed scans looked like 136,380 clean packages.
  3. Preflight check: scans one package before the real run and aborts if it
     fails or if any expected rule is missing from this GuardDog version.
     A broken setup now fails in seconds, not after hours.
  4. Summary at the end (failure rate, % flagged) so an all-zero result is
     impossible to miss.
  5. Per-rule columns (rule_<name>) kept alongside the group columns, so
     feature selection can happen later without re-scanning.

Only source-code rules are used. GuardDog's metadata rules (typosquatting,
empty_information, release_zero, ...) need live PyPI metadata that removed
malicious packages don't have, so using them would be a parity leak.

Usage
-----
  # quick test on the login node (small!)
  python src/extract_guarddog.py --mode pypi  --limit 20 --out /tmp/gd_test_pypi.csv
  python src/extract_guarddog.py --mode local --limit 20 --out /tmp/gd_test_local.csv

  # real runs (via Slurm)
  python src/extract_guarddog.py --mode pypi  --sample 15000 --out data/guarddog_normal.csv
  python src/extract_guarddog.py --mode local --out data/guarddog_malicious.csv
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import pandas as pd

DATA_DIR = os.path.expanduser("~/thesis/data")
MALREGISTRY_ROOT = os.path.expanduser("~/thesis/pypi_malregistry")
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
TIMEOUT = 600  # seconds per package

RULE_GROUPS = {
    "code_execution": ["code-execution", "exec-base64", "silent-process-execution"],
    "network_exfiltration": ["exfiltrate-sensitive-data", "download-executable", "shady-links"],
    "obfuscation": ["obfuscation", "api-obfuscation", "pyarmor", "unicode", "steganography"],
    "command_abuse": ["cmd-overwrite", "dll-hijacking"],
    "sensitive_access": ["clipboard-access", "suspicious_passwd_access_linux", "screenshot"],
}
SOURCE_RULES = [r for rules in RULE_GROUPS.values() for r in rules]
ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".zip", ".whl")


# ---------------------------------------------------------------------------
# Scanning + parsing
# ---------------------------------------------------------------------------
def empty_row(name, version, source, path=""):
    row = {"Name": name, "Version": str(version), "source": source, "path": path}
    for group in RULE_GROUPS:
        row[group] = 0
    for rule in SOURCE_RULES:
        row[f"rule_{rule}"] = 0
    row["code_issue_count"] = 0
    row["scan_failed"] = 0
    row["error"] = ""
    return row


def run_guarddog(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    return json.loads(result.stdout)


def parse_output(output, row):
    """Fill `row` from GuardDog JSON. Any reported error -> scan_failed=1."""
    errors = output.get("errors") or {}
    if errors:
        row["scan_failed"] = 1
        row["error"] = "; ".join(f"{k}: {v}" for k, v in errors.items())[:300]
        return row

    results = output.get("results")
    if not isinstance(results, dict):
        row["scan_failed"] = 1
        row["error"] = "no results field in GuardDog output"
        return row

    for group, rules in RULE_GROUPS.items():
        for rule in rules:
            val = results.get(rule)
            n_findings = len(val) if isinstance(val, (dict, list)) else 0
            if n_findings > 0:
                row[f"rule_{rule}"] = 1
                row[group] = 1
                row["code_issue_count"] += n_findings
    return row


def _scan(cmd, row):
    try:
        return parse_output(run_guarddog(cmd), row)
    except subprocess.TimeoutExpired:
        row["scan_failed"], row["error"] = 1, "timeout"
    except json.JSONDecodeError:
        row["scan_failed"], row["error"] = 1, "GuardDog output was not valid JSON"
    except Exception as e:  # noqa: BLE001
        row["scan_failed"], row["error"] = 1, f"{type(e).__name__}: {e}"[:300]
    return row


def scan_pypi(item):
    name, version = item
    cmd = ["guarddog", "pypi", "scan", name, "--version", str(version),
           "--output-format", "json"]
    return _scan(cmd, empty_row(name, version, "pypi"))


def scan_local(item):
    name, version, path = item
    cmd = ["guarddog", "pypi", "scan", path, "--output-format", "json"]
    return _scan(cmd, empty_row(name, version, "local", path))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_pypi_items(sample=None, seed=42):
    files = sorted(glob.glob(f"{DATA_DIR}/nodes-*.csv"))
    if not files:
        sys.exit(f"[error] no nodes-*.csv files in {DATA_DIR}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[["Name", "Version"]].drop_duplicates()
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
        print(f"[info] sampled {sample} of the normal packages (seed={seed})")
    return [(r.Name, str(r.Version)) for r in df.itertuples(index=False)]


def load_local_items(root):
    """
    Assumes the malregistry layout <root>/<name>/<version>/<archive>.
    CHECK THIS with `ls` on a couple of folders - the first few parsed items
    are printed so you can confirm name/version come out right.
    """
    items = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip .git
        for f in files:
            if f.endswith(ARCHIVE_EXTS):
                path = os.path.join(dirpath, f)
                parts = os.path.relpath(path, root).split(os.sep)
                name = parts[0] if len(parts) >= 2 else f
                version = parts[1] if len(parts) >= 3 else "unknown"
                items.append((name, version, path))
    if not items:
        sys.exit(f"[error] no archives found under {root}")
    print("[info] first parsed local items (check name/version!):")
    for name, version, path in items[:3]:
        print(f"       {name} | {version} | {path}")
    return items


def item_key(item):
    return item[2] if len(item) == 3 else f"{item[0]}=={item[1]}"


def load_done_keys(out_path, mode):
    if not os.path.exists(out_path):
        return set()
    done = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    if mode == "local":
        return set(done["path"])
    return set(done["Name"] + "==" + done["Version"])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(mode, first_item):
    print("[preflight] scanning one package to check the setup...")
    if mode == "pypi":
        cmd = ["guarddog", "pypi", "scan", "requests", "--version", "2.31.0",
               "--output-format", "json"]
    else:
        cmd = ["guarddog", "pypi", "scan", first_item[2], "--output-format", "json"]

    try:
        out = run_guarddog(cmd)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[preflight] FAILED to run GuardDog: {e}\nAborting before the real run.")

    if out.get("errors"):
        sys.exit(f"[preflight] FAILED: GuardDog reported errors: {out['errors']}\n"
                 "If mode=pypi on a compute node, this node may have no internet access.\n"
                 "Aborting before the real run.")

    results = out.get("results") or {}
    missing = [r for r in SOURCE_RULES if r not in results]
    if missing:
        sys.exit(f"[preflight] FAILED: these rules don't exist in this GuardDog version: "
                 f"{missing}\nFix RULE_GROUPS or change GuardDog version. Aborting.")
    print("[preflight] OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fixed GuardDog feature extraction")
    ap.add_argument("--mode", choices=["pypi", "local"], required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--local-root", default=MALREGISTRY_ROOT)
    ap.add_argument("--sample", type=int, default=None,
                    help="pypi mode: random sample of N normal packages")
    ap.add_argument("--limit", type=int, default=None,
                    help="only scan the first N items (for testing)")
    args = ap.parse_args()

    items = load_pypi_items(args.sample) if args.mode == "pypi" else load_local_items(args.local_root)
    if args.limit:
        items = items[: args.limit]

    preflight(args.mode, items[0])

    done = load_done_keys(args.out, args.mode)
    todo = [it for it in items if item_key(it) not in done]
    print(f"[info] total: {len(items)} | already done: {len(done)} | to scan: {len(todo)} "
          f"| workers: {NUM_WORKERS}", flush=True)

    scan_fn = scan_pypi if args.mode == "pypi" else scan_local
    write_header = not os.path.exists(args.out)

    if todo:
        with Pool(NUM_WORKERS) as pool:
            batch = []
            for i, row in enumerate(pool.imap_unordered(scan_fn, todo, chunksize=1), 1):
                batch.append(row)
                if len(batch) >= 20:
                    pd.DataFrame(batch).to_csv(args.out, mode="a", header=write_header, index=False)
                    write_header = False
                    batch = []
                    print(f"[progress] {i}/{len(todo)}", flush=True)
            if batch:
                pd.DataFrame(batch).to_csv(args.out, mode="a", header=write_header, index=False)

    # --- summary: an all-zero or all-failed result can't hide anymore ---
    df = pd.read_csv(args.out, dtype={"Version": str}, keep_default_na=False)
    n = len(df)
    failed = int(df["scan_failed"].astype(int).sum())
    ok = df[df["scan_failed"].astype(int) == 0]
    flagged = int((ok[list(RULE_GROUPS)].astype(int).sum(axis=1) > 0).sum())
    print("\n=== Summary ===")
    print(f"rows:            {n}")
    print(f"scan_failed:     {failed} ({100 * failed / max(n, 1):.1f}%)")
    print(f"flagged (any):   {flagged} of {len(ok)} successful scans "
          f"({100 * flagged / max(len(ok), 1):.1f}%)")
    for group in RULE_GROUPS:
        print(f"  {group:22s} {int(ok[group].astype(int).sum())}")
    if failed:
        print("\nMost common errors:")
        print(df.loc[df["scan_failed"].astype(int) == 1, "error"].value_counts().head(5).to_string())


if __name__ == "__main__":
    main()