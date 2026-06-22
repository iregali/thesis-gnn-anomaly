"""
extract_guarddog.py
Runs GuardDog over each package version in the subgraph in PARALLEL
and extracts code-level features as binary rule-trigger flags.

Robust: saves progress incrementally and is resumable.
"""

import glob
import json
import subprocess
import os
import pandas as pd
from multiprocessing import Pool

DATA_DIR = "/home/igalimi1/thesis/data"
OUTPUT_PATH = f"{DATA_DIR}/guarddog_features.csv"
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

RULE_GROUPS = {
    "code_execution": [
        "code-execution", "exec-base64", "silent-process-execution",
    ],
    "network_exfiltration": [
        "exfiltrate-sensitive-data", "download-executable", "shady-links",
    ],
    "obfuscation": [
        "obfuscation", "api-obfuscation", "pyarmor", "unicode",
        "steganography",
    ],
    "command_abuse": [
        "cmd-overwrite", "dll-hijacking",
    ],
    "sensitive_access": [
        "clipboard-access", "suspicious_passwd_access_linux",
    ],
}


def scan_package(args):
    """Run GuardDog on one package version, return a feature dict."""
    name, version = args
    feats = {
        "Name": name, "Version": version,
        "code_execution": 0, "network_exfiltration": 0,
        "obfuscation": 0, "command_abuse": 0,
        "sensitive_access": 0, "total_issues": 0, "scan_failed": 0,
    }
    try:
        result = subprocess.run(
            ["guarddog", "pypi", "scan",
             f"{name}=={version}", "--output-format", "json"],
            capture_output=True, text=True, timeout=120,
        )
        output = json.loads(result.stdout)
        results = output.get("results", {})
        feats["total_issues"] = output.get("issues", 0)
        for group, rules in RULE_GROUPS.items():
            for rule in rules:
                val = results.get(rule)
                if isinstance(val, dict) and len(val) > 0:
                    feats[group] = 1
                    break
    except Exception:
        feats["scan_failed"] = 1
    return feats


def load_packages():
    files = sorted(glob.glob(f"{DATA_DIR}/nodes-*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df[["Name", "Version"]]


def load_done():
    if os.path.exists(OUTPUT_PATH):
        done = pd.read_csv(OUTPUT_PATH)
        return set(done["Name"] + "==" + done["Version"].astype(str))
    return set()


def main():
    packages = load_packages()
    done_ids = load_done()
    print(f"Total packages: {len(packages)}, already done: {len(done_ids)}")

    # build the list of packages still to scan
    todo = [
        (row["Name"], row["Version"])
        for _, row in packages.iterrows()
        if f"{row['Name']}=={row['Version']}" not in done_ids
    ]
    print(f"To scan: {len(todo)} using {NUM_WORKERS} workers")

    write_header = not os.path.exists(OUTPUT_PATH)

    # process in parallel, writing results in batches as they complete
    with Pool(NUM_WORKERS) as pool:
        batch = []
        for i, feats in enumerate(
            pool.imap_unordered(scan_package, todo, chunksize=10)
        ):
            batch.append(feats)
            # flush every 200 results
            if len(batch) >= 200:
                pd.DataFrame(batch).to_csv(
                    OUTPUT_PATH, mode="a", header=write_header, index=False
                )
                write_header = False
                batch = []
                print(f"Processed {i+1}/{len(todo)}")
        # write any remaining
        if batch:
            pd.DataFrame(batch).to_csv(
                OUTPUT_PATH, mode="a", header=write_header, index=False
            )

    print("Done")


if __name__ == "__main__":
    main()