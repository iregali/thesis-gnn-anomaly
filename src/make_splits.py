"""
make_splits.py
Assigns every package to train / validation / test ONCE, and saves the
assignment to DATA_DIR/splits.csv. Every experiment (baselines, GNN) reads
this file, so all methods are evaluated on exactly the same packages.

Rules
  - Same package set as the baselines: readable sdists, one archive per
    release (uses load() from baseline_experiment.py, so the filtering is
    identical).
  - Assignment by a hash of the normalized package name, not by shuffling:
      * all releases of a package name always land in the same split
      * stable: adding packages later (e.g. extending the normal sample)
        never moves an existing package to another split
  - Normal:    50% train, 20% validation, 30% test
  - Malicious: 20% validation, 80% test (never train: the detectors and
    the GNN learn from normal packages only)

Usage
  python src/make_splits.py            # create, or add new packages
  python src/make_splits.py --force    # overwrite (only if you really mean it)
"""

import hashlib
import os
import re
import sys

import pandas as pd

from baseline_experiment import DATA_DIR, load

SALT = "thesis-split-v1"   # changing this reshuffles everything: don't
NORMAL_CUTS = (0.5, 0.7)   # < 0.5 train, < 0.7 val, else test
MALWARE_VAL = 0.2          # < 0.2 val, else test


def group_of(name):
    """PEP 503 normalized package name: all releases share one group."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def u01(group):
    """Stable pseudo-random number in [0, 1) from the package name."""
    h = hashlib.sha256(f"{SALT}:{group}".encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def assign(label, u):
    if label == 0:
        return "train" if u < NORMAL_CUTS[0] else ("val" if u < NORMAL_CUTS[1] else "test")
    return "val" if u < MALWARE_VAL else "test"


def main():
    out = f"{DATA_DIR}/splits.csv"
    feats = pd.concat([load("malicious", 1), load("normal", 0)], ignore_index=True)

    df = feats[["Name", "Version", "path", "label"]].copy()
    df["group"] = df["Name"].map(group_of)
    df["split"] = [assign(label, u01(g)) for label, g in zip(df["label"], df["group"])]

    # Safety check: an existing assignment must never change.
    if os.path.exists(out) and "--force" not in sys.argv:
        old = pd.read_csv(out, dtype={"Version": str})
        merged = df.merge(old[["path", "split"]], on="path", how="left", suffixes=("", "_old"))
        changed = merged["split_old"].notna() & (merged["split_old"] != merged["split"])
        if changed.any():
            sys.exit(f"[error] {int(changed.sum())} packages would change split. "
                     "Did SALT or the cut-offs change? Aborting; nothing written.")
        print(f"[info] existing assignments unchanged; "
              f"{int(merged['split_old'].isna().sum())} new packages added")

    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} packages to {out}")
    print(pd.crosstab(df["split"], df["label"].map({0: "normal", 1: "malicious"}),
                      margins=True).reindex(["train", "val", "test", "All"]))


if __name__ == "__main__":
    main()
