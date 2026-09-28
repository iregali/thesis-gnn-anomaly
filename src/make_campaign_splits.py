"""
make_campaign_splits.py
Creates DATA_DIR/splits_campaign.csv: the same packages as splits.csv, but
malware is assigned to validation / test BY CAMPAIGN FAMILY (from
campaigns.csv), so a family never appears on both sides. Every test package
then comes from a family not seen during model selection.

Rules
  - Normal packages: assigned exactly as in splits.csv (same hash rule, reused
    from make_splits.py), so they are identical in both split files.
  - Malware: family assigned by a hash of its campaign_id: ~20% of families to
    validation, ~80% to test.
  - Cap: at most CAP packages per family are kept in the evaluation sets; the
    rest get split = "excluded" (never used). Which members are kept is
    decided by a hash of the archive path, so it is reproducible.
    Without the cap, one template (~40% of all malware) would dominate the
    evaluation sets.
  - family_weight = 1 / (number of kept members of the family): with these
    weights every family counts equally in family-weighted metrics.
    Normal packages have weight 1.

Campaign families are used only to organise the data, never as features.

Usage
  python src/make_campaign_splits.py
  python src/make_campaign_splits.py --cap 20
"""

import argparse
import hashlib
import os

import pandas as pd

from baseline_experiment import DATA_DIR, load
from make_splits import assign, group_of, u01

SALT = "thesis-campaign-split-v1"
MALWARE_VAL = 0.2


def u01_salted(key):
    h = hashlib.sha256(f"{SALT}:{key}".encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def main():
    ap = argparse.ArgumentParser(description="Campaign-based split with a per-family cap")
    ap.add_argument("--cap", type=int, default=10, help="max packages per family in evaluation sets")
    ap.add_argument("--campaigns", default=f"{DATA_DIR}/campaigns.csv")
    ap.add_argument("--out", default=f"{DATA_DIR}/splits_campaign.csv")
    args = ap.parse_args()

    feats = pd.concat([load("malicious", 1), load("normal", 0)], ignore_index=True)
    df = feats[["Name", "Version", "path", "label"]].copy()
    df["group"] = df["Name"].map(group_of)

    camps = pd.read_csv(args.campaigns, dtype={"Version": str})
    df["campaign_id"] = df["path"].map(dict(zip(camps["path"], camps["campaign_id"])))
    missing = int(((df["label"] == 1) & df["campaign_id"].isna()).sum())
    if missing:
        raise SystemExit(f"{missing} malicious packages have no campaign in {args.campaigns}")

    # Normal: identical to splits.csv. Malware: by family.
    normal = df["label"] == 0
    df.loc[normal, "split"] = [assign(0, u01(g)) for g in df.loc[normal, "group"]]
    mal = ~normal
    df.loc[mal, "split"] = ["val" if u01_salted(c) < MALWARE_VAL else "test"
                            for c in df.loc[mal, "campaign_id"]]

    # Cap: keep at most CAP members per family (stable choice by path hash)
    df["_order"] = df["path"].map(u01_salted)
    rank = df[mal].sort_values("_order").groupby("campaign_id").cumcount()
    df.loc[rank.index[rank >= args.cap], "split"] = "excluded"
    df = df.drop(columns="_order")

    # Family weights over kept members
    kept_mal = mal & (df["split"] != "excluded")
    kept_per_family = df.loc[kept_mal].groupby("campaign_id")["path"].transform("size")
    df["family_weight"] = 1.0
    df.loc[kept_mal, "family_weight"] = 1.0 / kept_per_family
    df.loc[normal, "campaign_id"] = ""

    df.to_csv(args.out, index=False)

    # ---- Summary ----
    print(f"\nSaved {len(df)} packages to {args.out} (cap = {args.cap} per family)")
    print(pd.crosstab(df["split"], df["label"].map({0: "normal", 1: "malicious"}),
                      margins=True).reindex(["train", "val", "test", "excluded", "All"]))
    for s in ["val", "test"]:
        sel = df[(df["split"] == s) & mal]
        sizes = sel.groupby("campaign_id").size()
        print(f"\n{s}: {len(sel)} malicious packages from {sizes.size} families "
              f"({int((sizes == 1).sum())} with one kept package); "
              f"largest family share: {sizes.max() / max(len(sel), 1):.1%}")
    both = set(df.loc[(df["split"] == "val") & mal, "campaign_id"]) & \
        set(df.loc[(df["split"] == "test") & mal, "campaign_id"])
    print(f"\nfamilies in both validation and test: {len(both)} (must be 0)")
    reference = pd.read_csv(f"{DATA_DIR}/splits.csv", dtype={"Version": str}) \
        if os.path.exists(f"{DATA_DIR}/splits.csv") else None
    if reference is not None:
        ref = dict(zip(reference["path"], reference["split"]))
        same = (df.loc[normal, "path"].map(ref) == df.loc[normal, "split"]).mean()
        print(f"normal packages with the same split as splits.csv: {same:.1%} (must be 100%)")


if __name__ == "__main__":
    main()
