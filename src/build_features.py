"""
build_features.py
Computes node features for the PyPI subgraph:
  - Metadata: out-degree, in-degree, version age, recency, is-latest
  - Vulnerability: cve_count (max CVSS added later)
Saves a feature matrix aligned with the node order in build_graph.py.
"""

import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/igalimi1/thesis/data"


def load_nodes():
    files = sorted(glob.glob(f"{DATA_DIR}/nodes-*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df)} nodes")
    return df


def load_edges():
    files = sorted(glob.glob(f"{DATA_DIR}/edges-*.csv"))
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df)} edges")
    return df


def compute_features(nodes_df, edges_df):
    # Unique node id, same scheme as build_graph.py
    nodes_df["node_id"] = nodes_df["Name"] + "==" + nodes_df["Version"].astype(str)
    edges_df["from_id"] = edges_df["FromName"] + "==" + edges_df["FromVersion"].astype(str)
    edges_df["to_id"]   = edges_df["ToName"]   + "==" + edges_df["ToVersion"].astype(str)

    # ---- Degree features ----
    out_deg = edges_df.groupby("from_id").size()   # how many things each node depends on
    in_deg  = edges_df.groupby("to_id").size()     # how many things depend on each node

    nodes_df["out_degree"] = nodes_df["node_id"].map(out_deg).fillna(0)
    nodes_df["in_degree"]  = nodes_df["node_id"].map(in_deg).fillna(0)

    # ---- Temporal features ----
    nodes_df["published"] = pd.to_datetime(
        nodes_df["UpstreamPublishedAt"], errors="coerce", utc=True
    )

    # Reference date = latest publish date present in the snapshot.
    # This is the true snapshot boundary, derived from the data itself,
    # ensuring all version ages are non-negative and reproducible.
    reference_date = nodes_df["published"].max()
    print(f"Reference date (snapshot boundary): {reference_date}")

    # version age: days from publish date to snapshot boundary
    nodes_df["version_age"] = (
        reference_date - nodes_df["published"]
    ).dt.days

    # recency: days since the most recent release of the SAME package
    latest_per_pkg = nodes_df.groupby("Name")["published"].transform("max")
    nodes_df["recency"] = (latest_per_pkg - nodes_df["published"]).dt.days

    # is-latest: 1 if this version is the latest release of its package
    nodes_df["is_latest"] = (
        nodes_df["published"] == latest_per_pkg
    ).astype(int)

    # ---- Vulnerability features ----
    nodes_df["cve_count"] = nodes_df["cve_count"].fillna(0)
    # max_cvss added via a BigQuery join
    nodes_df["max_cvss"] = nodes_df["max_cvss"].fillna(0)

    # ---- Handle missing temporal values ----
    # impute missing ages/recency with the mean
    for col in ["version_age", "recency"]:
        mean_val = nodes_df[col].mean()
        nodes_df[col] = nodes_df[col].fillna(mean_val)

    return nodes_df


def main():
    nodes_df = load_nodes()
    edges_df = load_edges()

    nodes_df = compute_features(nodes_df, edges_df)

    feature_cols = [
        "out_degree", "in_degree",
        "version_age", "recency", "is_latest",
        "cve_count", "max_cvss",
    ]

    print("\n=== Feature summary ===")
    print(nodes_df[feature_cols].describe())

    # Save features aligned to node order (node_id is the key)
    out = nodes_df[["node_id"] + feature_cols]
    out_path = f"{DATA_DIR}/node_features.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved features to {out_path}")
    print(f"Feature matrix shape: {out[feature_cols].shape}")


if __name__ == "__main__":
    main()