"""
build_graph.py
Loads the PyPI subgraph CSVs into a PyTorch Geometric Data object
and verifies the node/edge/positive counts.
"""

import glob
import pandas as pd
import torch
from torch_geometric.data import Data


def load_nodes(data_dir):
    """Load all node CSV files into one dataframe."""
    files = sorted(glob.glob(f"{data_dir}/nodes-*.csv"))
    print(f"Found {len(files)} node files")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df)} nodes")
    return df


def load_edges(data_dir):
    """Load all edge CSV files into one dataframe."""
    files = sorted(glob.glob(f"{data_dir}/edges-*.csv"))
    print(f"Found {len(files)} edge files")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df)} edges")
    return df


def build_data(nodes_df, edges_df):
    """Construct a PyTorch Geometric Data object from the dataframes."""

    # Create a unique string id for each node: "name==version"
    nodes_df["node_id"] = nodes_df["Name"] + "==" + nodes_df["Version"].astype(str)

    # Map each node_id to an integer index (0, 1, 2, ...)
    node_to_idx = {nid: i for i, nid in enumerate(nodes_df["node_id"])}

    # Build edge id strings the same way
    edges_df["from_id"] = edges_df["FromName"] + "==" + edges_df["FromVersion"].astype(str)
    edges_df["to_id"]   = edges_df["ToName"]   + "==" + edges_df["ToVersion"].astype(str)

    # Keep only edges where BOTH endpoints exist in the node set
    valid = edges_df["from_id"].isin(node_to_idx) & edges_df["to_id"].isin(node_to_idx)
    edges_df = edges_df[valid]
    print(f"Valid edges (both endpoints in node set): {len(edges_df)}")

    # Convert edge endpoints to integer indices
    src = edges_df["from_id"].map(node_to_idx).values
    dst = edges_df["to_id"].map(node_to_idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Build a simple label vector: 1 if vulnerable (cve_count > 0), else 0
    # NOTE: these are NOT used for training — only for evaluation
    y = torch.tensor(
        (nodes_df["cve_count"] > 0).astype(int).values,
        dtype=torch.long
    )

    # For now, a placeholder feature matrix (we build real features later)
    # one feature: cve_count, just so the object is valid
    x = torch.tensor(
        nodes_df[["cve_count"]].fillna(0).values,
        dtype=torch.float
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, node_to_idx


def main():
    data_dir = "/home/igalimi1/thesis/data"

    nodes_df = load_nodes(data_dir)
    edges_df = load_edges(data_dir)

    data, node_to_idx = build_data(nodes_df, edges_df)

    print("\n=== Graph Summary ===")
    print(data)
    print(f"Num nodes:      {data.num_nodes}")
    print(f"Num edges:      {data.num_edges}")
    print(f"Num positives:  {int(data.y.sum())}")
    print(f"Num normals:    {int((data.y == 0).sum())}")

    # Verify against BigQuery numbers
    assert data.num_nodes == 136380, \
        f"Expected 136380 nodes, got {data.num_nodes}"
    assert int(data.y.sum()) == 39033, \
        f"Expected 39033 positives, got {int(data.y.sum())}"
    print("\n✓ Counts match BigQuery — data transferred correctly")


if __name__ == "__main__":
    main()