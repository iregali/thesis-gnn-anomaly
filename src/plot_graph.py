"""
plot_graph.py
Picture of the training graph for the thesis (Figure: "The training graph").

  (a) a random sample of the CONNECTED evaluated releases of one graph bundle, with every
      dependency target they declare (sampled releases or context nodes); hub context nodes
      are labelled. Normal releases blue, unlabelled malware of the contamination draw red,
      context nodes grey (size = number of releases in the full graph that depend on it).
  (b) share of releases that are isolated (no core dependency and no training release that
      depends on their name): training releases of the bundle by class, and the held-out
      test releases by class, as they are inserted for scoring.

Only reads a bundle written by build_graph.py; no training, no labels used for anything but
colouring. Runs on a login node in a few seconds.

  python src/plot_graph.py                                 # graph_c05_s0.pt, 300 releases
  python src/plot_graph.py --bundle graph_c00_s0.pt --sample 400 --seed 1
  python src/plot_graph.py --sample 150 --min-shared 1     # also draw one-off dependencies
Output: thesis_out/figures/fig_graph.{pdf,png}
"""
import argparse
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "thesis_out", "figures")
COL_NORMAL, COL_MAL, COL_CTX = "#2a6fdb", "#d6402b", "#9a9a9a"


def isolated_shares(n_eval, dep_src, dep_dst, is_contam, heldout_split, heldout_label,
                    heldout_has_deps, heldout_has_dependents):
    """Shares of isolated releases: training (normal / contamination) and test (normal / malicious)."""
    deg = np.bincount(np.concatenate([dep_src, dep_dst]), minlength=n_eval)[:n_eval]
    iso_train = deg == 0
    test = heldout_split == "test"
    iso_held = ~(heldout_has_deps | heldout_has_dependents)
    out = {"Training, normal": iso_train[~is_contam[:n_eval]].mean(),
           "Test, normal": iso_held[test & (heldout_label == 0)].mean(),
           "Test, malicious": iso_held[test & (heldout_label == 1)].mean()}
    if is_contam[:n_eval].any():
        out["Training, hidden malware"] = iso_train[is_contam[:n_eval]].mean()
        out = {k: out[k] for k in ["Training, normal", "Training, hidden malware",
                                   "Test, normal", "Test, malicious"]}
    return out


def draw(n_eval, n_nodes, dep_src, dep_dst, is_contam, names, shares, sample=300, seed=0,
         n_labels=6, out=OUT, title_note="", min_shared=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    rng = np.random.default_rng(seed)
    in_deg = np.bincount(dep_dst, minlength=n_nodes)            # dependents in the full graph
    out_deg = np.bincount(dep_src, minlength=n_nodes)
    connected = np.where((in_deg[:n_eval] + out_deg[:n_eval]) > 0)[0]
    pick = rng.choice(connected, size=min(sample, len(connected)), replace=False)
    # always show some hidden malware that is connected, if the draw has any
    mal_conn = np.intersect1d(connected, np.where(is_contam[:n_eval])[0])
    if len(mal_conn):
        pick = np.union1d(pick, rng.choice(mal_conn, size=min(12, len(mal_conn)), replace=False))
    pick_set = set(pick.tolist())
    keep = [(s, d) for s, d in zip(dep_src, dep_dst) if s in pick_set or d in pick_set]

    G = nx.Graph()
    G.add_nodes_from(pick.tolist())
    G.add_edges_from(keep)
    # context nodes used by fewer than min_shared sampled releases add only a fringe of leaves
    if min_shared > 1:
        G.remove_nodes_from([n for n in list(G.nodes()) if n >= n_eval and G.degree(n) < min_shared])
        G.remove_nodes_from(list(nx.isolates(G)))
    # small separate components would be pushed to the margins: draw the largest one
    n_before = G.number_of_nodes()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"largest component of the sample: {G.number_of_nodes()} of {n_before} nodes")
    pos = nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=seed, iterations=100))

    nodes = np.array(list(G.nodes()))
    ctx = nodes >= n_eval
    mal = np.array([n < n_eval and bool(is_contam[n]) for n in nodes])
    norm = ~ctx & ~mal

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.35, alpha=0.35, edge_color="#777777")
    size_ctx = 6 + 60 * np.sqrt(in_deg[nodes[ctx]] / max(in_deg[n_eval:].max(), 1))
    ax.scatter(*np.array([pos[n] for n in nodes[ctx]]).T, s=size_ctx, c=COL_CTX, edgecolors="white",
               linewidths=0.4, zorder=2, label="Context node shared by sampled releases")
    ax.scatter(*np.array([pos[n] for n in nodes[norm]]).T, s=9, c=COL_NORMAL, zorder=3,
               label="Normal release")
    if mal.any():
        ax.scatter(*np.array([pos[n] for n in nodes[mal]]).T, s=22, c=COL_MAL, marker="D", zorder=4,
                   edgecolors="white", linewidths=0.4, label="Hidden malware (contamination)")
    hubs = nodes[ctx][np.argsort(-in_deg[nodes[ctx]])][:n_labels]
    # spread labels: each label goes to the free side, away from labels already placed
    placed = []
    offsets = [(8, 8), (8, -14), (-8, 8), (-8, -14), (14, -3), (-14, -3)]
    span = np.ptp(np.array([pos[n] for n in nodes]), axis=0).max() or 1.0
    for h in hubs:
        x, y = pos[h]
        best = max(offsets, key=lambda o: min([np.hypot(x + o[0] * span / 400 - px, y + o[1] * span / 400 - py)
                                               for px, py in placed] or [1e9]))
        lx, ly = x + best[0] * span / 400, y + best[1] * span / 400
        placed.append((lx, ly))
        ax.annotate(names[h], (x, y), xytext=best, textcoords="offset points", fontsize=8,
                    ha="left" if best[0] > 0 else "right", va="center", color="#222222", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_axis_off()
    ax.set_title(f"(a) Sample of connected releases and their dependencies{title_note}",
                 fontsize=10, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, fontsize=8, frameon=False, markerscale=1.4)

    labs = list(shares)
    vals = [100 * shares[k] for k in labs]
    cols = [COL_MAL if "malic" in k or "malware" in k else COL_NORMAL for k in labs]
    bars = bx.barh(range(len(labs)), vals, color=cols, height=0.6)
    for b, v in zip(bars, vals):
        bx.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    bx.set_yticks(range(len(labs)), labs, fontsize=9)
    bx.invert_yaxis()
    bx.set_xlim(0, 100)
    bx.set_xlabel("Isolated releases (%)", fontsize=9)
    bx.spines[["top", "right"]].set_visible(False)
    bx.tick_params(axis="x", labelsize=8)
    bx.set_title("(b) Releases without any dependency relation", fontsize=10, loc="left")

    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_graph.{ext}"), dpi=200, bbox_inches="tight")
    print(f"sample: {len(pick)} releases, {int(ctx.sum())} context nodes, {G.number_of_edges()} edges "
          f"(of {len(connected)} connected releases); hubs: {', '.join(names[h] for h in hubs)}")
    print("isolated shares: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in shares.items()))
    print(f"-> {out}/fig_graph.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bundle", default="graph_c05_s0.pt", help="file name in DATA_DIR, or a path")
    ap.add_argument("--sample", type=int, default=300, help="connected releases to draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-shared", type=int, default=2,
                    help="draw only context nodes used by at least this many sampled releases (1 = all)")
    args = ap.parse_args()

    from build_graph import load_bundle
    from baseline_experiment import DATA_DIR
    path = args.bundle if os.path.exists(args.bundle) else os.path.join(DATA_DIR, args.bundle)
    b = load_bundle(path)
    g, nodes, h = b["graph"], b["nodes"], b["heldout"]
    n_eval = int((nodes["kind"] == "evaluated").sum())
    src, dst = (g.dep_edge_index[0].numpy(), g.dep_edge_index[1].numpy())
    names = nodes["name"].astype(str).tolist()
    is_contam = nodes["is_contamination"].to_numpy(bool)
    shares = isolated_shares(
        n_eval, src, dst, is_contam,
        h["split"].to_numpy(), h["label"].to_numpy(),
        np.array([len(d) > 0 for d in b["heldout_deps"]]),
        np.array([b["node_of_name"].get(n) in b["dependents"] for n in h["name"]]))
    lvl = b["config"]["contam"]
    draw(n_eval, g.num_nodes, src, dst, is_contam, names, shares, args.sample, args.seed,
         title_note=f" ({lvl:.0%} contamination)" if lvl else "", min_shared=args.min_shared)


if __name__ == "__main__":
    main()