"""
make_tables_figures.py
Every table and figure of the results chapter, built ONLY from the summary
result files in results/ (no data, no GPU, no Snellius). Numbers are never typed
by hand: re-run this script after any rerun and the thesis follows.

Inputs  results/*.csv   (baseline_protocol_results_conn / _expl, cola_results,
                         contrastive_results_* / contrastive_settings_*,
                         paired_comparisons(_expl), sensitivity_check_*,
                         learning_curve_raw, tuning_choice, sr_choice)
Outputs <out>/tables/*.tex   one table environment each (needs \\usepackage{booktabs})
        <out>/figures/*.pdf  vector figures for LaTeX (+ .png previews)
        <out>/all_tables.tex a standalone document that \\input's every table (check by compiling)

Conventions (as everywhere in the thesis)
  - test set, family-weighted metrics; "caught" = share of malware detected at 1% FPR
  - contrastive rows: the configuration (depth, scorer) chosen on VALIDATION per level
    (contrastive_settings, column `chosen`), plain scores (no per-group calibration)
  - caught for multi-seed models = mean over the 3 training seeds (+ sd);
    ROC-AUC = rank-averaged seed ensemble with its bootstrap 95% CI

Usage
  python src/make_tables_figures.py                       # results/ -> thesis_out/
  python src/make_tables_figures.py --results results --out thesis_out
"""
import argparse
import os

import numpy as np
import pandas as pd

LOSSES = [("infonce", "InfoNCE"), ("triplet", "Triplet"), ("dgi", "DGI")]
ENCODERS = [("gin", "GIN"), ("gat", "GAT"), ("tconv", "TransformerConv"), ("gps", "GraphGPS")]
VARIANTS = [("tuned", "Base (tuned)"), ("n1", "N1 same-group"), ("n2", "N2 shifts"),
            ("n3", "N3 full"), ("n4", "N4 uniform shifts")]
GD = "static+deps+names+GuardDog"

# validated categorical palette (dataviz reference palette, light mode), fixed order
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df"}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def esc(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def f3(v):
    return "--" if v is None or pd.isna(v) else f"{v:.3f}"


def pct(v, d=1):
    return "--" if v is None or pd.isna(v) else f"{100 * v:.{d}f}"


def ci3(r, k="auc_fw"):
    lo, hi = r.get(f"{k}_lo"), r.get(f"{k}_hi")
    return f"{f3(r[k])} [{f3(lo)}, {f3(hi)}]" if lo is not None and not pd.isna(lo) else f3(r[k])


def caught_sd(r):
    m, s = r.get("tpr1_fw_seed_mean"), r.get("tpr1_fw_seed_std")
    if m is None or pd.isna(m):
        return pct(r.get("tpr1_fw"))
    return f"{pct(m)} $\\pm$ {pct(s)}" if s is not None and not pd.isna(s) and s > 0 else pct(m)


def table(name, caption, label, header, rows, colspec, note=None, size=r"\small"):
    lines = [r"\begin{table}[htbp]", r"\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\begin{adjustbox}{max width=\linewidth}", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines += [r"\par\smallskip", f"\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}"]
    lines.append(r"\end{table}")
    with open(f"{OUT}/tables/{name}.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    TABLES.append(name)


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------
COMPACT = [("contrastive_results", "contrastive_results"), ("contrastive_settings", "contrastive_settings"),
           ("baseline_", "baseline_results"), ("cola_results", "cola_ocgnn_results"),
           ("ocgnn_results", "cola_ocgnn_results"), ("paired_comparisons", "paired_comparisons"),
           ("sensitivity_check", "sensitivity_check"), ("learning_curve", "learning_curve"),
           ("tuning_choice", "selection"), ("sr_choice", "selection"), ("label_noise", "label_noise_summary")]
_cache = {}


def read(name):
    """One result table by its original file name. Works with the per-run files and with the
    compacted layout (one file per result type, column `run` = original file name)."""
    f = f"{RES}/{name}.csv"
    if os.path.exists(f):
        d = pd.read_csv(f)
        if "run" in d.columns:            # compacted file with the same name as one of its runs
            d = d[d["run"] == name] if (d["run"] == name).any() else d
            d = d.drop(columns=["run"]).dropna(axis=1, how="all").reset_index(drop=True)
        return d
    for prefix, comp in COMPACT:
        if name.startswith(prefix) and os.path.exists(f"{RES}/{comp}.csv"):
            if comp not in _cache:
                _cache[comp] = pd.read_csv(f"{RES}/{comp}.csv", low_memory=False)
            d = _cache[comp]
            if "run" not in d.columns:
                return None
            d = d[d["run"] == name]
            if not len(d):
                return None
            return d.drop(columns=["run"]).dropna(axis=1, how="all").reset_index(drop=True)
    return None



def chosen_row(tag, level):
    """Validation-chosen configuration of a contrastive run at a level (plain scores)."""
    d, s = read(f"contrastive_results{tag}"), read(f"contrastive_settings{tag}")
    if d is None or s is None:
        return None
    ch = s[np.isclose(s["level"].astype(float), level) & s["chosen"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    if not len(ch):
        return None
    dep, sc = int(ch["depth"].iloc[0]), ch["scorer"].iloc[0]
    return method_row(d, level, sc, dep)


def method_row(d, level, scorer, depth):
    name = "no graph" if depth == 0 else f"depth {depth}"
    m = d[np.isclose(pd.to_numeric(d["level"], errors="coerce"), level)
          & d["method"].str.contains(f" {scorer}, ", regex=False) & d["method"].str.contains(name, regex=False)
          & ~d["method"].str.contains("calibrated")]
    if not len(m):
        return None
    r = m.iloc[0].copy()
    r["config"] = f"{scorer}, {'no graph' if depth == 0 else f'depth {depth}'}"
    return r


def baseline(method, level, which="conn"):
    d = read(f"baseline_protocol_results_{which}")
    lv = pd.to_numeric(d["level"], errors="coerce")
    m = d[(d["method"] == method) & ((np.isclose(lv, level)) if level is not None else d["level"].isin(["all", "ref"]))]
    return m.iloc[0] if len(m) else None


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def t_baselines():
    rows = []
    gd = baseline("GuardDog rule", None)
    for fs, fs_lab in [("static+deps+names", "without GuardDog"), (GD, "with GuardDog")]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{Features: static + dependencies + names, {fs_lab}}}}} \\\\")
        for det, lab in [("IF", "Isolation Forest"), ("LOF", "LOF"), ("OCSVM", "One-Class SVM"),
                         ("OCSVM (subgroup-calibrated)", "One-Class SVM, per-group thresholds")]:
            r = baseline(f"{det} ({fs})", 0.0)
            if r is not None:
                rows.append([lab, ci3(r), f"{pct(r['tpr1_fw'])} [{pct(r['tpr_fw_lo'])}, {pct(r['tpr_fw_hi'])}]",
                             f3(r["sg_auc_fw_conn"]), f3(r["sg_auc_fw_iso"])])
    rows.append(r"\midrule")
    rows.append(["GuardDog (findings + flag)", ci3(gd), "--", f3(gd["sg_auc_fw_conn"]), f3(gd["sg_auc_fw_iso"])])
    rf = baseline("RandomForest (supervised reference)", None)
    rows.append(["Random Forest, supervised (reference)", ci3(rf),
                 f"{pct(rf['tpr1_fw'])} [{pct(rf['tpr_fw_lo'])}, {pct(rf['tpr_fw_hi'])}]",
                 f3(rf["sg_auc_fw_conn"]), f3(rf["sg_auc_fw_iso"])])
    table("tab_baselines", "Feature-based detectors, clean training (test set, family-weighted).",
          "tab:baselines", ["Detector", "ROC-AUC [95\\% CI]", "Caught (\\%) [95\\% CI]",
                            "Conn. AUC", "Iso. AUC"], rows, "lcccc", size=r"\footnotesize",
          note="Settings chosen on validation (pAUC up to 5\\% FPR). Caught: family-weighted share of malware "
               "flagged at a 1\\% false-positive rate. Connected / isolated: releases with / without dependency "
               "relations to the normal training releases. GuardDog is ranked by its number of findings plus "
               "the flag for the ROC AUC; as a binary rule it has a single operating point (its own FPR, "
               "Table~\\ref{tab:operating}), so no caught value at 1\\% FPR is given. Scores of the 3 seeds are rank-averaged (ensemble); the "
               "One-Class SVM is deterministic, so its seeds coincide. The supervised reference uses labelled malware and is not "
               "comparable to the unsupervised detectors.")


def t_baselines_contam():
    rows = []
    for det, lab in [("IF", "Isolation Forest"), ("LOF", "LOF"), ("OCSVM", "One-Class SVM"),
                     ("OCSVM (subgroup-calibrated)", "One-Class SVM, per-group")]:
        cells = [lab]
        for lv in [0.0, 0.01, 0.02, 0.05]:
            r = baseline(f"{det} ({GD})", lv)
            cells.append(f"{f3(r['auc_fw'])} / {pct(r['tpr1_fw_seed_mean'])}" if r is not None else "--")
        rows.append(cells)
    table("tab_baselines_contam", "Feature-based detectors under contamination: ROC-AUC / caught at 1\\% FPR (\\%, "
          "mean over 3 contamination draws).", "tab:baselines-contam",
          ["Detector", "0\\%", "1\\%", "2\\%", "5\\%"], rows, "lcccc",
          note="All features including GuardDog. Contamination: share of unlabelled malware in the training set.")


def t_cola():
    d = read("cola_results")
    if d is None:
        return
    rows = []
    for _, r in d[~d["method"].str.contains("calibrated")].iterrows():
        rows.append([pct(float(r["level"]), 0) + "\\%", esc(r["method"]), ci3(r), pct(r["tpr1_fw_seed_mean"]),
                     f3(r["sg_auc_fw_conn"]), f3(r["sg_auc_fw_iso"])])
    table("tab_cola", "CoLA (re-implemented, inductive protocol): ROC-AUC below 0.5 means malware is ranked as "
          "more normal than normal packages.", "tab:cola",
          ["Contam.", "Variant", "ROC-AUC [95\\% CI]", "Caught (\\%)", "Connected AUC", "Isolated AUC"],
          rows, "llcccc")


def t_contrastive_tuned():
    rows = []
    for lv in [0.0, 0.01]:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Contamination {pct(lv, 0)}\\%}}}} \\\\")
        for L, Ln in LOSSES:
            for E, En in ENCODERS:
                r = chosen_row(f"_{L}_{E}_tuned", lv)
                if r is not None:
                    rows.append([Ln, En, r["config"], f3(r["auc_fw"]), caught_sd(r),
                                 f3(r["sg_auc_fw_conn"]), f3(r["sg_auc_fw_iso"])])
        ref = baseline(f"OCSVM ({GD})", lv)
        rows.append(["\\multicolumn{3}{l}{Raw One-Class SVM (reference)}", f3(ref["auc_fw"]), caught_sd(ref),
                     f3(ref["sg_auc_fw_conn"]), f3(ref["sg_auc_fw_iso"])])
    table("tab_contrastive_tuned", "Contrastive encoders with tuned training settings (validation-chosen depth and "
          "scorer).", "tab:contrastive-tuned",
          ["Loss", "Encoder", "Chosen", "ROC-AUC", "Caught (\\%)", "Conn. AUC", "Iso. AUC"], rows, "lllcccc",
          size=r"\footnotesize",
          note="Caught: mean $\\pm$ sd over 3 training seeds. ROC-AUC: rank-averaged seed ensemble.")


def t_capacity():
    rows = []
    for L, Ln, big in [("infonce", "InfoNCE", 128), ("triplet", "Triplet", 256), ("dgi", "DGI", 256)]:
        d = read(f"contrastive_results_{L}_gin_tuned")
        dd = read(f"contrastive_results_{L}_d0_dim{big}")
        if d is None or dd is None:
            continue
        for lab, r in [("No graph, tuned size", method_row(d, 0.0, "OCSVM", 0)),
                       (f"No graph, size {big}", method_row(dd, 0.0, "OCSVM", 0)),
                       ("GIN depth 1, tuned size", method_row(d, 0.0, "OCSVM", 1))]:
            rows.append([Ln, lab, f3(r["auc_fw"]), pct(r["tpr1_fw_seed_mean"]), f3(r["sg_auc_fw_conn"]),
                         f3(r["sg_auc_fw_iso"])])
        rows.append(r"\addlinespace")
    ref = baseline(f"OCSVM ({GD})", 0.0)
    rows.append(["\\multicolumn{2}{l}{Raw One-Class SVM}", f3(ref["auc_fw"]), pct(ref["tpr1_fw_seed_mean"]),
                 f3(ref["sg_auc_fw_conn"]), f3(ref["sg_auc_fw_iso"])])
    table("tab_capacity", "Capacity check: depth-1 embeddings are twice as large (own-feature projection + "
          "neighbourhood). The no-graph model is also trained at the same total size, to separate the effect "
          "of the graph from that of the embedding size.",
          "tab:capacity", ["Loss", "Model (One-Class SVM scorer)", "ROC-AUC", "Caught (\\%)", "Connected AUC",
                           "Isolated AUC"], rows, "llcccc",
          note="Clean training, diagnostic runs (not chosen on validation). Isolated packages have no "
               "neighbours, so a gain on them cannot come from the graph.")


def t_newloss():
    for lv, lab in [(0.0, "clean"), (0.01, "1pct")]:
        rows = []
        for L, Ln in LOSSES:
            for E, En in ENCODERS:
                cells = [Ln, En]
                for v, _ in VARIANTS:
                    r = chosen_row(f"_{L}_{E}_{v}", lv)
                    cells.append(f"{f3(r['auc_fw'])} / {pct(r['tpr1_fw_seed_mean'])}" if r is not None else "--")
                rows.append(cells)
            rows.append(r"\addlinespace")
        ref = [baseline(f"OCSVM ({GD})", lv), baseline(f"OCSVM (subgroup-calibrated) ({GD})", lv),
               baseline(f"IF ({GD})", lv)]
        note = ("Each cell: ROC-AUC / caught at 1\\% FPR (\\%, mean over 3 seeds); validation-chosen depth and "
                "scorer. N1: negatives from the anchor's connectivity group only; N2: rarity-shifted negatives "
                "only; N3: both (full loss); N4: both, shifted feature chosen among all varying features. "
                f"Raw detectors at this level: One-Class SVM {f3(ref[0]['auc_fw'])} / {pct(ref[0]['tpr1_fw_seed_mean'])}, "
                f"with per-group thresholds {f3(ref[1]['auc_fw'])} / {pct(ref[1]['tpr1_fw_seed_mean'])}, "
                f"Isolation Forest {f3(ref[2]['auc_fw'])} / {pct(ref[2]['tpr1_fw_seed_mean'])}.")
        title = "clean training" if lv == 0 else "1\\% contamination"
        table(f"tab_newloss_{lab}", f"Proposed loss and ablation conditions, {title}.",
              f"tab:newloss-{lab}", ["Loss", "Encoder"] + [v for _, v in VARIANTS], rows, "llccccc",
              note=note, size=r"\footnotesize")


def t_comparisons(fname, name, caption, label, note):
    d = read(fname)
    if d is None:
        return
    rows = []
    for fam, g in d.groupby("family", sort=False):
        fam_lab = {"primary": "Primary (pre-registered)", "ablation": "Ablation (pre-registered)",
                   "expl_vs_base": "Proposed loss vs its base", "expl_vs_raw": "Proposed loss vs raw-feature detectors",
                   "expl_fpr05": "Proposed loss vs raw-feature detectors, caught at 0.5% FPR",
                   "expl_fpr2": "Proposed loss vs raw-feature detectors, caught at 2% FPR"}.get(fam, fam)
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{esc(fam_lab)} ({len(g)} tests, Holm)}}}} \\\\")
        for r in g.itertuples():
            pctm = str(r.metric).startswith("tpr")
            fmt = (lambda v: f"{100 * v:+.1f}") if pctm else (lambda v: f"{v:+.3f}")
            val = (lambda v: f"{100 * v:.1f}") if pctm else (lambda v: f"{v:.3f}")
            sig = r"$^{*}$" if r.significant else ""
            q = r.question
            for a, b in [("+ gat", "+ GAT"), ("+ tconv", "+ TransformerConv"), ("+ gps", "+ GraphGPS"),
                         ("RQ2: ", ""), ("RQ3: ", ""), (" (chosen on validation)", "")]:
                q = q.replace(a, b)
            rows.append([r.id, esc(q), "caught" if pctm else "AUC",
                         f"{val(r.a_value)} vs {val(r.b_value)}",
                         f"{fmt(r.diff)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]", f"{r.p_holm:.3f}{sig}"])
    table(name, caption, label, ["ID", "Comparison (a vs b)", "Metric", "a vs b", "Difference [95\\% CI]",
                                 "Holm $p$"], rows, "lp{6.2cm}lccc", note=note, size=r"\scriptsize")


def t_contamination():
    rows = []
    for lv in [0.0, 0.01, 0.02, 0.05]:
        sfx = "" if lv <= 0.01 else "_c25"
        a, b = chosen_row(f"_triplet_gin_tuned{sfx}", lv), chosen_row(f"_triplet_gin_n3{sfx}", lv)
        ref = [baseline(f"OCSVM ({GD})", lv), baseline(f"OCSVM (subgroup-calibrated) ({GD})", lv),
               baseline(f"IF ({GD})", lv)]
        rows.append([pct(lv, 0) + "\\%", caught_sd(a), caught_sd(b)] + [pct(r["tpr1_fw_seed_mean"]) for r in ref]
                    + [f"{f3(a['auc_fw'])} / {f3(b['auc_fw'])}"])
    table("tab_contamination", "Caught at 1\\% FPR (\\%) as contamination grows: Triplet + GIN with and without "
          "the proposed loss, and the raw-feature detectors.", "tab:contamination",
          ["Contam.", "Triplet tuned", "Triplet N3", "OCSVM", "OCSVM per-group", "IF",
           "AUC tuned / N3"], rows, "lcccccc", size=r"\footnotesize",
          note="OCSVM, OCSVM per-group, IF: raw-feature detectors. Mean $\\pm$ sd over 3 seeds (each seed = a different contamination draw at levels above 0). "
               "Paired tests: Table~\\ref{tab:exploratory}.")


def t_settings():
    t, s = read("tuning_choice"), read("sr_choice")
    rows = []
    for L, Ln in LOSSES:
        a, b = t[t["loss"] == L].iloc[0], s[s["loss"] == L].iloc[0]
        rows.append([Ln, str(int(a["epochs"])), str(int(a["dim"])), f"{a['lr']:g}", f3(a["val_pauc_mean"]),
                     f"{pct(b['nu'], 0)}\\%", str(int(b["k"])), "--" if L == "dgi" else f"{b['weight']:g}",
                     f3(b["val_pauc_mean"])])
    table("tab_settings", "Settings chosen on validation (mean pAUC up to 5\\% FPR over 3 seeds, One-Class SVM scorer, "
          "GIN depth 1, clean training).", "tab:settings",
          ["Loss", "Epochs", "Size", "LR", "Val. pAUC", "$\\nu$", "$K$", "$\\lambda$ / $\\beta$",
           "Val. pAUC (N3)"], rows, "lcccccccc",
          note="Left: training settings of the base losses (grid of 18 per loss). Right: settings of the proposed "
               "loss (grid of 18 for InfoNCE / Triplet, 6 for DGI), used for N1--N4. Size: width of each part of "
               "the embedding; at depth $\\geq 1$ the own-feature projection and the graph output are "
               "concatenated, so the embedding is twice this size.")


SENS_EXCLUDED = {}   # tag -> features without a defined gain (filled by sens_summary)


def sens_summary(tag):
    d = read(f"sensitivity_check{tag}")
    if d is None:
        return None
    m = (d.groupby(["depth", "feature"])[["variability", "separation", "gain_trained", "ratio"]].mean()
         .reset_index())
    undef = m[(m["depth"] == 1) & m["gain_trained"].isna() & (m["feature"] != "is_context")]["feature"]
    SENS_EXCLUDED[tag] = sorted(undef)
    m = m.dropna(subset=["gain_trained"])
    m = m[(m["feature"] != "is_context") & (m["depth"] == 1)]
    m["group"] = pd.cut(m["variability"], [-0.01, 0.05, 0.25, 1.0], labels=["almost constant", "rare", "common"])
    return m


def t_sensitivity():
    base, new = sens_summary("_triplet_gin_tuned"), sens_summary("_triplet_gin_n3")
    if base is None or new is None:
        return
    rows = []
    for g, lab in [("almost constant", "Almost constant ($<$5\\%)"), ("rare", "Rare (5--25\\%)"),
                   ("common", "Common ($>$25\\%)")]:
        a, b = base[base["group"] == g], new[new["group"] == g]
        rows.append([lab, str(len(a)), f3(a["gain_trained"].mean()), f"{a['ratio'].mean():.2f}",
                     f3(b["gain_trained"].mean()), f"{b['ratio'].mean():.2f}"])
    rc = lambda m, y: m[["variability", y]].corr(method="spearman").iloc[0, 1]  # noqa: E731
    rows.append(r"\midrule")
    rows.append(["Rank corr. ratio vs variability", "", "", f"{rc(base, 'ratio'):+.2f}", "", f"{rc(new, 'ratio'):+.2f}"])
    rows.append(["Rank corr. gain vs separation", "", f"{base[['separation', 'gain_trained']].corr(method='spearman').iloc[0, 1]:+.2f}",
                 "", f"{new[['separation', 'gain_trained']].corr(method='spearman').iloc[0, 1]:+.2f}", ""])
    table("tab_sensitivity", "Mechanism check: how strongly the encoder reacts to each feature, by how much the "
          "feature varies among normal training packages (Triplet + GIN depth 1, validation).",
          "tab:sensitivity", ["Feature group", "Features", "Gain (base)", "Ratio (base)", "Gain (N3)",
                              "Ratio (N3)"], rows, "lccccc",
          note="Gain: embedding movement per unit feature change (malware-like values), normalised by the "
               "embedding spread. Ratio: trained / untrained gain. Means over features and 3 seeds. "
               + sens_excluded_note())


def sens_excluded_note():
    """Which non-indicator features are missing from the table, and why."""
    und = sorted(set(SENS_EXCLUDED.get("_triplet_gin_tuned", [])) | set(SENS_EXCLUDED.get("_triplet_gin_n3", [])))
    txt = "The context indicator \\texttt{is\\_context} is not a release feature and is excluded."
    if und:
        names = ", ".join(f"\\texttt{{{esc(f)}}}" for f in und)
        txt += (f" Also excluded: {names}, whose gain is undefined because the malware values drawn never "
                f"differ from the normal ones (almost every release has the same value).")
    return txt


def t_label_noise():
    d = read("label_noise_summary")
    if d is None:
        return
    lab = {"research": "Self-declared research / proof of concept", "ctf": "Capture-the-flag material",
           "self_declared": "Self-declared (research or CTF)", "callback": "Calls an out-of-band testing service",
           "test_name": "Test / demo package name", "any": "Any of the above"}
    rows = []
    c = d[d["section"] == "candidates"].set_index("category")
    t = d[(d["section"] == "by_split") & (d["group"] == "test")].set_index("category")
    for k in ["research", "ctf", "self_declared", "callback", "test_name", "any"]:
        if k in c.index:
            r = c.loc[k]
            tt = f"{int(t.loc[k, 'packages'])} ({pct(t.loc[k, 'share'])}\\%)" if k in t.index else "--"
            rows.append([lab[k], f"{int(r['packages'])} ({pct(r['share'])}\\%)", str(int(r["families"])), tt])
    cg = d[d["section"] == "caught_test_clean"]
    if len(cg):
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{4}{l}{\textit{Caught at 1\% FPR on test, clean training (\%)}} \\")
        for m, g in cg.groupby("category", sort=False):
            v = g.set_index("group")["caught"]
            rows.append([esc(m), f"self-declared {pct(v['self_declared'])} vs rest {pct(v['rest_of_self_declared'])}",
                         "", f"all {pct(v['all_test_malware_fw'])}, without probes {pct(v['without_self_declared_fw'])}"])
    table("tab_label_noise", "Label noise: malicious packages that are security-research probes, CTF material or "
          "test uploads (text-only scan of every malicious archive).", "tab:label-noise",
          ["Candidate type", "Malicious packages", "Families", "In test set"], rows, "p{5.4cm}ccc",
          note="Matches are candidates (keyword and domain patterns); a random sample was checked by hand. Callback "
               "services are also used by real malware, so that row is an indicator, not proof. Caught rates: "
               "unweighted for self-declared vs the rest; family-weighted for all test malware with and without "
               "the self-declared probes.", size=r"\footnotesize")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
PI_PROJECT = 1 / 65


def precision(tpr, fpr, pi=PI_PROJECT):
    """Precision at a base rate pi from a detection rate and a false-positive rate."""
    return tpr * pi / (tpr * pi + fpr * (1 - pi))


def t_operating():
    """GuardDog at its own operating point next to the score-based detectors at fixed FPRs."""
    gd = baseline("GuardDog rule", None)
    # GuardDog's own FPR on the test normals, recovered exactly from the stored per-release precision
    p, t = gd["prec@ownFPR_per_project_1:65"], gd["tpr_gd"]
    gd_fpr = t * PI_PROJECT * (1 - p) / (p * (1 - PI_PROJECT))
    rows = [r"\multicolumn{8}{l}{\textit{Clean training}} \\",
            ["GuardDog (findings + flag)", f3(gd["auc_fw"]), "--", "--", "--",
             f"{pct(gd['tpr_gd_fw'])} (FPR {pct(gd_fpr)})", "0.0",
             f"{pct(precision(gd['tpr_gd_fw'], gd_fpr))} (own FPR)"]]

    def row(lab, r):
        return [lab, f3(r["auc_fw"]), pct(r["tpr05_fw"]), pct(r["tpr1_fw"]), pct(r["tpr2_fw"]),
                pct(r["tpr_gd_fw"]), pct(r["tpr1_hard_fw"]), pct(r["prec_fw@1%FPR_per_project_1:65"])]

    for lv in [0.0, 0.01]:
        if lv:
            rows.append(r"\midrule")
            rows.append(r"\multicolumn{8}{l}{\textit{1\% contamination}} \\")
        oc = baseline(f"OCSVM ({GD})", lv)
        if oc is not None:
            rows.append(row("One-Class SVM, raw features", oc))
        for tag, lab in [("_triplet_gin_tuned", "Triplet + GIN, tuned base"),
                         ("_triplet_gin_n3", "Triplet + GIN, N3")]:
            r = chosen_row(tag, lv)
            if r is not None:
                rows.append(row(f"{lab} [{r['config']}]", r))
    table("tab_operating", "Operating points (test set, family-weighted, \\%): detection at fixed false-positive "
          "rates, at GuardDog's own false-positive rate, on GuardDog-hard malware, and precision at a base rate "
          "of one malicious project in 65.", "tab:operating",
          ["Detector", "ROC-AUC", "0.5\\%", "1\\%", "2\\%", "At GD FPR", "Hard, 1\\%", "Prec. 1:65"],
          rows, "lccccccc", size=r"\footnotesize",
          note="0.5\\%, 1\\%, 2\\%: share of malware caught at that false-positive rate. At GD FPR: caught at "
               "GuardDog's own false-positive rate. Hard: malware on which no GuardDog rule fires, caught at 1\\% "
               "FPR. Prec. 1:65: precision at 1\\% FPR (GuardDog: at its own FPR) for one malicious project in 65 "
               "(PyPI 2025). Contrastive models: validation-chosen depth and scorer, ensemble of 3 seeds (seed means: "
               "Table~\\ref{tab:operating-grid}). All thresholds are set on the normal test releases, so every "
               "detector operates at exactly the stated rate; thresholds fixed in advance: Table~\\ref{tab:deployment}.")


def t_svdd():
    """Deep SVDD control: same encoders with a one-class objective (preliminary runs, all columns rescaled)."""
    def get(run, method, lv):
        d = read(run)
        if d is None:
            return None
        m = d[(d["method"] == method) & np.isclose(pd.to_numeric(d["level"], errors="coerce"), lv)]
        return m.iloc[0] if len(m) else None

    def cells(run, method):
        r0, r1 = get(run, method, 0.0), get(run, method, 0.01)
        if r0 is None:
            return None
        return [f3(r0["auc_fw"]), caught_sd(r0), f3(r0["sg_auc_fw_conn"]), f3(r0["sg_auc_fw_iso"]),
                f3(r1["auc_fw"]) if r1 is not None else "--", caught_sd(r1) if r1 is not None else "--",
                f3(r1["sg_auc_fw_conn"]) if r1 is not None else "--"]

    spec = [("No graph (depth 0)", "ocgnn_results_gin_all", "Deep SVDD (no graph)")]
    for enc, run, name in [("GIN", "ocgnn_results_ginmean_all", "OC-GIN"), ("GAT", "ocgnn_results_gat_all", "OC-GAT"),
                           ("TransformerConv", "ocgnn_results_tconv_all", "OC-TransformerConv"),
                           ("GraphGPS", "ocgnn_results_gps_all", "OC-GPS")]:
        for dep in [1, 2]:
            spec.append((f"{enc}, depth {dep}", run, f"{name}, depth {dep}"))
    rows = []
    for lab, run, method in spec:
        c = cells(run, method)
        if c is not None:
            rows.append([lab] + c)
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{l}{\textit{GIN with sum aggregation (preliminary comparison)}} \\")
    for dep in [1, 2]:
        c = cells("ocgnn_results_gin_all", f"OC-GIN, depth {dep}")
        if c is not None:
            rows.append([f"GIN sum, depth {dep}"] + c)
    table("tab_svdd", "Control with a one-class objective: Deep SVDD on the same encoders (test set, "
          "family-weighted).", "tab:svdd",
          ["Encoder", "ROC-AUC", "Caught (\\%)", "Conn. AUC", "Iso. AUC", "AUC 1\\%", "Caught 1\\% (\\%)",
           "Conn. 1\\%"],
          rows, "lccccccc", size=r"\footnotesize",
          note="Clean training unless marked 1\\% (1\\% contamination); Conn. 1\\%: ROC AUC on connected "
               "releases at 1\\% contamination. Caught: share of malware flagged at 1\\% "
               "FPR, mean $\\pm$ standard deviation over 3 seeds; ROC-AUC of the rank-averaged ensemble. Plain "
               "scores. All encoders use mean aggregation except the last two rows, which differ from the GIN rows "
               "above only in summing the neighbours.")


# ---------------------------------------------------------------------------
# Revision (Oct 2026): operating points, thresholds fixed in advance, models without GuardDog
# (results/operating_points.csv from src/operating_points.py)
# ---------------------------------------------------------------------------
OPS_METHODS = [  # (label, method, calibrated)
    ("One-Class SVM", "One-Class SVM", False),
    ("One-Class SVM, per-group", "One-Class SVM", True),
    ("Isolation Forest", "Isolation Forest", False),
    ("Triplet + GIN, base", "Triplet + GIN, base", False),
    ("Triplet + GIN, base, per-group", "Triplet + GIN, base", True),
    ("Triplet + GIN, N3", "Triplet + GIN, N3", False),
    ("Triplet + GIN, N3, per-group", "Triplet + GIN, N3", True),
]


def group_row(lv, ncol=7):
    lab = "Clean training" if lv == 0 else f"{int(round(100 * lv))}\\% contamination"
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{lab}}}}} \\\\"


def ops_value(d, level, method, calibrated, metric, alpha, guarddog=True, field="mean"):
    m = d[np.isclose(d["level"].astype(float), level) & (d["method"] == method)
          & (d["calibrated"].astype(str).str.lower() == str(calibrated).lower())
          & (d["guarddog"].astype(str).str.lower() == str(guarddog).lower())
          & (d["metric"] == metric) & np.isclose(d["alpha"].astype(float), alpha)]
    return m[field].iloc[0] if len(m) else None


def t_operating_grid():
    d = read("operating_points")
    if d is None:
        return
    rows = []
    for i, lv in enumerate([0.0, 0.01, 0.02, 0.05]):
        if i:
            rows.append(r"\midrule")
        rows.append(group_row(lv))
        for lab, meth, cal in OPS_METHODS:
            vals = [ops_value(d, lv, meth, cal, "caught", a) for a in (0.005, 0.01, 0.02, 0.05)]
            if all(v is None for v in vals):
                continue
            rows.append([lab] + [pct(v) for v in vals]
                        + [pct(ops_value(d, lv, meth, cal, "caught_hard", 0.01)),
                           f3(ops_value(d, lv, meth, cal, "auc", 0.0))])
    table("tab_operating_grid", "Share of malware caught (\\%) at four false-positive rates, by contamination "
          "level: raw-feature detectors and Triplet + GIN with and without the proposed loss, with plain and "
          "per-group calibrated scores.", "tab:operating-grid",
          ["Detector", "0.5\\%", "1\\%", "2\\%", "5\\%", "Hard, 1\\%", "ROC-AUC"], rows, "lcccccc",
          size=r"\footnotesize",
          note="Test set, family-weighted; mean over 3 seeds (at levels above 0, each seed is a different "
               "contamination draw). Threshold: the (1 $-$ FPR) quantile of the normal test scores. Hard: malware "
               "on which no GuardDog rule fires. Per-group: scores calibrated within the subgroups with and without "
               "dependency peers (Equation~\\ref{eq:calibration}). Contrastive models: depth and scorer chosen on "
               "validation; all detectors use the GuardDog features. At 0.5\\% FPR the threshold rests on about "
               "23 normal test releases.")


def t_deployment():
    d = read("operating_points")
    if d is None:
        return
    rows = []
    for i, lv in enumerate([0.0, 0.01]):
        if i:
            rows.append(r"\midrule")
        rows.append(group_row(lv))
        for lab, meth, cal in OPS_METHODS:
            if ops_value(d, lv, meth, cal, "caught", 0.01) is None:
                continue
            cells = [lab]
            for a in (0.005, 0.01):
                cells += [pct(ops_value(d, lv, meth, cal, "deploy_fpr", a), 2),
                          pct(ops_value(d, lv, meth, cal, "deploy_caught", a))]
            cells += [pct(ops_value(d, lv, meth, cal, "caught", 0.01)),
                      pct(ops_value(d, lv, meth, cal, "deploy_caught_hard", 0.01))]
            rows.append(cells)
    table("tab_deployment", "Thresholds fixed in advance: the threshold is set on the normal validation releases "
          "for a target false-positive rate and applied unchanged to the test set.", "tab:deployment",
          ["Detector", "0.5\\%: actual FPR", "0.5\\%: caught", "1\\%: actual FPR", "1\\%: caught",
           "1\\%: caught, test threshold", "1\\%: hard"], rows, "lcccccc", size=r"\footnotesize",
          note="0.5\\% and 1\\%: target false-positive rate. Actual FPR: share of normal test releases above the threshold. Caught: family-weighted share of "
               "test malware above it; Hard: the same on malware no GuardDog rule detects. Caught, test threshold: "
               "the convention of the other tables, with the threshold set on the normal test releases. Mean over "
               "3 seeds. About 30 normal validation releases lie above the 1\\% threshold.")


def t_nogd():
    d = read("operating_points")
    if d is None:
        return
    meths = [("One-Class SVM", "One-Class SVM", False), ("One-Class SVM, per-group", "One-Class SVM", True),
             ("Isolation Forest", "Isolation Forest", False), ("Triplet + GIN, base", "Triplet + GIN, base", False),
             ("Triplet + GIN, N3", "Triplet + GIN, N3", False)]
    rows = []
    for i, lv in enumerate([0.0, 0.01]):
        if i:
            rows.append(r"\midrule")
        rows.append(group_row(lv))
        for lab, meth, cal in meths:
            cells = [lab]
            for metric, alpha in [("auc", 0.0), ("caught", 0.01), ("caught_hard", 0.01)]:
                for gd in (True, False):
                    v = ops_value(d, lv, meth, cal, metric, alpha, guarddog=gd)
                    cells.append(f3(v) if metric == "auc" else pct(v))
            if any(c != "--" for c in cells[1:]):
                rows.append(cells)
    table("tab_nogd", "With and without the GuardDog features: ROC-AUC, share caught at 1\\% FPR, and share caught "
          "of the malware no GuardDog rule detects.", "tab:nogd",
          ["Detector", "AUC, with", "AUC, without", "Caught, with", "Caught, without", "Hard, with",
           "Hard, without"], rows, "lcccccc", size=r"\footnotesize",
          note="With / without: all features, or all features except the GuardDog counts (code issues and the five "
               "rule groups). The contrastive models without GuardDog reuse the training settings chosen with it; "
               "depth and scorer are chosen on validation. Test set, family-weighted, mean over 3 seeds.")


def t_hard_ops():
    """GuardDog-hard malware at four false-positive rates, with and without the GuardDog features
    (revision, point 5: what the detectors catch where no rule fires, across operating points)."""
    d = read("operating_points")
    if d is None:
        return
    meths = [("One-Class SVM", "One-Class SVM", False), ("One-Class SVM, per-group", "One-Class SVM", True),
             ("Isolation Forest", "Isolation Forest", False), ("Triplet + GIN, base", "Triplet + GIN, base", False),
             ("Triplet + GIN, N3", "Triplet + GIN, N3", False)]
    alphas = (0.005, 0.01, 0.02, 0.05)
    rows = []
    for i, lv in enumerate([0.0, 0.01]):
        if i:
            rows.append(r"\midrule")
        rows.append(group_row(lv, ncol=9))
        for lab, meth, cal in meths:
            cells = [lab] + [pct(ops_value(d, lv, meth, cal, "caught_hard", a, guarddog=gd))
                             for gd in (True, False) for a in alphas]
            if any(c != "--" for c in cells[1:]):
                rows.append(cells)
    table("tab_hard_ops", "Malware on which no GuardDog rule fires: share caught (\\%) at four false-positive "
          "rates, with and without the GuardDog features.", "tab:hard-ops",
          [r"& \multicolumn{4}{c}{With GuardDog features} & \multicolumn{4}{c}{Without GuardDog features} \\"
           r" \cmidrule(lr){2-5}\cmidrule(lr){6-9} Detector"] + [f"{a * 100:g}\\%" for a in alphas] * 2,
          rows, "lcccccccc", size=r"\footnotesize",
          note="Family-weighted share of the GuardDog-hard test malware above the threshold; the threshold is the "
               "(1 $-$ FPR) quantile of all normal test scores, as in Table~\\ref{tab:operating-grid}. Mean over 3 "
               "seeds. Contrastive models without GuardDog reuse the training settings chosen with it.")


def f_fpr_curves():
    import matplotlib.pyplot as plt
    d = read("operating_points")
    if d is None:
        return
    series = [("Triplet + GIN, N3", "Triplet + GIN, N3", False, C["blue"], "-"),
              ("Triplet + GIN, N3, per-group", "Triplet + GIN, N3", True, C["blue"], ":"),
              ("Triplet + GIN, base", "Triplet + GIN, base", False, C["orange"], "-"),
              ("OCSVM, per-group", "One-Class SVM", True, C["aqua"], "--"),
              ("OCSVM", "One-Class SVM", False, C["magenta"], "--"),
              ("Isolation Forest", "Isolation Forest", False, C["yellow"], "--")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, lv in zip(axes, [0.0, 0.01]):
        for lab, meth, cal, col, ls in series:
            m = d[np.isclose(d["level"].astype(float), lv) & (d["method"] == meth)
                  & (d["calibrated"].astype(str).str.lower() == str(cal).lower())
                  & (d["guarddog"].astype(str).str.lower() == "true") & (d["metric"] == "caught")]
            if not len(m):
                continue
            m = m.sort_values("alpha")
            x, y, e = 100 * m["alpha"].values, 100 * m["mean"].values, 100 * m["sd"].fillna(0).values
            ax.plot(x, y, color=col, linestyle=ls, linewidth=2 if ls == "-" else 1.4, label=lab)
            if ls == "-":
                ax.fill_between(x, y - e, y + e, color=col, alpha=0.12, linewidth=0)
        ax.axvline(1.0, color=C["ink2"], linewidth=0.8, linestyle=":")
        ax.axvspan(0.2, 0.5, color=C["grid"], alpha=0.5, linewidth=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
        ax.minorticks_off()
        ax.set_xlim(0.2, 5)
        ax.set_title("Clean training" if lv == 0 else "1% contamination", loc="left")
        ax.set_xlabel("False-positive rate (%, log scale)")
    axes[0].set_ylabel("Caught (%)")
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_fpr_curves")
    plt.close(fig)


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.edgecolor": C["muted"],
        "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"], "text.color": C["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 200})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/figures/{name}.{ext}")
    FIGS.append(name)


def f_contamination():
    import matplotlib.pyplot as plt
    levels = [0.0, 0.01, 0.02, 0.05]
    series = []
    tun = [chosen_row(f"_triplet_gin_tuned{'' if lv <= 0.01 else '_c25'}", lv) for lv in levels]
    n3 = [chosen_row(f"_triplet_gin_n3{'' if lv <= 0.01 else '_c25'}", lv) for lv in levels]
    series.append(("Triplet + proposed loss (N3)", n3, C["blue"], "o", True))
    series.append(("Triplet, tuned base", tun, C["orange"], "s", True))
    series.append(("Raw OCSVM, per-group thresholds", [baseline(f"OCSVM (subgroup-calibrated) ({GD})", lv)
                                                       for lv in levels], C["aqua"], "^", False))
    series.append(("Raw Isolation Forest", [baseline(f"IF ({GD})", lv) for lv in levels], C["yellow"], "D", False))
    series.append(("Raw OCSVM", [baseline(f"OCSVM ({GD})", lv) for lv in levels], C["magenta"], "v", False))
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    x = np.arange(len(levels))
    for lab, rows, col, mk, sd in series:
        y = np.array([100 * r["tpr1_fw_seed_mean"] for r in rows])
        ax.plot(x, y, color=col, marker=mk, markersize=5.5, markeredgecolor="white", markeredgewidth=1.2,
                label=lab, zorder=3 if sd else 2, linestyle="-" if sd else "--", linewidth=2 if sd else 1.4)
        if sd:
            e = np.array([100 * r["tpr1_fw_seed_std"] for r in rows])
            ax.fill_between(x, y - e, y + e, color=col, alpha=0.12, linewidth=0)
    ax.set_xticks(x, [f"{int(100 * lv)}%" for lv in levels])
    ax.set_xlabel("Malware hidden in the training set (contamination)")
    ax.set_ylabel("Caught at 1% FPR (%)")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig_contamination")
    plt.close(fig)


def f_sensitivity():
    import matplotlib.pyplot as plt
    base, new = sens_summary("_triplet_gin_tuned"), sens_summary("_triplet_gin_n3")
    if base is None or new is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
    for ax, m, title, col in [(axes[0], base, "Tuned Triplet (base)", C["orange"]),
                              (axes[1], new, "With the proposed loss (N3)", C["blue"])]:
        v = m["variability"].clip(lower=1e-4)
        ax.scatter(v, m["ratio"], s=22, color=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xscale("log")
        ax.axvspan(1e-4, 0.05, color=C["grid"], alpha=0.45, linewidth=0, zorder=0)
        rho = m[["variability", "ratio"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{title}\nrank corr. {rho:+.2f}", loc="left")
    fig.supxlabel("Share of training packages not at the feature's common value (log scale)", fontsize=8.5,
                  color=C["ink2"], y=-0.02)
    axes[0].set_ylabel("Response to the feature\n(trained / untrained)")
    axes[0].text(1.2e-4, axes[0].get_ylim()[1] * 0.97, "almost constant", fontsize=7, color=C["ink2"], va="top")
    save(fig, "fig_sensitivity")
    plt.close(fig)


def f_learning_curve():
    import matplotlib.pyplot as plt
    raw = read("learning_curve_raw")
    if raw is None:
        return
    fr = [0.10, 0.25, 0.50, 1.00]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for lab, meth, col, mk in [("Raw OCSVM", "OCSVM, raw features", C["aqua"], "^"),
                               ("Raw Isolation Forest", "Isolation Forest, raw features", C["yellow"], "D")]:
        y = [raw[(raw["method"] == meth) & np.isclose(raw["train_frac"], f)]["auc_fw"].iloc[0] for f in fr]
        ax.plot(fr, y, color=col, marker=mk, markersize=5.5, markeredgecolor="white", label=lab, linestyle="--",
                linewidth=1.4)
    for lab, dep, col, mk in [("InfoNCE + GIN, depth 1", 1, C["blue"], "o"), ("InfoNCE, no graph", 0, C["orange"], "s")]:
        y = []
        for f in fr:
            d = read(f"contrastive_results_infonce_gin_lc{int(round(100 * f))}")
            r = method_row(d, 0.0, "OCSVM", dep) if d is not None else None
            y.append(r["auc_fw"] if r is not None else np.nan)
        ax.plot(fr, y, color=col, marker=mk, markersize=5.5, markeredgecolor="white", label=lab)
    ax.set_xscale("log")
    ax.set_xticks(fr, ["10%", "25%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Share of the clean training packages used")
    ax.set_ylabel("ROC-AUC (family-weighted)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "fig_learning_curve")
    plt.close(fig)


def f_newloss_encoders():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.9), sharey=True)
    cols = {"tuned": C["orange"], "n1": C["muted"], "n2": C["aqua"], "n3": C["blue"], "n4": C["yellow"]}
    width = 0.16
    for ax, (L, Ln) in zip(axes, LOSSES):
        x = np.arange(len(ENCODERS))
        for i, (v, vl) in enumerate(VARIANTS):
            y = []
            for E, _ in ENCODERS:
                r = chosen_row(f"_{L}_{E}_{v}", 0.01)
                y.append(100 * r["tpr1_fw_seed_mean"] if r is not None else np.nan)
            ax.bar(x + (i - 2) * width, y, width * 0.88, color=cols[v], label=vl, zorder=3)
        ax.set_xticks(x, [e for _, e in ENCODERS], rotation=30, ha="right")
        ax.set_title(Ln, loc="left")
        ax.grid(axis="x", visible=False)
    ref = baseline(f"OCSVM (subgroup-calibrated) ({GD})", 0.01)["tpr1_fw_seed_mean"] * 100
    for ax in axes:
        ax.axhline(ref, color=C["ink2"], linewidth=1, linestyle=":", zorder=4)
    axes[2].text(3.45, ref + 1.5, "raw OCSVM,\nper-group", fontsize=6.5, color=C["ink2"], ha="right", va="bottom")
    axes[0].set_ylabel("Caught at 1% FPR (%)\n1% contamination")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig_newloss_encoders")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    global RES, OUT, TABLES, FIGS
    ap = argparse.ArgumentParser(description="Tables and figures of the results chapter")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="thesis_out")
    args = ap.parse_args()
    RES, OUT, TABLES, FIGS = args.results, args.out, [], []
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    os.makedirs(f"{OUT}/figures", exist_ok=True)

    t_baselines()
    t_baselines_contam()
    t_cola()
    t_contrastive_tuned()
    t_capacity()
    t_settings()
    t_newloss()
    t_comparisons("paired_comparisons", "tab_primary", "Pre-registered comparisons (clean training, GIN). "
                  "Difference a $-$ b with paired two-level bootstrap 95\\% CI; Holm correction per family.",
                  "tab:primary", "Caught in percentage points. $^{*}$ significant at $\\alpha = 0.05$ after Holm. "
                  "Primary family: answers RQ2/RQ3; ablation family: N3 against N1, N2 and N4.")
    t_comparisons("paired_comparisons_expl", "tab_exploratory", "Exploratory comparisons under contamination "
                  "(specified after seeing the descriptive results; not pre-registered).", "tab:exploratory",
                  "Same procedure as Table~\\ref{tab:primary}. Raw detectors use the same contamination draws as "
                  "the contrastive runs.")
    t_contamination()
    t_sensitivity()
    t_label_noise()
    t_operating()
    t_svdd()
    # revision (Oct 2026)
    t_operating_grid()
    t_deployment()
    t_nogd()
    t_hard_ops()
    t_comparisons("paired_comparisons_ops", "tab_exploratory_ops", "Exploratory comparisons at the neighbouring "
                  "operating points 0.5\\% and 2\\% FPR (added in the revision; not pre-registered).",
                  "tab:exploratory-ops", "Same procedure and comparisons as Table~\\ref{tab:exploratory}, with the "
                  "threshold at 0.5\\% or 2\\% FPR.")

    style()
    f_contamination()
    f_sensitivity()
    f_learning_curve()
    f_newloss_encoders()
    f_fpr_curves()

    doc = [r"\documentclass[11pt]{article}", r"\usepackage[a4paper,margin=2cm]{geometry}",
           r"\usepackage{booktabs}", r"\usepackage{graphicx}", r"\usepackage{adjustbox}", r"\begin{document}"]
    doc += [f"\\input{{tables/{t}}}" for t in TABLES]
    doc += [f"\\begin{{figure}}[htbp]\\centering\\includegraphics[width=\\linewidth]{{figures/{f}.pdf}}"
            f"\\caption{{{esc(f)}}}\\end{{figure}}" for f in FIGS]
    doc.append(r"\end{document}")
    with open(f"{OUT}/all_tables.tex", "w") as fh:
        fh.write("\n".join(doc) + "\n")
    print(f"{len(TABLES)} tables, {len(FIGS)} figures -> {OUT}/")


if __name__ == "__main__":
    main()