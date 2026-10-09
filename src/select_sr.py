"""
select_sr.py
Choose the settings of the proposed loss (nu, copies K, weight) per loss from
the runs of slurm/tune_sr.sh, on VALIDATION only: highest mean validation pAUC
(<= 5% FPR) over the 3 seeds, One-Class SVM scorer, GIN depth 1, clean
training, full loss (N3). Test results are printed for information only and
are not used to choose. (Methodology section 38; same rule as select_tuning.py)

Usage
  python src/select_sr.py            -> RESULTS_DIR/sr_choice.csv
"""

import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import RESULTS_DIR  # noqa: E402

rows = []
for f in glob.glob(f"{RESULTS_DIR}/contrastive_valseeds_srtune_*.csv"):
    m = re.search(r"_srtune_(\w+?)_nu([\d.]+)_k(\d+)_w([\d.]+)\.csv$", f)
    if not m:
        continue
    loss, nu, k, w = m.group(1), float(m.group(2)), int(m.group(3)), float(m.group(4))
    v = pd.read_csv(f)
    v = v[(v["scorer"] == "OCSVM") & (v["depth"] == 1)]
    res = pd.read_csv(f.replace("contrastive_valseeds", "contrastive_results"))
    r = res[res["method"].str.contains("OCSVM") & ~res["method"].str.contains("calibrated|DGI-D")].iloc[0]
    rows.append({"loss": loss, "nu": nu, "k": k, "weight": w, "val_pauc_mean": v["val_pauc"].mean(),
                 "val_pauc_sd": v["val_pauc"].std(), "seeds": len(v),
                 "test_auc (info)": r["auc_fw"], "test_caught (info)": r["tpr1_fw_seed_mean"]})
if not rows:
    raise SystemExit("no tuning results found (contrastive_valseeds_srtune_*.csv)")
t = pd.DataFrame(rows).sort_values(["loss", "val_pauc_mean"], ascending=[True, False])
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(t.round(3).to_string(index=False))
expected = {"infonce": 18, "triplet": 18, "dgi": 6}
for loss, n in expected.items():
    got = int((t["loss"] == loss).sum())
    if got != n:
        print(f"WARNING: {loss}: {got} of {n} tuning runs found (failed tasks?)")
best = t.groupby("loss").head(1)
print("\n=== Chosen on validation ===")
print(best[["loss", "nu", "k", "weight", "val_pauc_mean", "val_pauc_sd"]].round(3).to_string(index=False))
best.to_csv(f"{RESULTS_DIR}/sr_choice.csv", index=False)
print(f"\nSaved {RESULTS_DIR}/sr_choice.csv")
