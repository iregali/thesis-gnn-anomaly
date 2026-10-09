"""
select_tuning.py
Choose the training settings (epochs, embedding size, learning rate) per loss
from the tuning runs of slurm/tune_contrastive.sh, on VALIDATION only: highest
mean validation pAUC (<= 5% FPR) over the 3 seeds, One-Class SVM scorer, GIN
depth 1, clean training. Test results are printed for information only and
are not used to choose. (Methodology §37)

Usage
  python src/select_tuning.py
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import RESULTS_DIR  # noqa: E402

rows = []
for f in glob.glob(f"{RESULTS_DIR}/contrastive_valseeds_tune_*.csv"):
    m = re.search(r"_tune_(\w+?)_e(\d+)_d(\d+)_lr([\d.e-]+)\.csv$", f)
    if not m:
        continue
    loss, ep, dim, lr = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    v = pd.read_csv(f)
    v = v[(v["scorer"] == "OCSVM") & (v["depth"] == 1)]
    res = pd.read_csv(f.replace("contrastive_valseeds", "contrastive_results"))
    r = res[res["method"].str.contains("OCSVM") & ~res["method"].str.contains("calibrated|DGI-D")].iloc[0]
    rows.append({"loss": loss, "epochs": ep, "dim": dim, "lr": lr, "val_pauc_mean": v["val_pauc"].mean(),
                 "val_pauc_sd": v["val_pauc"].std(), "seeds": len(v),
                 "test_auc (info)": r["auc_fw"], "test_caught (info)": r["tpr1_fw_seed_mean"]})
if not rows:
    raise SystemExit("no tuning results found (contrastive_valseeds_tune_*.csv)")
t = pd.DataFrame(rows).sort_values(["loss", "val_pauc_mean"], ascending=[True, False])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(t.round(3).to_string(index=False))
best = t.groupby("loss").head(1)
print("\n=== Chosen on validation (default: epochs 50, dim 64, lr 0.001) ===")
print(best[["loss", "epochs", "dim", "lr", "val_pauc_mean", "val_pauc_sd"]].round(3).to_string(index=False))
best.to_csv(f"{RESULTS_DIR}/tuning_choice.csv", index=False)
print(f"\nSaved {RESULTS_DIR}/tuning_choice.csv")
