"""
gnn_eval.py
Evaluation shared by the GNN scripts, identical to baseline_protocol.py:
same test packages, family weights, GuardDog-hard set, release-history groups,
connectivity subgroups, bootstrap, seed reporting and per-subgroup calibration.

  ev = Evaluator(boot=500)
  ev.val_paths, ev.test_paths       order in which scores must be given
  ev.row(level, name, per_seed_test_scores, setting)            -> result row
  ev.calibrated(val_scores, test_scores, fit_paths)             -> calibrated test scores
"""

import numpy as np
import pandas as pd

from baseline_protocol import (DATA_DIR, EXTRA_FPRS, SPLIT_FILE, bootstrap_ci, combine, load_features,
                               normal_groups, point_metrics, population_metrics, subgroup_metrics, to_ranks)
from connectivity import PeerIndex, connectivity_table, load_core_deps


class Evaluator:
    def __init__(self, boot=500):
        self.boot = boot
        feats, _ = load_features()
        self.feats = feats
        y_all = feats["label"].values
        splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
        role = feats["path"].map(dict(zip(splits["path"], splits["split"])))
        fam_all = feats["path"].map(dict(zip(splits["path"], splits["campaign_id"]))).fillna("").values
        w_all = feats["path"].map(dict(zip(splits["path"], splits["family_weight"].astype(float)))).values
        self.train_idx = np.where((role == "train") & (y_all == 0))[0]
        self.val_idx, self.test_idx = np.where(role == "val")[0], np.where(role == "test")[0]
        self.y_val, self.y = y_all[self.val_idx], y_all[self.test_idx]
        self.w, self.fam = w_all[self.test_idx], fam_all[self.test_idx]
        self.hard = (self.y == 1) & (feats["gd_flagged"].values[self.test_idx] == 0)
        self.gd_fpr = float(feats["gd_flagged"].values[self.test_idx][self.y == 0].mean())
        self.pop_groups, _ = normal_groups(feats, self.test_idx, self.y)
        self.deps_of = load_core_deps(DATA_DIR)
        conn = connectivity_table(feats, self.train_idx, self.test_idx, self.deps_of)
        self.sg = {"conn": conn["connected"].values, "iso": ~conn["connected"].values,
                   "peers": conn["has_peers"].values, "nopeers": ~conn["has_peers"].values}
        self.val_paths = feats["path"].values[self.val_idx]
        self.test_paths = feats["path"].values[self.test_idx]
        print(f"test {int((self.y == 0).sum())}+{int((self.y == 1).sum())} | val "
              f"{int((self.y_val == 0).sum())}+{int((self.y_val == 1).sum())} | connected test "
              f"{self.sg['conn'].mean():.1%}", flush=True)

    def point(self, s):
        return point_metrics(s, self.y, self.w, self.hard, self.gd_fpr)

    def row(self, level, name, per_seed, setting):
        """per_seed: list of test-score arrays (test order). Rank-average + seed spread."""
        s_ens = np.mean([to_ranks(s) for s in per_seed], axis=0)
        r = {"level": level, "method": name, **self.point(s_ens),
             **bootstrap_ci(s_ens, self.y, self.w, self.fam, self.boot),
             **population_metrics(s_ens, self.y, self.w, self.pop_groups),
             **subgroup_metrics(s_ens, self.y, self.w, self.sg)}
        per = pd.DataFrame([self.point(s) for s in per_seed])
        for col in ["auc_fw", "tpr1_fw", "tpr1_flagged_fw", "tpr1_hard_fw", *EXTRA_FPRS]:
            v = per[col]
            r[f"{col}_seed_mean"], r[f"{col}_seed_std"] = v.mean(), (v.std() if len(v) > 1 else 0.0)
            r[f"{col}_seed_min"], r[f"{col}_seed_max"] = v.min(), v.max()
        r["setting"] = setting
        print(f"level {level:.0%} | {name:44s} ROC-AUC (fw) {r['auc_fw']:.3f} [{r['auc_fw_lo']:.3f}, "
              f"{r['auc_fw_hi']:.3f}] | caught@1%FPR (fw, seed mean) {r['tpr1_fw_seed_mean']:.1%} | "
              f"@0.5% {r['tpr05_fw']:.1%} | connected: AUC {r.get('sg_auc_fw_conn', np.nan):.3f}, caught "
              f"{r.get('sg_tpr1_fw_conn', np.nan):.1%} | isolated: AUC {r.get('sg_auc_fw_iso', np.nan):.3f}",
              flush=True)
        print(f"    per seed caught@1%FPR (fw): {', '.join(f'{v:.1%}' for v in per['tpr1_fw'])}", flush=True)
        return r

    def calibrated(self, sv, st, fit_paths):
        """Per-subgroup calibration (with / without dependency peers among fit_paths),
        the same function as the baselines' 'subgroup-calibrated' rows."""
        f = self.feats
        fit_mask = f["path"].isin(set(fit_paths)).values
        idx = PeerIndex(f["path"].values[fit_mask], f["Name"].values[fit_mask], self.deps_of)
        has_v = np.asarray((idx.query(self.val_paths, f["Name"].values[self.val_idx], self.deps_of) > 0)
                           .sum(1)).ravel() > 0
        has_t = np.asarray((idx.query(self.test_paths, f["Name"].values[self.test_idx], self.deps_of) > 0)
                           .sum(1)).ravel() > 0
        return combine(np.where(has_v, sv, np.nan), np.where(has_t, st, np.nan), sv, st, self.y_val)
