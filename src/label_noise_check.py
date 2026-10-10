"""
label_noise_check.py
How much of the malware set is not "real" malware? (Methodology open item, section 15.)
Counts packages that look like proof-of-concept / security-research probes, CTF
material or test uploads, and checks whether detectors treat them differently.

Archives are read as TEXT only, in memory, with the same reader as the feature
extraction (extract_static_features.read_archive). Nothing is extracted to disk,
installed or executed.

Categories (case-insensitive, any match)
  research   self-declared research / proof of concept / bug bounty / dependency
             confusion test, in the package name, Summary or description
  callback   code or metadata contacts a known out-of-band testing service
             (Burp Collaborator, interact.sh, OAST, canarytokens, dnslog, ...).
             Typical of dependency-confusion probes, but real malware also
             uses some of them: an indicator, not proof
  ctf        capture-the-flag material
  test_name  the name itself says test / demo / poc / hello-world
Matches are candidates: the output lists the matched text so a sample can be
checked by hand before reporting a number.

Detection check (test malware only, clean training): share caught at 1% FPR
(threshold from the test normals, per seed, then averaged) for candidates vs the
rest, for the raw OCSVM (baseline_protocol_scores_primary.csv) and Triplet + GIN
with the new loss (contrastive_scores_triplet_gin_n3.csv, validation-chosen
configuration). If candidates are caught much less (or more) often, label noise
changes what "caught" means.

Outputs: RESULTS_DIR/label_noise_candidates.csv (one row per malicious package:
flags, matched snippets, split, family) and a printed summary.

Usage
  python src/label_noise_check.py
"""
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_protocol import DATA_DIR, RESULTS_DIR, SPLIT_FILE, load_features  # noqa: E402
from extract_static_features import read_archive  # noqa: E402

PATTERNS = {
    "research": re.compile(
        r"proof[\s_-]*of[\s_-]*concept|\bpoc\b|security[\s_-]*research|bug[\s_-]*bounty|hackerone|bugcrowd|"
        r"intigriti|yeswehack|dependency[\s_-]*confusion|research[\s_-]*purpose|for[\s_-]*research|"
        r"penetration[\s_-]*test|\bpentest|red[\s_-]*team|security[\s_-]*test|vulnerability[\s_-]*research|"
        r"responsible[\s_-]*disclosure|do[\s_-]*not[\s_-]*(install|use)|this[\s_-]*package[\s_-]*is[\s_-]*(a[\s_-]*)?"
        r"(test|placeholder|security)|harmless|benign", re.I),
    "callback": re.compile(
        r"burpcollaborator\.net|oastify\.com|interact\.sh|\boast\.(pro|live|site|online|fun|me)\b|canarytokens|"
        r"dnslog\.cn|ceye\.io|requestbin|pipedream\.net|webhook\.site|beeceptor|requestcatcher|hookbin|"
        r"ngrok\.io|interactsh", re.I),
    "ctf": re.compile(r"\bctf\b|capture[\s_-]*the[\s_-]*flag|flag\{", re.I),
}
TEST_NAME = re.compile(r"^(test|testing|demo|poc|hello|helloworld|example|dummy|sample)([-_.]|$)|"
                       r"[-_.](test|poc|demo)$", re.I)
MAX_TEXT = 200_000      # characters of code scanned per package


def snippet(text, m, width=60):
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return re.sub(r"\s+", " ", text[a:b])[:160]


def scan(item):
    """(path, name) -> flags and snippets. Text only, never executed."""
    path, name = item
    out = {"path": path, "read_failed": 0}
    try:
        _, texts, _ = read_archive(path)
    except Exception as e:  # noqa: BLE001 - corrupt archives are data
        out.update(read_failed=1, error=f"{type(e).__name__}"[:80])
        return out
    meta = [t for k, t in texts.items() if os.path.basename(k) in ("PKG-INFO", "METADATA")]
    meta_text = (meta[0] if meta else "")[:50_000]
    code_text = "\n".join(t for k, t in texts.items() if k.endswith(".py"))[:MAX_TEXT]
    where = {"research": f"{name}\n{meta_text}", "callback": f"{meta_text}\n{code_text}",
             "ctf": f"{name}\n{meta_text}\n{code_text}"}
    for cat, rx in PATTERNS.items():
        m = rx.search(where[cat])
        out[cat] = int(m is not None)
        out[f"{cat}_match"] = snippet(where[cat], m) if m else ""
    out["test_name"] = int(bool(TEST_NAME.search(name)))
    return out


def caught_by_group(scores, y, cand):
    """Mean over seeds of the share caught at 1% FPR, for candidates and for the rest (test malware)."""
    res = {"candidates": [], "rest": []}
    for s in scores:
        thr = np.quantile(s[y == 0], 0.99)
        hit = s > thr
        res["candidates"].append(hit[(y == 1) & cand].mean() if ((y == 1) & cand).any() else np.nan)
        res["rest"].append(hit[(y == 1) & ~cand].mean())
    return {k: float(np.nanmean(v)) for k, v in res.items()}


def seed_scores(df, paths):
    out = []
    for _, g in df.groupby("seed"):
        s = pd.Series(g["score"].astype(float).values, index=g["path"].values)
        out.append(s.reindex(paths).values)
    return out


def main():
    feats, _ = load_features()
    splits = pd.read_csv(f"{DATA_DIR}/{SPLIT_FILE}", dtype=str, keep_default_na=False)
    role = dict(zip(splits["path"], splits["split"]))
    fam = dict(zip(splits["path"], splits["campaign_id"]))
    mal = feats[feats["label"] == 1][["path", "Name"]].copy()
    print(f"scanning {len(mal)} malicious archives (text only)", flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as ex:
        rows = list(ex.map(scan, zip(mal["path"], mal["Name"]), chunksize=64))
    res = mal.merge(pd.DataFrame(rows), on="path")
    res["split"] = res["path"].map(role).fillna("")
    res["family"] = res["path"].map(fam).fillna("")
    cats = ["research", "callback", "ctf", "test_name"]
    res[cats] = res[cats].fillna(0).astype(int)
    res["any"] = (res[cats].sum(axis=1) > 0).astype(int)
    res["self_declared"] = ((res["research"] + res["ctf"]) > 0).astype(int)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = f"{RESULTS_DIR}/label_noise_candidates.csv"
    res.to_csv(out, index=False)

    ok = res[res["read_failed"] == 0]
    print(f"\nread {len(ok)} of {len(res)} archives\n")
    print("=== Candidates (packages; families) ===")
    for c in cats + ["self_declared", "any"]:
        n, f = int(ok[c].sum()), ok.loc[ok[c] == 1, "family"].nunique()
        print(f"{c:14s} {n:6d} packages ({n / len(ok):5.1%}) in {f:5d} families")
    print("\n=== By split (share of malicious packages flagged: self-declared / any) ===")
    for sp, g in ok.groupby("split"):
        print(f"{sp or '(none)':8s} {len(g):6d} packages | self-declared {g['self_declared'].mean():5.1%} | "
              f"any {g['any'].mean():5.1%}")
    print("\n=== 10 random self-declared candidates (check by hand) ===")
    sd = ok[ok["self_declared"] == 1]
    for r in sd.sample(min(10, len(sd)), random_state=0).itertuples():
        m = r.research_match or r.ctf_match
        print(f"  {r.Name[:30]:30s} [{r.split}] {m[:110]}")

    # detection: test malware, clean training
    test_paths = feats.loc[feats["path"].map(role) == "test", "path"].values
    y = feats.set_index("path").loc[test_paths, "label"].values
    flag = res.set_index("path")["self_declared"].reindex(test_paths).fillna(0).astype(bool).values
    anyf = res.set_index("path")["any"].reindex(test_paths).fillna(0).astype(bool).values
    print(f"\n=== Caught at 1% FPR on test, clean training (self-declared: {int(flag[y == 1].sum())} of "
          f"{int((y == 1).sum())} test malware; any flag: {int(anyf[y == 1].sum())}) ===")
    sources = []
    f = f"{RESULTS_DIR}/baseline_protocol_scores_primary.csv"
    if os.path.exists(f):
        d = pd.read_csv(f, dtype={"path": str}, keep_default_na=False)
        d = d[(d["level"].astype(float) == 0) & (d["detector"] == "OCSVM") & (d["variant"] == "naive")
              & (d["features"] == "static+deps+names+GuardDog")]
        sources.append(("raw OCSVM", d))
    f = f"{RESULTS_DIR}/contrastive_scores_triplet_gin_n3.csv"
    if os.path.exists(f):
        st = pd.read_csv(f"{RESULTS_DIR}/contrastive_settings_triplet_gin_n3.csv")
        ch = st[(st["level"] == 0) & st["chosen"].astype(bool)].iloc[0]
        d = pd.read_csv(f, dtype={"path": str}, keep_default_na=False)
        d = d[(d["split"] == "test") & (d["level"] == 0) & (d["depth"] == ch["depth"]) & (d["scorer"] == ch["scorer"])]
        sources.append((f"Triplet+GIN N3 (depth {int(ch['depth'])} {ch['scorer']})", d))
    for name, d in sources:
        S = seed_scores(d, test_paths)
        a, b = caught_by_group(S, y, flag), caught_by_group(S, y, anyf)
        print(f"{name:34s} self-declared {a['candidates']:5.1%} vs rest {a['rest']:5.1%} | "
              f"any flag {b['candidates']:5.1%} vs rest {b['rest']:5.1%}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
