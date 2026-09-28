"""
extract_dependencies.py
Extracts each package's DECLARED dependencies from its archive, text-only:
nothing is executed or installed, and the same code is used for malicious
and normal packages (parity). The output becomes the dependency edges of
the graph.

Sources (all read, results combined):
  - PKG-INFO / METADATA     Requires-Dist lines
  - *.egg-info/requires.txt  written by setuptools; [section] = optional extra
  - setup.py                 install_requires=[...] / extras_require (string literals only)
  - setup.cfg                [options] install_requires / [options.extras_require]
  - pyproject.toml           [project] dependencies / optional-dependencies,
                             [tool.poetry.dependencies] / group dependencies
  - requirements.txt         top-level file, one requirement per line

Each dependency is marked `optional` if it only appears as an extra
(e.g. `pip install pkg[dev]`) or in a development group. Core dependencies
are what a plain `pip install` pulls in; they are the main edges.

A package whose setup.py computes install_requires with code (not a literal
list) and declares nothing elsewhere is marked `dynamic`: its dependencies
cannot be read without running code. Reported, never guessed.

Outputs (per archive folder):
  <out-prefix>_edges.csv     Name, Version, path, dep, optional, sources
  <out-prefix>_packages.csv  Name, Version, path, n_core, n_optional, sources_found, status

Usage
  python src/extract_dependencies.py --root ~/thesis/pypi_malregistry \\
      --out-prefix ~/thesis/data/deps_malicious
  python src/extract_dependencies.py --root ~/thesis/data/normal_archives \\
      --out-prefix ~/thesis/data/deps_normal
"""

import argparse
import ast
import configparser
import email.parser
import os
import re
import tarfile
import zipfile
from collections import Counter
from multiprocessing import Pool

import pandas as pd

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:  # pragma: no cover
    Requirement = None

from extract_guarddog import load_local_items

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
MAX_FILE_BYTES = 2 * 1024 * 1024
WANTED = ("PKG-INFO", "METADATA", "requires.txt", "setup.py", "setup.cfg",
          "pyproject.toml", "requirements.txt")
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def pep503(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


# ---------------------------------------------------------------------------
# Reading (in memory, only the few files that declare dependencies)
# ---------------------------------------------------------------------------
def depth(name):
    return name.count("/")


def read_declaration_files(path):
    """Return {relative_name: text} for dependency-declaring files, keeping only
    top-level ones (package root), plus egg-info/requires.txt."""
    files = {}

    def keep(name):
        base = os.path.basename(name)
        if base not in WANTED:
            return False
        if base == "requires.txt":
            return ".egg-info/" in name and depth(name) <= 2
        if base == "METADATA":
            return ".dist-info/" in name
        return depth(name) <= 1  # e.g. "pkg-1.0/setup.py"

    if path.endswith((".zip", ".whl")):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir() and keep(info.filename) and info.file_size <= MAX_FILE_BYTES:
                    files[info.filename] = z.read(info).decode("utf-8", "replace")
    else:
        with tarfile.open(path, "r:*") as t:
            for m in t:
                if m.isfile() and keep(m.name) and m.size <= MAX_FILE_BYTES:
                    files[m.name] = t.extractfile(m).read().decode("utf-8", "replace")
    return files


# ---------------------------------------------------------------------------
# Parsing requirement strings
# ---------------------------------------------------------------------------
def req_name(text):
    """Distribution name from a requirement string, or None."""
    text = text.strip()
    if not text or text.startswith(("#", "-", "git+", "http:", "https:", "file:")):
        return None
    if Requirement is not None:
        try:
            return pep503(Requirement(text).name)
        except InvalidRequirement:
            pass
    m = NAME_RE.match(text)
    return pep503(m.group(1)) if m else None


def req_is_extra_only(text):
    """True for Requires-Dist entries that only apply to an extra."""
    return bool(re.search(r"""extra\s*==\s*['"]""", text))


# ---------------------------------------------------------------------------
# One parser per source; each returns (core_names, optional_names, found)
# ---------------------------------------------------------------------------
def from_metadata(text):
    msg = email.parser.Parser().parsestr(text)
    core, opt = set(), set()
    for entry in msg.get_all("Requires-Dist") or []:
        name = req_name(entry.split(";")[0])
        if name:
            (opt if req_is_extra_only(entry) else core).add(name)
    return core, opt, bool(core or opt)


def from_requires_txt(text):
    core, opt, section = set(), set(), None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            inner = line[1:-1]
            # "[:python_version<'3.8']" = conditional core; "[dev]" = extra
            section = None if inner.startswith(":") else inner
            continue
        name = req_name(line)
        if name:
            (opt if section else core).add(name)
    return core, opt, True


def _literal_strings(node):
    """String literals inside a list/tuple AST node (ignores anything computed)."""
    out = []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
    return out


def from_setup_py(text):
    """install_requires / extras_require as literal lists in setup(...).
    Parsing is attempted with ast (safe: parses, never executes) and only on
    files of reasonable size; obfuscated giants fall back to a regex."""
    core, opt, found, dynamic = set(), set(), False, False
    tree = None
    if len(text) < 500_000 and text.count("(") < 20_000:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "install_requires":
                    found = True
                    names = [req_name(s) for s in _literal_strings(kw.value)]
                    if not isinstance(kw.value, (ast.List, ast.Tuple, ast.Set)):
                        dynamic = True
                    core |= {n for n in names if n}
                elif kw.arg == "extras_require" and isinstance(kw.value, ast.Dict):
                    found = True
                    for v in kw.value.values:
                        opt |= {n for n in (req_name(s) for s in _literal_strings(v)) if n}
    else:
        m = re.search(r"install_requires\s*=\s*\[([^\]]*)\]", text[:200_000])
        if m:
            found = True
            for s in re.findall(r"""['"]([^'"]+)['"]""", m.group(1)):
                n = req_name(s)
                if n:
                    core.add(n)
        elif "install_requires" in text:
            found, dynamic = True, True
    return core, opt, found, dynamic


def from_setup_cfg(text):
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        cp.read_string(text)
    except configparser.Error:
        return set(), set(), False
    core, opt, found = set(), set(), False
    if cp.has_option("options", "install_requires"):
        found = True
        for line in cp.get("options", "install_requires").splitlines():
            n = req_name(line.split(";")[0])
            if n:
                core.add(n)
    if cp.has_section("options.extras_require"):
        found = True
        for _, value in cp.items("options.extras_require"):
            for line in value.splitlines():
                n = req_name(line.split(";")[0])
                if n:
                    opt.add(n)
    return core, opt, found


def from_pyproject(text):
    if tomllib is None:
        return set(), set(), False
    try:
        data = tomllib.loads(text)
    except Exception:  # noqa: BLE001
        return set(), set(), False
    core, opt, found = set(), set(), False
    project = data.get("project", {})
    if "dependencies" in project:
        found = True
        core |= {n for n in (req_name(s) for s in project.get("dependencies") or []) if n}
    for group in (project.get("optional-dependencies") or {}).values():
        found = True
        opt |= {n for n in (req_name(s) for s in group or []) if n}
    poetry = data.get("tool", {}).get("poetry", {})
    if "dependencies" in poetry:
        found = True
        core |= {pep503(k) for k in poetry["dependencies"] if k.lower() != "python"}
    for key in ("dev-dependencies",):
        opt |= {pep503(k) for k in poetry.get(key, {}) or {}}
    for group in (poetry.get("group") or {}).values():
        opt |= {pep503(k) for k in (group or {}).get("dependencies", {}) or {}}
    return core, opt, found


def from_requirements_txt(text):
    core = {n for n in (req_name(line.split(";")[0]) for line in text.splitlines()) if n}
    return core, set(), True


# ---------------------------------------------------------------------------
# Per archive
# ---------------------------------------------------------------------------
def extract_one(item):
    name, version, path = item
    own = pep503(name)
    try:
        files = read_declaration_files(path)
    except Exception as e:  # noqa: BLE001
        return {"Name": name, "Version": str(version), "path": path, "status": "unreadable",
                "error": f"{type(e).__name__}: {e}"[:200]}, []

    core, opt = {}, {}  # dep -> set of sources
    sources_found, dynamic = [], False

    def add(names, target, source):
        for n in names:
            if n and n != own:
                target.setdefault(n, set()).add(source)

    for fname, text in sorted(files.items(), key=lambda kv: depth(kv[0])):
        base = os.path.basename(fname)
        if base in ("PKG-INFO", "METADATA"):
            c, o, f = from_metadata(text)
            src = "metadata"
        elif base == "requires.txt":
            c, o, f = from_requires_txt(text)
            src = "requires.txt"
        elif base == "setup.py":
            c, o, f, dyn = from_setup_py(text)
            dynamic = dynamic or dyn
            src = "setup.py"
        elif base == "setup.cfg":
            c, o, f = from_setup_cfg(text)
            src = "setup.cfg"
        elif base == "pyproject.toml":
            c, o, f = from_pyproject(text)
            src = "pyproject.toml"
        else:
            c, o, f = from_requirements_txt(text)
            src = "requirements.txt"
        if f:
            sources_found.append(src)
        add(c, core, src)
        add(o, opt, src)

    # a dependency declared as core anywhere is core
    opt = {k: v for k, v in opt.items() if k not in core}

    if core or opt:
        status = "found"
    elif dynamic:
        status = "dynamic"
    elif sources_found:
        status = "none_declared"
    else:
        status = "no_declaration_files"

    summary = {"Name": name, "Version": str(version), "path": path,
               "n_core": len(core), "n_optional": len(opt),
               "sources_found": ";".join(sorted(set(sources_found))),
               "status": status, "error": ""}
    edges = [{"Name": name, "Version": str(version), "path": path, "dep": d,
              "optional": 0, "sources": ";".join(sorted(s))} for d, s in sorted(core.items())]
    edges += [{"Name": name, "Version": str(version), "path": path, "dep": d,
               "optional": 1, "sources": ";".join(sorted(s))} for d, s in sorted(opt.items())]
    return summary, edges


def main():
    ap = argparse.ArgumentParser(description="Declared dependencies from package archives")
    ap.add_argument("--root", required=True, help="folder with <name>/<version>/<archive>")
    ap.add_argument("--out-prefix", required=True, help="e.g. ~/thesis/data/deps_normal")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root, prefix = os.path.expanduser(args.root), os.path.expanduser(args.out_prefix)

    items = load_local_items(root)
    if args.limit:
        items = items[:args.limit]
    print(f"[info] {len(items)} archives, {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        results = pool.map(extract_one, items, chunksize=16)

    packages = pd.DataFrame([r[0] for r in results])
    edges = pd.DataFrame([e for r in results for e in r[1]],
                         columns=["Name", "Version", "path", "dep", "optional", "sources"])
    packages.to_csv(f"{prefix}_packages.csv", index=False)
    edges.to_csv(f"{prefix}_edges.csv", index=False)

    # ---- Summary ----
    ok = packages[packages["status"] != "unreadable"]
    print("\n=== Dependency extraction ===")
    print(f"archives: {len(packages)} | unreadable: {len(packages) - len(ok)}")
    print("status:", ok["status"].value_counts().to_dict())
    print(f"with >= 1 core dependency: {(ok['n_core'] > 0).mean():.1%} | "
          f"core deps per package: median {ok['n_core'].median():.0f}, "
          f"mean {ok['n_core'].mean():.1f}, max {ok['n_core'].max()}")
    src_counts = Counter(s for v in ok["sources_found"] if v for s in v.split(";"))
    print("declaration sources found (packages):", dict(src_counts.most_common()))
    core_edges = edges[edges["optional"] == 0]
    print(f"\ncore edges: {len(core_edges)} | optional edges: {int(edges['optional'].sum())} | "
          f"distinct core targets: {core_edges['dep'].nunique()}")
    print("\nmost common core dependencies:")
    for dep, n in core_edges["dep"].value_counts().head(20).items():
        print(f"  {dep:30s} {n:6d} ({n / max(len(ok), 1):.1%} of packages)")
    print(f"\nSaved {prefix}_packages.csv and {prefix}_edges.csv")


if __name__ == "__main__":
    main()
