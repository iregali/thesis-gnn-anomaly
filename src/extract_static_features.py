"""
extract_static_features.py

Lightweight, text-only feature extraction from package archives.
Each archive is read ONCE, in memory. Nothing is executed and the code is
never parsed, so this cannot hang the way GuardDog does on obfuscated code.

Feature groups
  - metadata     from PKG-INFO: links, license, description, classifiers, ...
  - code shape   line length, nesting depth, entropy, symbol ratio, ...
  - capability   text searches: exec/eval, dynamic builtins, decode chains, ...
  - install-time setup.py size and share, argv checks, cmdclass overrides
  - build setup  which build files exist: setup.py, pyproject.toml, setup.cfg
  - shape        number of files, compression ratio, binaries

Why this exists
  GuardDog hangs on heavily obfuscated packages (e.g. a 42,000-character line
  of nested arithmetic) and times out on ~12% of malicious packages. These
  features come from plain text scanning, so they also cover exactly the
  packages GuardDog can't scan.

Parity
  Both classes go through the SAME code on the SAME kind of input: a local
  sdist in the layout <root>/<name>/<version>/<archive>. Malicious archives
  come from malregistry, normal ones from sample_normal.py. Folder parsing is
  imported from extract_guarddog.py, so both extractors see the same packages.
  Features that only one class could have (e.g. "also has a wheel on PyPI")
  are deliberately NOT included.

Usage
  python src/extract_static_features.py --root ~/thesis/pypi_malregistry \\
      --out ~/thesis/data/static_malicious.csv
  python src/extract_static_features.py --root ~/thesis/data/normal_archives \\
      --out ~/thesis/data/static_normal.csv
"""

import argparse
import email.parser
import math
import os
import re
import tarfile
import zipfile
from collections import Counter
from multiprocessing import Pool

import pandas as pd

# Same <name>/<version>/<archive> parsing as the GuardDog extractor.
from extract_guarddog import load_local_items

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
MAX_FILE_BYTES = 5 * 1024 * 1024    # skip reading any single file above this
MAX_TOTAL_BYTES = 50 * 1024 * 1024  # stop reading text after this per package
LONG_LINE = 1000
BINARY_EXTS = (".so", ".pyd", ".dll", ".exe", ".dylib")

README_RE = re.compile(r"(^|/)readme(\.[a-z]+)?$", re.I)
LICENSE_FILE_RE = re.compile(r"(^|/)(license|licence|copying)(\.[a-z]+)?$", re.I)
REPO_RE = re.compile(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/([^/\s#?]+)", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
EMAIL_LOCAL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@")

# All patterns are simple (no nested quantifiers), so they run in linear time
# even on pathological input.
RE_STRING = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")
RE_BASE64 = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
RE_DYN_EXEC = re.compile(r"(?<![.\w])(?:exec|eval|compile)\s*\(")  # excludes re.compile(
RE_DYN_BUILTINS = re.compile(
    r"getattr\s*\(\s*__builtins__|__import__\s*\(|globals\s*\(\s*\)\s*\[|__builtins__\s*\.\s*__dict__")
RE_DECODE = re.compile(
    r"b64decode|b32decode|b16decode|a85decode|b85decode|zlib\.decompress|marshal\.loads"
    r"|codecs\.decode|bytes\.fromhex|lzma\.decompress|bz2\.decompress")
RE_SPAWN = re.compile(r"\bsubprocess\b|os\.system|os\.popen|os\.exec[lv]p?e?|\bPopen\b|os\.startfile")
RE_NET_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:socket|urllib|urllib2|urllib3|requests|http\.client|httpx|aiohttp"
    r"|ftplib|smtplib)\b", re.M)
RE_PLATFORM = re.compile(r"sys\.platform|os\.name|platform\.system")


# ---------------------------------------------------------------------------
# Reading archives (in memory, never extracted to disk)
# ---------------------------------------------------------------------------
def _wants_text(name):
    return name.endswith(".py") or os.path.basename(name) in ("PKG-INFO", "METADATA")


def read_archive(path):
    """Return (files, texts, truncated): all (name, size) pairs, the text of
    .py / PKG-INFO / METADATA files, and whether size caps cut anything."""
    files, texts, truncated, total = [], {}, False, 0

    def take(name, size, read_fn):
        nonlocal total, truncated
        if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
            truncated = True
            return
        data = read_fn()[:MAX_FILE_BYTES]
        total += len(data)
        texts[name] = data.decode("utf-8", errors="replace")

    if path.endswith((".zip", ".whl")):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                files.append((info.filename, info.file_size))
                if _wants_text(info.filename):
                    take(info.filename, info.file_size, lambda i=info: z.read(i))
    else:
        with tarfile.open(path, "r:*") as t:
            for m in t:
                if not m.isfile():  # skips dirs and symlinks
                    continue
                files.append((m.name, m.size))
                if _wants_text(m.name):
                    take(m.name, m.size, lambda mm=m: t.extractfile(mm).read(MAX_FILE_BYTES + 1))
    return files, texts, truncated


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
def _clean(value):
    value = (value or "").strip()
    return "" if value.lower() in ("", "unknown", "none") else value


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _top_level(name):
    """True for files at the top of the package (e.g. 'pkg-1.0/setup.py')."""
    return name.count("/") <= 1


def metadata_features(pkg_info_text, file_names):
    f = {"has_pkg_info": int(pkg_info_text is not None)}
    msg = email.parser.Parser().parsestr(pkg_info_text) if pkg_info_text else None

    def get(key):
        return _clean(msg.get(key)) if msg else ""

    def get_all(key):
        return (msg.get_all(key) or []) if msg else []

    home, download = get("Home-page"), get("Download-URL")
    project_urls = []
    for entry in get_all("Project-URL"):
        label, _, url = entry.partition(",")
        url = _clean(url)
        if url:
            project_urls.append((label.strip().lower(), url))
    urls = {u for u in (home, download) if u} | {u for _, u in project_urls}

    repo_owner = None
    for u in urls:
        m = REPO_RE.search(u)
        if m:
            repo_owner = m.group(1).lower()
            break

    f["has_source_repo"] = int(repo_owner is not None)
    f["has_homepage"] = int(bool(home))
    f["has_issue_tracker"] = int(any(
        "issue" in lab or "bug" in lab or "tracker" in lab or "/issues" in u for lab, u in project_urls))
    f["has_documentation"] = int(any(
        "doc" in lab or "readthedocs" in u or "/docs" in u for lab, u in project_urls))
    f["link_count"] = len(urls)

    classifiers = get_all("Classifier")
    license_classifiers = [c for c in classifiers if c.strip().startswith("License ::")]
    license_field = get("License") or get("License-Expression")
    license_file = any(LICENSE_FILE_RE.search(n) for n in file_names)
    f["has_license"] = int(bool(license_field) or bool(license_classifiers) or license_file)
    f["license_count"] = len(license_classifiers) + int(bool(license_field))

    body = msg.get_payload() if msg else ""
    description = (body.strip() if isinstance(body, str) else "") or get("Description")
    f["has_description"] = int(bool(get("Summary")))
    f["description_length"] = len(description)
    f["classifier_count"] = len(classifiers)
    f["has_readme"] = int(any(README_RE.search(n) for n in file_names))

    # Does the declared author plausibly own the linked repo?
    # Undefined when there is no repo or no author -> defined flag = 0.
    author = " ".join(get(k) for k in ("Author", "Author-email", "Maintainer", "Maintainer-email"))
    if repo_owner and author.strip():
        owner = _norm(repo_owner)
        name_part = EMAIL_RE.sub(" ", author)
        tokens = [_norm(t) for t in re.split(r"[\s,<>]+", name_part)]
        tokens += [_norm(loc) for loc in EMAIL_LOCAL_RE.findall(author)]
        tokens = [t for t in tokens if len(t) >= 3]
        consistent = owner in _norm(name_part) or any(t in owner or owner in t for t in tokens)
        f["author_repo_consistency"], f["author_repo_defined"] = int(consistent), 1
    else:
        f["author_repo_consistency"], f["author_repo_defined"] = 0, 0
    return f


def code_features(py_texts, setup_text):
    f = {}
    code = "\n".join(py_texts)
    n = len(code)

    max_line = n_long = 0
    for text in py_texts:
        for line in text.splitlines():
            length = len(line)
            max_line = max(max_line, length)
            n_long += length > LONG_LINE

    # One pass: bracket nesting (naive: brackets inside strings count too),
    # letters vs. non-whitespace characters.
    depth = max_depth = letters = nonws = 0
    for ch in code:
        if ch in "([{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        if not ch.isspace():
            nonws += 1
            if ch.isalpha():
                letters += 1

    counts = Counter(code)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) if n else 0.0
    longest_string = max((m.end() - m.start() - 2 for m in RE_STRING.finditer(code)), default=0)

    f["max_line_length"] = max_line
    f["n_long_lines"] = n_long
    f["max_nesting_depth"] = max_depth
    f["code_entropy"] = round(entropy, 4)
    f["symbol_ratio"] = round(1 - letters / nonws, 4) if nonws else 0.0
    f["bitshift_density"] = round((code.count("<<") + code.count(">>")) * 1000 / n, 4) if n else 0.0
    f["max_string_literal_length"] = max(longest_string, 0)
    f["has_base64_blob"] = int(bool(RE_BASE64.search(code)))

    f["has_dynamic_execution"] = int(bool(RE_DYN_EXEC.search(code)))
    f["has_dynamic_builtins"] = int(bool(RE_DYN_BUILTINS.search(code)))
    f["has_decode_chain"] = int(bool(RE_DECODE.search(code)))
    f["has_process_spawn"] = int(bool(RE_SPAWN.search(code)))
    f["has_network_import"] = int(bool(RE_NET_IMPORT.search(code)))
    f["has_platform_check"] = int(bool(RE_PLATFORM.search(code)))

    f["setup_py_size"] = len(setup_text)
    f["setup_py_to_body_ratio"] = round(len(setup_text) / n, 4) if n else 0.0
    f["has_argv_check_in_setup"] = int("sys.argv" in setup_text)
    f["has_cmdclass_override"] = int("cmdclass" in setup_text)
    return f


def build_setup_features(file_names):
    """Which build files the package ships. setup.py can run code at install
    time; pyproject.toml and setup.cfg are declarative. Read from the archive
    itself, so it is computed identically for both classes."""
    top = {os.path.basename(n) for n in file_names if _top_level(n)}
    return {
        "has_setup_py": int("setup.py" in top),
        "has_pyproject_toml": int("pyproject.toml" in top),
        "has_setup_cfg": int("setup.cfg" in top),
    }


def shape_features(files, path):
    names = [fname for fname, _ in files]
    total_kb = sum(size for _, size in files) / 1024
    archive_kb = os.path.getsize(path) / 1024
    return {
        "n_files": len(files),
        "n_py_files": sum(1 for fname in names if fname.endswith(".py")),
        "total_uncompressed_kb": round(total_kb, 2),
        "compression_ratio": round(total_kb / archive_kb, 3) if archive_kb else 0.0,
        "has_binary_files": int(any(fname.lower().endswith(BINARY_EXTS) for fname in names)),
    }


BASE_KEYS = ["Name", "Version", "path", "archive_type", "read_failed", "error", "truncated"]
SHAPE_KEYS = ["n_files", "n_py_files", "total_uncompressed_kb", "compression_ratio", "has_binary_files"]
COLUMNS = (BASE_KEYS
           + list(metadata_features(None, []).keys())
           + list(code_features([], "").keys())
           + list(build_setup_features([]).keys())
           + SHAPE_KEYS)


def extract_one(item):
    name, version, path = item
    row = {"Name": name, "Version": str(version), "path": path,
           "archive_type": "wheel" if path.endswith(".whl") else "sdist",
           "read_failed": 0, "error": "", "truncated": 0}
    try:
        files, texts, truncated = read_archive(path)
    except Exception as e:  # noqa: BLE001 - corrupt/odd archives are data, not crashes
        row.update(read_failed=1, error=f"{type(e).__name__}: {e}"[:300])
        return row

    names = [fname for fname, _ in files]
    by_depth = lambda k: k.count("/")  # noqa: E731 - prefer top-level files
    meta = sorted((k for k in texts if os.path.basename(k) in ("PKG-INFO", "METADATA")), key=by_depth)
    py_names = [k for k in texts if k.endswith(".py")]
    setups = sorted((k for k in py_names if os.path.basename(k) == "setup.py"), key=by_depth)

    row["truncated"] = int(truncated)
    row.update(metadata_features(texts[meta[0]] if meta else None, names))
    row.update(code_features([texts[k] for k in py_names], texts[setups[0]] if setups else ""))
    row.update(build_setup_features(names))
    row.update(shape_features(files, path))
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_extract(root, out, workers, limit):
    items = load_local_items(root)
    if limit:
        items = items[:limit]
    done = set()
    if os.path.exists(out):
        done = set(pd.read_csv(out, dtype=str, keep_default_na=False)["path"])
    todo = [it for it in items if it[2] not in done]
    print(f"[extract] total: {len(items)} | already done: {len(done)} | to do: {len(todo)} "
          f"| workers: {workers}", flush=True)

    write_header = not os.path.exists(out)
    with Pool(workers) as pool:
        batch = []
        for i, row in enumerate(pool.imap_unordered(extract_one, todo, chunksize=4), 1):
            batch.append(row)
            if len(batch) >= 200:
                pd.DataFrame(batch).reindex(columns=COLUMNS).to_csv(
                    out, mode="a", header=write_header, index=False)
                write_header, batch = False, []
                print(f"[progress] {i}/{len(todo)}", flush=True)
        if batch:
            pd.DataFrame(batch).reindex(columns=COLUMNS).to_csv(
                out, mode="a", header=write_header, index=False)

    df = pd.read_csv(out, dtype={"Version": str}, keep_default_na=False)
    ok = df[df["read_failed"].astype(int) == 0]
    print("\n=== Extract summary ===")
    print(f"rows: {len(df)} | read_failed: {len(df) - len(ok)} | truncated: "
          f"{int(ok['truncated'].astype(int).sum())}")
    print("archive types:", df["archive_type"].value_counts().to_dict())
    print("\nmedians:")
    for col in ["max_line_length", "max_nesting_depth", "code_entropy", "symbol_ratio",
                "setup_py_size", "setup_py_to_body_ratio", "compression_ratio", "n_files"]:
        print(f"  {col:24s} {pd.to_numeric(ok[col], errors='coerce').median()}")
    print("\nshare with flag = 1:")
    for col in ["has_source_repo", "has_license", "has_readme", "has_setup_py", "has_pyproject_toml",
                "has_dynamic_execution", "has_dynamic_builtins", "has_decode_chain",
                "has_process_spawn", "has_network_import", "has_argv_check_in_setup",
                "has_base64_blob"]:
        print(f"  {col:24s} {pd.to_numeric(ok[col], errors='coerce').mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description="Text-only static features from package archives")
    ap.add_argument("--root", required=True, help="folder with <name>/<version>/<archive>")
    ap.add_argument("--out", required=True, help="output CSV (resumable)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N archives (testing)")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()
    run_extract(os.path.expanduser(args.root), os.path.expanduser(args.out), args.workers, args.limit)


if __name__ == "__main__":
    main()
