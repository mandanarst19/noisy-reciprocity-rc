"""Fetch the authors' public code at pinned commits and import it, patched IN MEMORY.

Why not vendor it?  The metrics repository ships no LICENSE file, so this repo does not
redistribute copies of it.  Instead `load()` clones both repositories into ./external/ (git
clone, pinned commit) and imports their modules.  Three problems in the published code are
fixed at import time (nothing on disk in ./external is modified) and are listed in REPORT.md:

 P1  metrics.py: the module docstring is never closed, `W_in =` is split over two lines, and
     `W.float()` (a torch method) is called on numpy arrays -> file does not parse as published.
 P2  echoes 0.0.8 calls sklearn Ridge(normalize=...), removed in scikit-learn 1.2 -> TypeError.
     The pinned scikit-learn==1.3.2 in the authors' requirements.txt fails in the same way.
     Shim: drop the `normalize` keyword (it was False/deprecated anyway).
 P3  (none else patched)  The numerical behaviour of the authors' functions is untouched.
"""
import importlib.util, os, subprocess, sys, types

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "external")
REPOS = {
    "rc":  ("https://github.com/m00rcheh/network-reciprocity-RC-memory-capacity-and-kernel-rank",
            "6ca9d1fcf06e23d5c3d8e17d70c4b9abb9f9430a"),
    "nrc": ("https://github.com/m00rcheh/NRC_binary_and_weighted_Network_Reciprocity_Control",
            "8a9b08111379516540164ec0e525807697125841"),
}


def _fetch():
    os.makedirs(EXT, exist_ok=True)
    for name, (url, commit) in REPOS.items():
        d = os.path.join(EXT, name)
        if not os.path.isdir(d):
            subprocess.run(["git", "clone", "-q", url, d], check=True)
        subprocess.run(["git", "-C", d, "checkout", "-q", commit], check=True)


def _patch_echoes_ridge():
    """P2: make echoes' Ridge call work with scikit-learn >= 1.2."""
    import echoes.esn._base as base
    from sklearn.linear_model import Ridge as _Ridge
    if getattr(base, "_ridge_patched", False):
        return

    def Ridge(*a, **kw):
        kw.pop("normalize", None)
        return _Ridge(*a, **kw)
    base.Ridge = Ridge
    base._ridge_patched = True


def _patch_metrics_source(src: str) -> str:
    """P1: minimal textual fixes so that metrics.py parses and runs on numpy arrays."""
    # (a) close the module docstring that the authors left open
    marker = "https://github.com/neurogym/neurogym\n\nimport warnings"
    assert marker in src, "metrics.py layout changed; patch no longer applies"
    src = src.replace(marker, 'https://github.com/neurogym/neurogym\n"""\nimport warnings', 1)
    # (b) `W_in    = \n   rng.uniform(...)` -> single line
    import re
    src, n = re.subn(r"W_in\s*=\s*\n\s*rng\.uniform", "W_in = rng.uniform", src)
    assert n == 1, "W_in line fix did not apply exactly once"
    # (c) torch-style .float() on numpy arrays
    src = src.replace("W.float()", "W.astype(float)")
    return src


def _load_module(name, path, src=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    code = src if src is not None else open(path).read()
    exec(compile(code, path, "exec"), mod.__dict__)
    return mod


def load():
    """Return (utils, metrics) modules of the authors' metrics repository."""
    _fetch()
    _patch_echoes_ridge()
    rc = os.path.join(EXT, "rc")
    utils = _load_module("authors_utils", os.path.join(rc, "utils.py"))
    mpath = os.path.join(rc, "metrics.py")
    metrics = _load_module("authors_metrics", mpath, _patch_metrics_source(open(mpath).read()))
    return utils, metrics
