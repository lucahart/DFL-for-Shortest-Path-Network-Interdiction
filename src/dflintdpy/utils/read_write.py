"""
read_write.py

A small, robust caching layer for simulation artefacts (datasets, interdictions,
predictors, results, and figures).

Goals
-----
1) Deterministic cache keys (hashes):
   - Same *semantic* cfg content -> same hash.
   - Small, irrelevant cfg changes (e.g., logging path) can be excluded from hashes.
   - Stable across cfg composition changes (new keys appear / old keys disappear).

2) Avoid accidental cache duplication:
   - Canonicalize cfg to a deterministic representation before hashing.
   - Hash only the relevant subset of cfg for each artefact type.

3) Make cached content inspectable and uniquely identifiable:
   - Each stored artefact has a sidecar metadata JSON that contains:
       * canonical cfg-subset used for hashing
       * the full "head" hash (hash of the entire cfg snapshot)
       * a timestamp (optional)
   - This means: given a hash, you can look up exactly which cfg-subset produced it.

File layout (relative to this file)
-----------------------------------
store_data/
  datasets/         data_<data_hash>.npz + data_<data_hash>.meta.json
  interdictions/    rnd_<intd_hash>.pkl  + rnd_<intd_hash>.meta.json
                    adv_<intd_hash>.pkl  + adv_<intd_hash>.meta.json
  predictors/       pred_<pred_hash>.pkl + pred_<pred_hash>.meta.json
  results/          result_<res_hash>.pkl+ result_<res_hash>.meta.json
  figures/          <fig_type>_<res_hash>.png + <fig_type>_<res_hash>.meta.json

Notes
-----
- We store datasets as compressed NumPy .npz because those are commonly arrays.
- We store most other artefacts as pickle because they can be arbitrary Python objects.
- Figures are saved as PNG by default (easy, compact). Adjust if you prefer PDF/SVG.

You MUST define which cfg parameters matter for each artefact type:
- DATA_KEYS:     keys that define the dataset generation
- INTD_KEYS:     keys that define interdiction generation
- PRED_KEYS:     keys that define predictor training architecture & settings
- RESULT_KEYS:   keys that define result generation/evaluation (often includes others)

These keys are expressed as "dotted paths" (e.g., "data.n_nodes", "training.seed").
The cfg can be a dict, a dataclass, a namespace-like object, or a nested combination.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# NumPy is optional but strongly recommended for dataset storage.
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# =============================================================================
# 1) Define which cfg parameters uniquely identify each artefact type
# =============================================================================
#
# IMPORTANT:
# - These should be your "semantic" parameters.
# - Do NOT include ephemeral runtime keys (timestamps, log dirs, etc.).
# - If you refactor your config (rename keys), decide whether you want to:
#     a) break cache intentionally (new hash), OR
#     b) keep cache continuity (add aliases below in KEY_ALIASES).
#
# Dotted-path examples: "data.n_nodes", "solver.tol", "seed"
#
DATA_KEYS: Set[str] = {
    # --- example placeholders; replace with your real cfg keys ---
    "num_features",
    "num_train_samples",
    "num_val_samples",
    "num_test_samples",
    "grid_size",
    "deg",
    "noise_width",
    "seed",
    "random_seed",
    "loader_seed",
}

INTD_KEYS: Set[str] = {
    # Keys that define interdiction generation / environment.
    *DATA_KEYS,  # interdiction generation depends on data generation
    "budget",
    "num_scenarios",
    "benders_max_count",
    "benders_eps",
    "lsd",
    "intd_seed",
}

PRED_KEYS: Set[str] = {
    *DATA_KEYS,  # predictor training depends on data generation
    *INTD_KEYS,  # predictor training can depend on interdiction generation
    "batch_size",
    "po_epochs",
    "spo_epochs",
    "po_lr",
    "spo_lr",
}

RESULT_KEYS: Set[str] = {
    # Keys that define evaluation pipeline and result-generation.
    # Often you want to include predictor+data+interdiction keys and evaluation settings.
    *DATA_KEYS,
    *INTD_KEYS,
    *PRED_KEYS,
    "num_seeds",
    "seed_sweep_offset",
}


# Keys that should NEVER affect hashes even if present in cfg
# (you can add things like "logdir", "timestamp", "run_id", etc.)
EPHEMERAL_KEYS: Set[str] = {
    "timestamp",
    "time",
    "logdir",
    "logger",
    "wandb",
    "notes",
    "run_id",
    "git_commit",
}

# Optional: map new key names to old key names (or vice versa) to avoid duplication
# when you refactor config naming.
# Example:
#   If you renamed "training.learning_rate" -> "training.lr",
#   you can make both map to a single canonical key "training.lr".
KEY_ALIASES: Dict[str, str] = {
    # "training.learning_rate": "training.lr",
}


# =============================================================================
# 2) Artefact types and mapping to which hash subset to use
# =============================================================================

class Artefacts(Enum):
    """All supported artefact types that can be cached."""
    DATA = "DATA"
    INTD_RND = "INTD_RND"
    INTD_ADV = "INTD_ADV" # rnd and adv use same INTD_KEYS but different prefixes
    PRED = "PRED"
    RESULT = "RESULT"
    FIG = "FIG"  # figures are keyed off RESULT hash by design


# Map artefact -> which cfg subset hash type it uses.
# "head" means: hash of ALL cfg (after canonicalization / dropping ephemeral keys)
ARTEFACT_TYPE: Dict[Artefacts, str] = {
    Artefacts.DATA: "data",
    Artefacts.INTD_RND: "intd",
    Artefacts.INTD_ADV: "intd",
    Artefacts.PRED: "pred",
    Artefacts.RESULT: "result",
    # Artefacts.FIG is not used in read_cache (figs handled separately)
}

ARTEFACT_EXTENSION: Dict[Artefacts, str] = {
    Artefacts.DATA: ".npz",
    Artefacts.INTD_RND: ".pkl",
    Artefacts.INTD_ADV: ".pkl",
    Artefacts.PRED: ".pkl",
    Artefacts.RESULT: ".pkl",
    Artefacts.FIG: ".png",  # default figure format
}


# =============================================================================
# 3) Core canonicalization & hashing helpers
# =============================================================================

def _root_path() -> Path:
    """
    Return absolute path to the store_data folder.

    We locate it relative to THIS file to make the cache portable.
    If you prefer an environment variable override, uncomment below.
    """
    # Optional override via env var:
    # env = os.getenv("SIM_CACHE_ROOT", "").strip()
    # if env:
    #     p = Path(env).expanduser().resolve()
    #     p.mkdir(parents=True, exist_ok=True)
    #     return p

    root = Path(__file__).resolve().parents[3] / "store_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get_path(artifact: Artefacts) -> Path:
    """
    Return absolute path to the folder for a given artefact type.

    Examples:
      DATA -> store_data/datasets/
      INTD_* -> store_data/interdictions/
      PRED -> store_data/predictors/
      RESULT -> store_data/results/
      FIG -> store_data/figures/
    """
    root = _root_path()
    if artifact == Artefacts.DATA:
        p = root / "datasets"
    elif artifact in (Artefacts.INTD_RND, Artefacts.INTD_ADV):
        p = root / "interdictions"
    elif artifact == Artefacts.PRED:
        p = root / "predictors"
    elif artifact == Artefacts.RESULT:
        p = root / "results"
    elif artifact == Artefacts.FIG:
        p = root / "figures"
    else:
        raise ValueError(f"Unknown artefact type: {artifact}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_numpy_obj(x: Any) -> bool:
    """Return True if x looks like a numpy scalar/array without requiring numpy."""
    mod = type(x).__module__.lower()
    return "numpy" in mod


def _to_builtin(x: Any) -> Any:
    """
    Convert x into JSON-serializable builtin types deterministically.

    This is critical for stable hashes: identical semantic cfg -> identical string.
    """
    # Dataclasses -> dict
    if dataclasses.is_dataclass(x):
        return _to_builtin(dataclasses.asdict(x))

    # pathlib.Path -> absolute resolved string
    if isinstance(x, Path):
        return str(x.expanduser().resolve())

    # Dict-like
    if isinstance(x, dict):
        # Convert keys to strings and recurse
        return {str(k): _to_builtin(v) for k, v in x.items()}

    # Iterables
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, set):
        # Sets are unordered; sort for deterministic representation
        return sorted(_to_builtin(v) for v in x)

    # Numpy scalars/arrays
    if _is_numpy_obj(x):
        # scalar: try .item()
        if hasattr(x, "item"):
            try:
                return _to_builtin(x.item())
            except Exception:
                pass
        # array: try .tolist()
        if hasattr(x, "tolist"):
            try:
                return _to_builtin(x.tolist())
            except Exception:
                pass
        # last resort
        return repr(x)

    # Floats: use stable formatting (IEEE double round-trip is safe with 17 sig figs)
    if isinstance(x, float):
        return float(f"{x:.17g}")

    # JSON primitives
    if x is None or isinstance(x, (str, int, bool)):
        return x

    # Fallback: represent unknown objects deterministically
    # NOTE: repr() can still contain memory addresses for some objects.
    # If that happens for your cfg, explicitly handle that object type above.
    return repr(x)


def _as_mapping(cfg: Any) -> Dict[str, Any]:
    """
    Convert cfg to a nested dict-like mapping.

    Supports:
    - dict
    - dataclass instances
    - objects with __dict__ (simple namespaces / argparse Namespace)
    """
    if isinstance(cfg, dict):
        return cfg
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    raise TypeError(
        "cfg must be a dict, dataclass, or namespace-like object (has __dict__)."
    )


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested dict into dotted paths:
      {"a": {"b": 1}, "c": 2} -> {"a.b": 1, "c": 2}

    Lists are kept as lists (not expanded) to avoid exploding keys.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key))
        else:
            out[key] = v
    return out


def _apply_aliases(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply KEY_ALIASES so renamed keys can map to a single canonical key.

    If both old and new exist, the canonical key wins (you can change policy).
    """
    if not KEY_ALIASES:
        return flat

    out = dict(flat)
    for old, new in KEY_ALIASES.items():
        if old in out and new not in out:
            out[new] = out[old]
        # keep old as well? Usually no; drop it to avoid ambiguity.
        if old in out:
            del out[old]
    return out


def _drop_ephemeral(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop ephemeral keys by:
    - exact match on last path segment (e.g., "training.logdir" -> "logdir")
    - exact match on full key as well
    """
    out: Dict[str, Any] = {}
    for k, v in flat.items():
        leaf = k.split(".")[-1]
        if k in EPHEMERAL_KEYS or leaf in EPHEMERAL_KEYS:
            continue
        out[k] = v
    return out


def _canonical_subset(cfg: Any, keys: Optional[Set[str]]) -> Dict[str, Any]:
    """
    Produce a deterministic, JSON-safe dict representing cfg restricted to `keys`.

    - Convert cfg to nested mapping
    - Flatten to dotted paths
    - Apply key aliases
    - Drop ephemeral keys
    - If keys is None: keep everything ("head")
    - Else: take intersection between existing cfg paths and requested key set
    """
    nested = _as_mapping(cfg)
    flat = _flatten_dict(_to_builtin(nested))
    flat = _apply_aliases(flat)
    flat = _drop_ephemeral(flat)

    if keys is None:
        # "head" subset: keep everything (minus ephemeral)
        subset = flat
    else:
        # Only keys that exist in cfg (intersection); ignore missing keys safely.
        subset = {k: flat[k] for k in keys if k in flat}

    # Sort keys deterministically by building an ordered dict-like mapping.
    # (In Python 3.7+, dict preserves insertion order.)
    return {k: subset[k] for k in sorted(subset.keys())}


def _unique_hash(cfg: Any, type: str = "head") -> str:
    """
    Compute a deterministic hash for cfg.

    Hash types:
      - "head"   : all cfg keys (minus ephemeral), canonicalized
      - "data"   : only DATA_KEYS ∩ cfg_keys
      - "intd"   : only INTD_KEYS ∩ cfg_keys
      - "pred"   : only PRED_KEYS ∩ cfg_keys
      - "result" : only RESULT_KEYS ∩ cfg_keys

    Properties:
      - Deterministic: same semantic cfg -> same hash
      - Sensitive: different relevant cfg -> different hash
      - Robust: adding unrelated cfg fields won't change subset hashes

    Important edge case:
      If a subset becomes empty because none of the expected keys are present,
      we raise to prevent creating a "hash of nothing".
    """
    if type == "head":
        subset = _canonical_subset(cfg, keys=None)
    elif type == "data":
        subset = _canonical_subset(cfg, keys=DATA_KEYS)
    elif type == "intd":
        subset = _canonical_subset(cfg, keys=INTD_KEYS)
    elif type == "pred":
        subset = _canonical_subset(cfg, keys=PRED_KEYS)
    elif type == "result":
        subset = _canonical_subset(cfg, keys=RESULT_KEYS)
    else:
        raise ValueError(f"Unknown hash type: {type}")

    if type != "head" and len(subset) == 0:
        raise ValueError(
            f"Hash type '{type}' has empty intersection with cfg keys. "
            "Check your *_KEYS sets and cfg structure."
        )

    # Deterministic JSON string: sorted keys + compact separators
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # Use SHA-256: collisions are negligible for caching
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_meta(meta_path: Path, *, cfg: Any, hash_type: str, artefact: Artefacts) -> None:
    """Write sidecar metadata JSON for traceability."""
    meta = {
        "created_unix": time.time(),
        "artefact": artefact.value,
        "hash_type": hash_type,
        "hash": _unique_hash(cfg, type=hash_type),
        "head_hash": _unique_hash(cfg, type="head"),
        "canonical_cfg_subset": _canonical_subset(cfg, keys=None if hash_type == "head" else {
            "data": DATA_KEYS,
            "intd": INTD_KEYS,
            "pred": PRED_KEYS,
            "result": RESULT_KEYS,
        }[hash_type]),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# 4) Filename conventions and (de)serialization
# =============================================================================

def _artifact_prefix(artifact: Artefacts) -> str:
    """Return file prefix for each artefact type."""
    if artifact == Artefacts.DATA:
        return "data"
    if artifact == Artefacts.INTD_RND:
        return "rnd"
    if artifact == Artefacts.INTD_ADV:
        return "adv"
    if artifact == Artefacts.PRED:
        return "pred"
    if artifact == Artefacts.RESULT:
        return "result"
    raise ValueError(f"Unexpected artefact for prefix: {artifact}")


def _artifact_file_paths(cfg: Any, artifact: Artefacts) -> Tuple[Path, Path]:
    """
    Compute (data_path, meta_path) for an artefact based on cfg and artefact type.
    """
    if artifact == Artefacts.FIG:
        raise ValueError("Figures are handled by write_fig/read_fig separately.")

    hash_type = ARTEFACT_TYPE[artifact]
    h = _unique_hash(cfg, type=hash_type)
    folder = _get_path(artifact)
    prefix = _artifact_prefix(artifact)

    # Decide extension by artefact type
    if artifact in ARTEFACT_EXTENSION.keys():
        ext = ARTEFACT_EXTENSION[artifact]
    else:
        raise ValueError(f"Unexpected artefact for extension: {artifact}")

    data_path = folder / f"{prefix}_{h}{ext}"
    meta_path = folder / f"{prefix}_{h}.meta.json"
    return data_path, meta_path


def _pickle_dump(path: Path, obj: Any) -> None:
    """Write object via pickle safely."""
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path: Path) -> Any:
    """Load object via pickle."""
    with path.open("rb") as f:
        return pickle.load(f)


# =============================================================================
# 5) Public API: read/write cache for core artefacts
# =============================================================================

def read_cache(cfg: Any, artifact: Artefacts) -> Optional[Any]:
    """
    Load a cached artefact given cfg and artefact type.

    artefact excludes FIG (figures handled separately).

    Returns:
      - None if file does not exist
      - Otherwise the loaded artefact object
    """
    if artifact == Artefacts.FIG:
        raise ValueError("Use read_fig for figures.")

    data_path, _meta_path = _artifact_file_paths(cfg, artifact)

    if not data_path.exists():
        return None

    if ARTEFACT_EXTENSION[artifact] == ".npz":
        if np is None:
            raise RuntimeError("NumPy is required to read .npz dataset files.")
        # Expect stored arrays under keys 'feats' and 'costs'
        with np.load(data_path, allow_pickle=False) as z:
            return {"feats": z["feats"], "costs": z["costs"]}
    if ARTEFACT_EXTENSION[artifact] == ".pkl":
        # All other artefacts stored as pickle
        return _pickle_load(data_path)
    else:
        raise ValueError(f"Unsupported artefact extension: {ARTEFACT_EXTENSION[artifact]}")


def write_data(cfg: Any, feats: Any, costs: Any) -> Path:
    """
    Store dataset (features, costs) under datasets/data_<data_hash>.npz.

    - Raises FileExistsError if cache file already exists.
    - Returns the written file path.
    """
    if np is None:
        raise RuntimeError("NumPy is required to write .npz dataset files.")

    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.DATA)
    if data_path.exists():
        raise FileExistsError(f"Dataset already exists: {data_path.name}")

    # Store as compressed .npz
    np.savez_compressed(data_path, feats=np.asarray(feats), costs=np.asarray(costs))
    _write_meta(meta_path, cfg=cfg, hash_type="data", artefact=Artefacts.DATA)
    return data_path


def write_rnd_intd(cfg: Any, intd: Any) -> Path:
    """
    Store random interdictions under interdictions/rnd_<intd_hash>.pkl.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.INTD_RND)
    if data_path.exists():
        raise FileExistsError(f"Random interdictions already exist: {data_path.name}")

    _pickle_dump(data_path, intd)
    _write_meta(meta_path, cfg=cfg, hash_type="intd", artefact=Artefacts.INTD_RND)
    return data_path


def write_adv_intd(cfg: Any, intd: Any) -> Path:
    """
    Store adversarial interdictions under interdictions/adv_<intd_hash>.pkl.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.INTD_ADV)
    if data_path.exists():
        raise FileExistsError(f"Adversarial interdictions already exist: {data_path.name}")

    _pickle_dump(data_path, intd)
    _write_meta(meta_path, cfg=cfg, hash_type="intd", artefact=Artefacts.INTD_ADV)
    return data_path


def write_pred(cfg: Any, pred_model: Any) -> Path:
    """
    Store predictor model under predictors/pred_<pred_hash>.pkl.

    Note: If pred_model contains GPU tensors, open file portability can suffer.
          Consider saving state_dicts instead (PyTorch) and reloading separately.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.PRED)
    if data_path.exists():
        raise FileExistsError(f"Predictor already exists: {data_path.name}")

    _pickle_dump(data_path, pred_model)
    _write_meta(meta_path, cfg=cfg, hash_type="pred", artefact=Artefacts.PRED)
    return data_path


def write_results(cfg: Any, results: Any) -> Path:
    """
    Store results under results/result_<result_hash>.pkl.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.RESULT)
    if data_path.exists():
        raise FileExistsError(f"Results already exist: {data_path.name}")

    _pickle_dump(data_path, results)
    _write_meta(meta_path, cfg=cfg, hash_type="result", artefact=Artefacts.RESULT)
    return data_path


# =============================================================================
# 6) Figures: keyed by RESULT hash (to avoid storing too many)
# =============================================================================

def _sanitize_fig_type(fig_type: str) -> str:
    """
    Make fig_type filesystem-friendly.
    Keep it short and predictable so figure filenames are stable.
    """
    fig_type = fig_type.strip().lower()
    fig_type = re.sub(r"[^a-z0-9_\-]+", "_", fig_type)
    fig_type = re.sub(r"_+", "_", fig_type).strip("_")
    return fig_type or "figure"


def write_fig(cfg: Any, fig: Any) -> Path:
    """
    Store a figure under figures/<fig_type>_<result_hash>.png.

    Design decision:
      - Figures are keyed on the RESULT hash to avoid combinatorial figure storage.
        If you rerun training but get same results-hash (same cfg subset), the
        figure name is identical.

    fig_type inference:
      - If fig has attribute 'fig_type' (string), use it.
      - Else if fig has 'get_label()' (matplotlib Figure), use it if non-empty.
      - Else fallback to 'figure'.

    fig serialization:
      - If fig is a matplotlib Figure (has savefig), we call fig.savefig(...).
      - Otherwise we pickle the object into a .pkl (fallback), but filename
        still uses <fig_type>_<result_hash>.
    """
    result_hash = _unique_hash(cfg, type="result")
    folder = _get_path(Artefacts.FIG)

    # Determine figure type
    fig_type = getattr(fig, "fig_type", None)
    if not fig_type and hasattr(fig, "get_label"):
        try:
            label = fig.get_label()
            fig_type = label if isinstance(label, str) and label.strip() else None
        except Exception:
            fig_type = None
    fig_type = _sanitize_fig_type(fig_type or "figure")

    # Default to PNG if matplotlib-like
    if hasattr(fig, "savefig"):
        out_path = folder / f"{fig_type}_{result_hash}" + ARTEFACT_EXTENSION[Artefacts.FIG]
        meta_path = folder / f"{fig_type}_{result_hash}.meta.json"
        if out_path.exists():
            raise FileExistsError(f"Figure already exists: {out_path.name}")

        # Save with tight layout; you can add dpi if you want larger images.
        fig.savefig(out_path, bbox_inches="tight")
        _write_meta(meta_path, cfg=cfg, hash_type="result", artefact=Artefacts.FIG)
        return out_path

    # Fallback: pickle unknown figure object
    out_path = folder / f"{fig_type}_{result_hash}.pkl"
    meta_path = folder / f"{fig_type}_{result_hash}.meta.json"
    if out_path.exists():
        raise FileExistsError(f"Figure object already exists: {out_path.name}")

    _pickle_dump(out_path, fig)
    _write_meta(meta_path, cfg=cfg, hash_type="result", artefact=Artefacts.FIG)
    return out_path


def read_fig(cfg: Any, fig_type: Union[str, Sequence[str]] = "all") -> Union[Dict[str, List[Path]], List[Path]]:
    """
    Retrieve figure file paths for a given cfg's RESULT hash.

    Parameters
    ----------
    cfg:
      config instance used to compute the RESULT hash

    fig_type:
      - "all" -> return dict mapping {fig_type: [paths...]} for this result_hash
      - "<fig_type>" -> return list of matching paths for that type
      - ["t1","t2",...] -> return dict mapping those types to lists of paths

    Returns
    -------
    Either:
      - list[Path] for a single fig_type
      - dict[str, list[Path]] for "all" or list of fig types

    Notes
    -----
    We do not "plot" automatically (no implicit UI side-effects).
    You can open the returned PNGs or load them as needed.
    """
    folder = _get_path(Artefacts.FIG)
    result_hash = _unique_hash(cfg, type="result")

    # Helper to collect paths for a given figure type
    def _collect_one(ft: str) -> List[Path]:
        ft = _sanitize_fig_type(ft)
        # Match both png and pkl
        png = sorted(folder.glob(f"{ft}_{result_hash}.png"))
        pkl = sorted(folder.glob(f"{ft}_{result_hash}.pkl"))
        return png + pkl

    if isinstance(fig_type, str):
        if fig_type == "all":
            # Collect all figure types for this result hash
            matches = sorted(folder.glob(f"*_{result_hash}.png")) + sorted(folder.glob(f"*_{result_hash}.pkl"))
            out: Dict[str, List[Path]] = {}
            for p in matches:
                # file is "<fig_type>_<hash>.<ext>"
                ft = p.name.rsplit("_", 1)[0]
                out.setdefault(ft, []).append(p)
            return out
        else:
            return _collect_one(fig_type)

    # Sequence of fig types
    out = {str(ft): _collect_one(str(ft)) for ft in fig_type}
    return out