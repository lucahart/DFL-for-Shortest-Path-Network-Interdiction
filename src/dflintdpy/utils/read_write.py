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
import shutil
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
    "load_real_world_graph",
    "source_node",
    "target_node",
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

PRED_BASE_KEYS: Set[str] = {
    *DATA_KEYS,  # predictor training depends on data generation
    "batch_size",
    "pfl_epochs",
    "dfl_epochs",
    "pfl_lr",
    "dfl_lr",
    "hidden_size_1",
    "max_lr_reductions",
    "surrogate_underprediction_penalty_weight",
    "surrogate_underprediction_margin",
}

# Interdiction-specific keys that can be excluded for non-interdiction predictors.
PRED_INTD_EXTRA_KEYS: Set[str] = INTD_KEYS.difference(DATA_KEYS)

# Default predictor hash keys (keeps previous behavior for most predictor tags).
PRED_KEYS: Set[str] = {
    *PRED_BASE_KEYS,
    *PRED_INTD_EXTRA_KEYS,
}

# Predictor-tag overrides for cache hashing.
# - "pfl" and "dfl" should not depend on interdiction-specific cfg.
# - other tags (e.g., adfl/rdfl/madfl/mrdfl) use default PRED_KEYS.
PRED_TAG_KEY_OVERRIDES: Dict[str, Set[str]] = {
    "pfl": set(PRED_BASE_KEYS),
    "dfl": set(PRED_BASE_KEYS),
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

# Maps old config key names to their canonical replacements.
# Any old key encountered during cache-hash computation is folded into the
# new name so that results generated before the rename still hit the cache.
KEY_ALIASES: Dict[str, str] = {
    "po_epochs":  "pfl_epochs",
    "spo_epochs": "dfl_epochs",
    "po_lr":      "pfl_lr",
    "spo_lr":     "dfl_lr",
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


@dataclasses.dataclass
class CacheReplaceOptions:
    """
    Central overwrite policy for cache writers/readers.

    Set once via `set_cache_replace_options(...)`, then pass around (or rely on
    module-global defaults) from training/data setup functions.
    """
    replace_data: bool = False
    replace_intd_rnd: bool = False
    replace_intd_adv: bool = False
    replace_pred: bool = False
    replace_result: bool = False
    replace_fig: bool = False
    archive_replaced: bool = True

    def for_artifact(self, artefact: Artefacts) -> bool:
        if artefact == Artefacts.DATA:
            return self.replace_data
        if artefact == Artefacts.INTD_RND:
            return self.replace_intd_rnd
        if artefact == Artefacts.INTD_ADV:
            return self.replace_intd_adv
        if artefact == Artefacts.PRED:
            return self.replace_pred
        if artefact == Artefacts.RESULT:
            return self.replace_result
        if artefact == Artefacts.FIG:
            return self.replace_fig
        return False


_GLOBAL_CACHE_REPLACE_OPTIONS = CacheReplaceOptions()


def get_cache_replace_options() -> CacheReplaceOptions:
    """Return a copy of current global cache replace options."""
    return dataclasses.replace(_GLOBAL_CACHE_REPLACE_OPTIONS)


def set_cache_replace_options(**kwargs: Any) -> CacheReplaceOptions:
    """
    Update module-global cache replace options.

    Example:
      set_cache_replace_options(replace_data=True, replace_pred=True)
    """
    global _GLOBAL_CACHE_REPLACE_OPTIONS
    updated = dataclasses.replace(_GLOBAL_CACHE_REPLACE_OPTIONS)
    for key, value in kwargs.items():
        if not hasattr(updated, key):
            raise ValueError(f"Unknown cache replace option: {key}")
        setattr(updated, key, bool(value))
    _GLOBAL_CACHE_REPLACE_OPTIONS = updated
    return get_cache_replace_options()


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


def _hash_from_subset(subset: Dict[str, Any]) -> str:
    """Create deterministic SHA-256 hash from a canonical cfg subset."""
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pred_keys_for_tag(artifact_tag: Optional[str]) -> Set[str]:
    """
    Return predictor-hash keys for a cache tag.

    Unknown tags default to full `PRED_KEYS` (includes interdiction keys).
    """
    if artifact_tag is None:
        return set(PRED_KEYS)
    tag = _sanitize_artifact_tag(str(artifact_tag))
    return set(PRED_TAG_KEY_OVERRIDES.get(tag, PRED_KEYS))


def _unique_pred_hash(cfg: Any, artifact_tag: Optional[str]) -> str:
    """Compute predictor hash with tag-specific key selection."""
    subset = _canonical_subset(cfg, keys=_pred_keys_for_tag(artifact_tag))
    if len(subset) == 0:
        raise ValueError(
            "Predictor hash has empty intersection with cfg keys. "
            "Check predictor key sets and cfg structure."
        )
    return _hash_from_subset(subset)


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

    return _hash_from_subset(subset)


def _write_meta(
    meta_path: Path,
    *,
    cfg: Any,
    hash_type: str,
    artefact: Artefacts,
    artifact_tag: Optional[str] = None,
) -> None:
    """Write sidecar metadata JSON for traceability."""
    if hash_type == "pred":
        subset_keys = _pred_keys_for_tag(artifact_tag)
        canonical_subset = _canonical_subset(cfg, keys=subset_keys)
        hash_value = _hash_from_subset(canonical_subset)
    else:
        subset_keys = None if hash_type == "head" else {
            "data": DATA_KEYS,
            "intd": INTD_KEYS,
            "pred": PRED_KEYS,
            "result": RESULT_KEYS,
        }[hash_type]
        canonical_subset = _canonical_subset(cfg, keys=subset_keys)
        hash_value = _hash_from_subset(canonical_subset)

    meta = {
        "created_unix": time.time(),
        "artefact": artefact.value,
        "artifact_tag": artifact_tag,
        "hash_type": hash_type,
        "hash": hash_value,
        "head_hash": _unique_hash(cfg, type="head"),
        "canonical_cfg_subset": canonical_subset,
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


def _sanitize_artifact_tag(tag: str) -> str:
    """Make a cache tag filesystem-friendly and deterministic."""
    tag = tag.strip().lower()
    tag = re.sub(r"[^a-z0-9_\-]+", "_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    if not tag:
        raise ValueError("artifact tag must contain at least one alphanumeric character")
    return tag


def _artifact_file_paths(
    cfg: Any,
    artifact: Artefacts,
    artifact_tag: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Compute (data_path, meta_path) for an artefact based on cfg and artefact type.
    """
    if artifact == Artefacts.FIG:
        raise ValueError("Figures are handled by write_fig/read_fig separately.")

    folder = _get_path(artifact)
    prefix = _artifact_prefix(artifact)

    # Decide extension by artefact type
    if artifact in ARTEFACT_EXTENSION.keys():
        ext = ARTEFACT_EXTENSION[artifact]
    else:
        raise ValueError(f"Unexpected artefact for extension: {artifact}")

    if artifact == Artefacts.PRED:
        if artifact_tag is None:
            raise ValueError("artifact_tag is required for predictor artefacts")
        tag = _sanitize_artifact_tag(artifact_tag)
        suffix = f"_{tag}"
        h = _unique_pred_hash(cfg, artifact_tag=tag)
    else:
        suffix = ""
        hash_type = ARTEFACT_TYPE[artifact]
        h = _unique_hash(cfg, type=hash_type)

    data_path = folder / f"{prefix}{suffix}_{h}{ext}"
    meta_path = folder / f"{prefix}{suffix}_{h}.meta.json"
    return data_path, meta_path


def _pickle_dump(path: Path, obj: Any) -> None:
    """Write object via pickle safely."""
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path: Path) -> Any:
    """Load object via pickle."""
    with path.open("rb") as f:
        return pickle.load(f)


def _next_archive_hash(base_hash: str) -> str:
    """Derive a unique archival hash from the previous hash and current time."""
    raw = f"{base_hash}|replaced|{time.time_ns()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _archive_existing_artifact(
    data_path: Path,
    meta_path: Path,
    *,
    reason: str,
    enabled: bool = True,
) -> Optional[Dict[str, str]]:
    """
    Archive an existing artefact before overwrite.

    The archived copy gets a new unique hash and an `update_tag` so it remains
    retrievable later while freeing the original hash path for the replacement.
    """
    if not enabled:
        # No archival requested, caller may overwrite directly.
        return None

    data_exists = data_path.exists()
    meta_exists = meta_path.exists()
    if not data_exists and not meta_exists:
        return None

    base_stem = data_path.stem if data_exists else meta_path.name.removesuffix(".meta.json")
    if "_" not in base_stem:
        raise ValueError(f"Cannot archive malformed artefact filename: {base_stem}")
    prefix, old_hash = base_stem.rsplit("_", 1)

    archive_hash = _next_archive_hash(old_hash)
    archived_data_path = data_path.with_name(f"{prefix}_{archive_hash}{data_path.suffix}")
    archived_meta_path = meta_path.with_name(f"{prefix}_{archive_hash}.meta.json")

    # In the unlikely case of collision, refresh hash.
    while archived_data_path.exists() or archived_meta_path.exists():
        archive_hash = _next_archive_hash(old_hash)
        archived_data_path = data_path.with_name(f"{prefix}_{archive_hash}{data_path.suffix}")
        archived_meta_path = meta_path.with_name(f"{prefix}_{archive_hash}.meta.json")

    # Move data payload first so the bytes are preserved regardless of meta parsing.
    if data_exists:
        data_path.replace(archived_data_path)

    archived_meta_obj: Dict[str, Any]
    if meta_exists:
        try:
            archived_meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(archived_meta_obj, dict):
                archived_meta_obj = {"raw_meta": archived_meta_obj}
        except Exception:
            archived_meta_obj = {"raw_meta_text": meta_path.read_text(encoding="utf-8", errors="replace")}
        finally:
            try:
                meta_path.unlink(missing_ok=True)
            except TypeError:
                if meta_path.exists():
                    meta_path.unlink()
    else:
        archived_meta_obj = {}

    update_tag = f"replaced_{int(time.time())}"
    archived_meta_obj["hash"] = archive_hash
    archived_meta_obj["archived_from_hash"] = old_hash
    archived_meta_obj["update_tag"] = update_tag
    archived_meta_obj["archived_unix"] = time.time()
    archived_meta_obj["archive_reason"] = reason
    archived_meta_obj["archived_from_data_path"] = data_path.name
    archived_meta_obj["archived_from_meta_path"] = meta_path.name

    archived_meta_path.write_text(
        json.dumps(archived_meta_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "archived_hash": archive_hash,
        "archived_data_path": str(archived_data_path),
        "archived_meta_path": str(archived_meta_path),
        "update_tag": update_tag,
    }


# =============================================================================
# 5) Public API: read/write cache for core artefacts
# =============================================================================

def read_cache(cfg: Any, artifact: Artefacts, artifact_tag: Optional[str] = None) -> Optional[Any]:
    """
    Load a cached artefact given cfg and artefact type.

    artefact excludes FIG (figures handled separately).

    Returns:
      - None if file does not exist
      - Otherwise the loaded artefact object
    """
    if artifact == Artefacts.FIG:
        raise ValueError("Use read_fig for figures.")

    data_path, _meta_path = _artifact_file_paths(cfg, artifact, artifact_tag)

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


def write_data(
    cfg: Any,
    feats: Any,
    costs: Any,
    *,
    replace: bool = False,
    archive_replaced: Optional[bool] = None,
) -> Path:
    """
    Store dataset (features, costs) under datasets/data_<data_hash>.npz.

    - Raises FileExistsError if cache file already exists.
    - Returns the written file path.
    """
    if np is None:
        raise RuntimeError("NumPy is required to write .npz dataset files.")

    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.DATA)
    if data_path.exists() or meta_path.exists():
        if not replace:
            raise FileExistsError(f"Dataset already exists: {data_path.name}")
        if archive_replaced is None:
            archive_replaced = get_cache_replace_options().archive_replaced
        _archive_existing_artifact(
            data_path,
            meta_path,
            reason="replace=True in write_data",
            enabled=bool(archive_replaced),
        )

    # Store as compressed .npz
    np.savez_compressed(data_path, feats=np.asarray(feats), costs=np.asarray(costs))
    _write_meta(meta_path, cfg=cfg, hash_type="data", artefact=Artefacts.DATA)
    return data_path


def write_rnd_intd(
    cfg: Any,
    intd: Any,
    *,
    replace: bool = False,
    archive_replaced: Optional[bool] = None,
) -> Path:
    """
    Store random interdictions under interdictions/rnd_<intd_hash>.pkl.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.INTD_RND)
    if data_path.exists() or meta_path.exists():
        if not replace:
            raise FileExistsError(f"Random interdictions already exist: {data_path.name}")
        if archive_replaced is None:
            archive_replaced = get_cache_replace_options().archive_replaced
        _archive_existing_artifact(
            data_path,
            meta_path,
            reason="replace=True in write_rnd_intd",
            enabled=bool(archive_replaced),
        )

    _pickle_dump(data_path, intd)
    _write_meta(meta_path, cfg=cfg, hash_type="intd", artefact=Artefacts.INTD_RND)
    return data_path


def write_adv_intd(
    cfg: Any,
    intd: Any,
    *,
    replace: bool = False,
    archive_replaced: Optional[bool] = None,
) -> Path:
    """
    Store adversarial interdictions under interdictions/adv_<intd_hash>.pkl.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.INTD_ADV)
    if data_path.exists() or meta_path.exists():
        if not replace:
            raise FileExistsError(f"Adversarial interdictions already exist: {data_path.name}")
        if archive_replaced is None:
            archive_replaced = get_cache_replace_options().archive_replaced
        _archive_existing_artifact(
            data_path,
            meta_path,
            reason="replace=True in write_adv_intd",
            enabled=bool(archive_replaced),
        )

    _pickle_dump(data_path, intd)
    _write_meta(meta_path, cfg=cfg, hash_type="intd", artefact=Artefacts.INTD_ADV)
    return data_path


def write_pred(
    cfg: Any,
    pred_model: Any,
    artifact_tag: str,
    *,
    replace: bool = False,
    archive_replaced: Optional[bool] = None,
) -> Path:
    """
    Store predictor model under predictors/pred_<tag>_<pred_hash>.pkl.

    Note: If pred_model contains GPU tensors, open file portability can suffer.
          Consider saving state_dicts instead (PyTorch) and reloading separately.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.PRED, artifact_tag)
    if data_path.exists() or meta_path.exists():
        if not replace:
            raise FileExistsError(f"Predictor already exists: {data_path.name}")
        if archive_replaced is None:
            archive_replaced = get_cache_replace_options().archive_replaced
        _archive_existing_artifact(
            data_path,
            meta_path,
            reason="replace=True in write_pred",
            enabled=bool(archive_replaced),
        )

    _pickle_dump(data_path, pred_model)
    _write_meta(
        meta_path,
        cfg=cfg,
        hash_type="pred",
        artefact=Artefacts.PRED,
        artifact_tag=artifact_tag,
    )
    return data_path


def write_results(
    cfg: Any,
    results: Any,
    *,
    replace: bool = False,
    archive_replaced: Optional[bool] = None,
) -> Path:
    """
    Store results under results/result_<result_hash>.pkl.
    """
    data_path, meta_path = _artifact_file_paths(cfg, Artefacts.RESULT)
    if data_path.exists() or meta_path.exists():
        if not replace:
            raise FileExistsError(f"Results already exist: {data_path.name}")
        if archive_replaced is None:
            archive_replaced = get_cache_replace_options().archive_replaced
        _archive_existing_artifact(
            data_path,
            meta_path,
            reason="replace=True in write_results",
            enabled=bool(archive_replaced),
        )

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


def write_fig(
    cfg: Any,
    fig: Any,
    *,
    replace: bool = False,
    archive_replaced: Optional[bool] = None,
) -> Path:
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
        if out_path.exists() or meta_path.exists():
            if not replace:
                raise FileExistsError(f"Figure already exists: {out_path.name}")
            if archive_replaced is None:
                archive_replaced = get_cache_replace_options().archive_replaced
            _archive_existing_artifact(
                out_path,
                meta_path,
                reason="replace=True in write_fig",
                enabled=bool(archive_replaced),
            )

        # Save with tight layout; you can add dpi if you want larger images.
        fig.savefig(out_path, bbox_inches="tight")
        _write_meta(meta_path, cfg=cfg, hash_type="result", artefact=Artefacts.FIG)
        return out_path

    # Fallback: pickle unknown figure object
    out_path = folder / f"{fig_type}_{result_hash}.pkl"
    meta_path = folder / f"{fig_type}_{result_hash}.meta.json"
    if out_path.exists() or meta_path.exists():
        if not replace:
            raise FileExistsError(f"Figure object already exists: {out_path.name}")
        if archive_replaced is None:
            archive_replaced = get_cache_replace_options().archive_replaced
        _archive_existing_artifact(
            out_path,
            meta_path,
            reason="replace=True in write_fig",
            enabled=bool(archive_replaced),
        )

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


# =============================================================================
# 7) Seed-repair utility for predictor/dataset cache files
# =============================================================================

def _seed_triplet_from_sweep_seed(seed: int) -> Tuple[int, int, int]:
    """
    Reproduce the simulator's seed expansion:
      np.random.seed(seed); np.random.randint(0, 150, 3)
    """
    local_np = np
    if local_np is None:
        try:
            import numpy as local_np  # type: ignore
        except Exception as exc:
            raise RuntimeError("NumPy is required to reproduce sweep seed triplets.") from exc
    rng = local_np.random.RandomState(int(seed))
    vals = rng.randint(0, 150, 3).tolist()
    return int(vals[0]), int(vals[1]), int(vals[2])


def _safe_int(value: Any) -> Optional[int]:
    """Best-effort conversion to int; return None if conversion fails."""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 digest for conflict checks when target files already exist."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_repair_history(meta: Dict[str, Any], repair_event: Dict[str, Any]) -> None:
    """Append a repair event while preserving any existing metadata fields."""
    hist = meta.get("seed_repair_history")
    if isinstance(hist, list):
        hist.append(repair_event)
    elif hist is None:
        meta["seed_repair_history"] = [repair_event]
    else:
        meta["seed_repair_history"] = [hist, repair_event]


def _seed_lookup(candidate_seeds: Sequence[int]) -> Tuple[
    Dict[Tuple[int, int, int], List[int]],
    Dict[Tuple[int, int], List[int]],
]:
    """
    Build lookup tables:
      - full triplet (random, intd, loader) -> candidate seeds
      - data pair   (random, loader)        -> candidate seeds
    """
    triplet_map: Dict[Tuple[int, int, int], List[int]] = {}
    pair_map: Dict[Tuple[int, int], List[int]] = {}
    for seed in candidate_seeds:
        random_seed, intd_seed, loader_seed = _seed_triplet_from_sweep_seed(seed)
        triplet_map.setdefault((random_seed, intd_seed, loader_seed), []).append(seed)
        pair_map.setdefault((random_seed, loader_seed), []).append(seed)
    return triplet_map, pair_map


def repair_seed_sweep_hashes(
    *,
    seed_0: int,
    num_seeds: int,
    candidate_seeds: Optional[Sequence[int]] = None,
    root_dir: Optional[Union[str, Path]] = None,
    include_datasets: bool = True,
    include_predictors: bool = True,
    dry_run: bool = True,
    backup_originals: bool = True,
    backup_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Repair cached dataset/predictor artefacts where `seed` is incorrect but
    `random_seed` / `intd_seed` / `loader_seed` are correct.

    Method
    ------
    1) Build candidate seeds from either:
       - explicit `candidate_seeds`, OR
       - the sweep range: seed_idx + seed_0 for seed_idx in [0, num_seeds)
    2) Parse each `.meta.json` in datasets/predictors.
    3) Infer the expected sweep seed by matching:
       - predictors: (random_seed, intd_seed, loader_seed)
       - datasets:   (random_seed, loader_seed)
    4) Recompute the correct hash after replacing `seed`.
    5) Materialize corrected files first; then archive originals to backup.

    Safety
    ------
    - No destructive overwrite: existing target files are checked for content
      conflicts before writing.
    - If `backup_originals=True` and `dry_run=False`, originals are moved to a
      timestamped backup folder only after corrected files are in place.
    """
    if candidate_seeds is None:
        if num_seeds <= 0:
            raise ValueError("num_seeds must be > 0 when candidate_seeds is not provided.")
        candidate_seeds = [int(seed_0) + i for i in range(int(num_seeds))]
    else:
        candidate_seeds = [int(s) for s in candidate_seeds]

    # Deduplicate while preserving deterministic order.
    candidate_seeds = sorted(set(candidate_seeds))
    if not candidate_seeds:
        raise ValueError("No candidate seeds provided for repair.")

    root = Path(root_dir).expanduser().resolve() if root_dir is not None else _root_path()
    dataset_dir = root / "datasets"
    predictor_dir = root / "predictors"

    meta_files: List[Path] = []
    if include_datasets and dataset_dir.exists():
        meta_files.extend(sorted(dataset_dir.glob("*.meta.json")))
    if include_predictors and predictor_dir.exists():
        meta_files.extend(sorted(predictor_dir.glob("*.meta.json")))

    triplet_map, pair_map = _seed_lookup(candidate_seeds)
    lookup_key_counts = {
        "triplet_ambiguous": int(sum(1 for vals in triplet_map.values() if len(vals) > 1)),
        "pair_ambiguous": int(sum(1 for vals in pair_map.values() if len(vals) > 1)),
    }

    run_backup_dir: Optional[Path] = None
    if not dry_run and backup_originals:
        if backup_dir is None:
            run_backup_dir = root / f"seed_repair_backup_{int(time.time())}"
        else:
            run_backup_dir = Path(backup_dir).expanduser().resolve()
        run_backup_dir.mkdir(parents=True, exist_ok=True)

    report_entries: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    def _record(entry: Dict[str, Any]) -> None:
        status = str(entry.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        report_entries.append(entry)

    for meta_path in meta_files:
        entry: Dict[str, Any] = {"meta_path": str(meta_path)}

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            entry.update({"status": "skip_bad_json", "reason": str(exc)})
            _record(entry)
            continue

        artefact_raw = str(meta.get("artefact", "")).upper()
        if artefact_raw not in {"DATA", "PRED"}:
            entry.update({"status": "skip_non_target", "reason": f"artefact={artefact_raw}"})
            _record(entry)
            continue

        hash_type = str(meta.get("hash_type", "")).lower()
        if hash_type not in {"data", "pred"}:
            entry.update({"status": "skip_bad_hash_type", "reason": f"hash_type={hash_type}"})
            _record(entry)
            continue

        subset = meta.get("canonical_cfg_subset")
        if not isinstance(subset, dict):
            entry.update({"status": "skip_bad_subset", "reason": "canonical_cfg_subset missing or not dict"})
            _record(entry)
            continue

        random_seed = _safe_int(subset.get("random_seed"))
        loader_seed = _safe_int(subset.get("loader_seed"))
        intd_seed = _safe_int(subset.get("intd_seed"))
        current_seed = _safe_int(subset.get("seed"))

        if random_seed is None or loader_seed is None:
            entry.update({
                "status": "skip_missing_required_seeds",
                "reason": "random_seed/loader_seed missing",
            })
            _record(entry)
            continue

        # Predictors use full triplet; datasets use pair (intd_seed is not part of DATA hash).
        candidate_matches: List[int]
        if artefact_raw == "PRED":
            if intd_seed is None:
                entry.update({
                    "status": "skip_missing_required_seeds",
                    "reason": "intd_seed missing for predictor",
                })
                _record(entry)
                continue
            candidate_matches = triplet_map.get((random_seed, intd_seed, loader_seed), [])
            match_scope = "triplet"
        else:
            if intd_seed is not None:
                candidate_matches = triplet_map.get((random_seed, intd_seed, loader_seed), [])
                match_scope = "triplet"
            else:
                candidate_matches = pair_map.get((random_seed, loader_seed), [])
                match_scope = "pair"

        if len(candidate_matches) == 0:
            entry.update({
                "status": "skip_no_seed_match",
                "reason": f"no {match_scope} match in candidate seed set",
                "current_seed": current_seed,
                "random_seed": random_seed,
                "intd_seed": intd_seed,
                "loader_seed": loader_seed,
            })
            _record(entry)
            continue
        if len(candidate_matches) > 1:
            entry.update({
                "status": "skip_ambiguous_seed_match",
                "reason": f"multiple {match_scope} matches",
                "matches": candidate_matches,
            })
            _record(entry)
            continue

        inferred_seed = candidate_matches[0]
        if current_seed == inferred_seed:
            entry.update({"status": "unchanged", "seed": current_seed})
            _record(entry)
            continue

        fixed_subset = dict(subset)
        fixed_subset["seed"] = inferred_seed
        meta_artifact_tag = meta.get("artifact_tag")
        try:
            if hash_type == "pred":
                fixed_hash = _unique_pred_hash(fixed_subset, artifact_tag=meta_artifact_tag)
            else:
                fixed_hash = _unique_hash(fixed_subset, type=hash_type)
        except Exception as exc:
            entry.update({"status": "skip_hash_error", "reason": str(exc)})
            _record(entry)
            continue

        old_hash_from_meta = str(meta.get("hash", ""))
        meta_stem = meta_path.name.removesuffix(".meta.json")
        if "_" not in meta_stem:
            entry.update({"status": "skip_bad_filename", "reason": "cannot split filename hash"})
            _record(entry)
            continue
        filename_prefix, filename_hash = meta_stem.rsplit("_", 1)
        if not filename_hash:
            entry.update({"status": "skip_bad_filename", "reason": "empty filename hash"})
            _record(entry)
            continue

        if artefact_raw == "DATA":
            ext = ARTEFACT_EXTENSION[Artefacts.DATA]
        else:
            ext = ARTEFACT_EXTENSION[Artefacts.PRED]

        old_data_path = meta_path.with_name(f"{filename_prefix}_{filename_hash}{ext}")
        new_meta_path = meta_path.with_name(f"{filename_prefix}_{fixed_hash}.meta.json")
        new_data_path = old_data_path.with_name(f"{filename_prefix}_{fixed_hash}{ext}")

        fixed_meta = dict(meta)
        fixed_meta["hash"] = fixed_hash
        # Keep subset canonical and deterministic.
        subset_keys = DATA_KEYS if hash_type == "data" else _pred_keys_for_tag(meta_artifact_tag)
        fixed_meta["canonical_cfg_subset"] = _canonical_subset(fixed_subset, subset_keys)
        _append_repair_history(
            fixed_meta,
            {
                "repaired_unix": time.time(),
                "old_seed": current_seed,
                "new_seed": inferred_seed,
                "old_hash": old_hash_from_meta,
                "new_hash": fixed_hash,
                "match_scope": match_scope,
                "preserved_head_hash": meta.get("head_hash"),
            },
        )

        entry.update(
            {
                "status": "would_repair" if dry_run else "repaired",
                "old_seed": current_seed,
                "new_seed": inferred_seed,
                "old_hash_meta": old_hash_from_meta,
                "old_hash_filename": filename_hash,
                "new_hash": fixed_hash,
                "old_meta_path": str(meta_path),
                "new_meta_path": str(new_meta_path),
                "old_data_path": str(old_data_path),
                "new_data_path": str(new_data_path),
            }
        )

        if dry_run:
            _record(entry)
            continue

        # Conflict checks for target paths.
        conflict_reason: Optional[str] = None
        if new_data_path.exists() and old_data_path.exists():
            if _file_sha256(new_data_path) != _file_sha256(old_data_path):
                conflict_reason = "target data file exists with different content"
        if new_meta_path.exists():
            try:
                existing_meta = json.loads(new_meta_path.read_text(encoding="utf-8"))
                if existing_meta != fixed_meta:
                    conflict_reason = "target meta file exists with different content"
            except Exception:
                conflict_reason = "target meta file exists but cannot be parsed"

        if conflict_reason is not None:
            entry["status"] = "skip_conflict"
            entry["reason"] = conflict_reason
            _record(entry)
            continue

        # Materialize corrected files first.
        try:
            if old_data_path.exists() and old_data_path != new_data_path and not new_data_path.exists():
                shutil.copy2(old_data_path, new_data_path)

            if new_meta_path != meta_path:
                tmp_meta = new_meta_path.with_suffix(new_meta_path.suffix + ".tmp")
                tmp_meta.write_text(json.dumps(fixed_meta, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp_meta.replace(new_meta_path)
            else:
                meta_path.write_text(json.dumps(fixed_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            entry["status"] = "skip_write_error"
            entry["reason"] = str(exc)
            _record(entry)
            continue

        # Archive originals only after corrected artefacts exist.
        if backup_originals and run_backup_dir is not None:
            try:
                if old_data_path.exists() and old_data_path != new_data_path:
                    data_backup = run_backup_dir / old_data_path.relative_to(root)
                    data_backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_data_path), str(data_backup))
                    entry["backup_data_path"] = str(data_backup)

                if meta_path.exists() and meta_path != new_meta_path:
                    meta_backup = run_backup_dir / meta_path.relative_to(root)
                    meta_backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(meta_path), str(meta_backup))
                    entry["backup_meta_path"] = str(meta_backup)
            except Exception as exc:
                # Corrected files already exist; keep that success but report backup issue.
                entry["backup_error"] = str(exc)

        _record(entry)

    return {
        "root_dir": str(root),
        "dry_run": dry_run,
        "candidate_seeds": list(candidate_seeds),
        "lookup_key_counts": lookup_key_counts,
        "backup_dir": str(run_backup_dir) if run_backup_dir is not None else None,
        "counts": counts,
        "entries": report_entries,
    }
