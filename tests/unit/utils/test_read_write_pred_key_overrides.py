from pathlib import Path

import pytest

from dflintdpy.data.config import HP
from dflintdpy.utils import read_write as rw
from dflintdpy.utils.read_write import Artefacts, _artifact_file_paths


def _pred_data_path(cfg: HP, tag: str) -> str:
    data_path, _ = _artifact_file_paths(cfg, Artefacts.PRED, artifact_tag=tag)
    return data_path.name


def test_pfl_and_dfl_ignore_interdiction_keys_in_hash():
    cfg_a = HP()
    cfg_b = HP()
    cfg_b.set("budget", cfg_a.get("budget") + 3)
    cfg_b.set("num_scenarios", cfg_a.get("num_scenarios") + 2)
    cfg_b.set("intd_seed", cfg_a.get("intd_seed") + 11)

    assert _pred_data_path(cfg_a, "pfl") == _pred_data_path(cfg_b, "pfl")
    assert _pred_data_path(cfg_a, "dfl") == _pred_data_path(cfg_b, "dfl")


def test_adfl_still_depends_on_interdiction_keys_in_hash():
    cfg_a = HP()
    cfg_b = HP()
    cfg_b.set("budget", cfg_a.get("budget") + 1)
    cfg_b.set("intd_seed", cfg_a.get("intd_seed") + 5)

    assert _pred_data_path(cfg_a, "adfl") != _pred_data_path(cfg_b, "adfl")


def test_read_cache_supports_legacy_pfl_hash(monkeypatch, tmp_path: Path):
    store = tmp_path / "store_data"
    store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rw, "_root_path", lambda: store)

    cfg = HP()
    data_path, _ = _artifact_file_paths(cfg, Artefacts.PRED, artifact_tag="pfl")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    rw._pickle_dump(data_path, {"legacy": True})

    loaded = rw.read_cache(cfg, Artefacts.PRED, artifact_tag="pfl")
    assert loaded == {"legacy": True}
