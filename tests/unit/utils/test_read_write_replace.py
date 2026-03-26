from pathlib import Path
import json

import pytest

np = pytest.importorskip("numpy")

from dflintdpy.data.config import HP
from dflintdpy.utils import read_write as rw


def _set_tmp_store(monkeypatch, tmp_path: Path) -> Path:
    store = tmp_path / "store_data"
    store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rw, "_root_path", lambda: store)
    return store


def test_write_data_replace_archives_old_version(monkeypatch, tmp_path: Path):
    _set_tmp_store(monkeypatch, tmp_path)
    cfg = HP()

    feats_a = np.zeros((2, 3), dtype=float)
    costs_a = np.ones((2, 4), dtype=float)
    feats_b = np.full((2, 3), 7.0, dtype=float)
    costs_b = np.full((2, 4), 9.0, dtype=float)

    data_path = rw.write_data(cfg, feats_a, costs_a)
    meta_path = data_path.with_suffix(".meta.json")
    old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old_hash = old_meta["hash"]

    rw.write_data(cfg, feats_b, costs_b, replace=True)

    with np.load(data_path, allow_pickle=False) as z:
        np.testing.assert_allclose(z["feats"], feats_b)
        np.testing.assert_allclose(z["costs"], costs_b)

    all_meta = sorted((data_path.parent).glob("data_*.meta.json"))
    assert len(all_meta) == 2

    archive_meta_path = next(p for p in all_meta if p.name != meta_path.name)
    archive_meta = json.loads(archive_meta_path.read_text(encoding="utf-8"))
    assert archive_meta["archived_from_hash"] == old_hash
    assert archive_meta["hash"] != old_hash
    assert "update_tag" in archive_meta

    archive_data_path = archive_meta_path.with_name(
        archive_meta_path.name.removesuffix(".meta.json") + ".npz"
    )
    assert archive_data_path.exists()


def test_write_pred_replace_archives_old_version(monkeypatch, tmp_path: Path):
    _set_tmp_store(monkeypatch, tmp_path)
    cfg = HP()

    state_a = {"weights": [1, 2, 3]}
    state_b = {"weights": [9, 8, 7]}

    pred_path = rw.write_pred(cfg, state_a, artifact_tag="adfl")
    meta_path = pred_path.with_suffix(".meta.json")
    old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old_hash = old_meta["hash"]

    rw.write_pred(cfg, state_b, artifact_tag="adfl", replace=True)

    loaded = rw._pickle_load(pred_path)
    assert loaded == state_b

    all_meta = sorted((pred_path.parent).glob("pred_adfl_*.meta.json"))
    assert len(all_meta) == 2

    archive_meta_path = next(p for p in all_meta if p.name != meta_path.name)
    archive_meta = json.loads(archive_meta_path.read_text(encoding="utf-8"))
    assert archive_meta["archived_from_hash"] == old_hash
    assert archive_meta["hash"] != old_hash
    assert "update_tag" in archive_meta

    archive_data_path = archive_meta_path.with_name(
        archive_meta_path.name.removesuffix(".meta.json") + ".pkl"
    )
    assert archive_data_path.exists()
