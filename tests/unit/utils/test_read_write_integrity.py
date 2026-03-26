from pathlib import Path
import json

import pytest

from dflintdpy.data.config import HP
from dflintdpy.utils import read_write as rw
from dflintdpy.utils.read_write import Artefacts


def _set_tmp_store(monkeypatch, tmp_path: Path) -> Path:
    store = tmp_path / "store_data"
    store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rw, "_root_path", lambda: store)
    return store


def test_pred_meta_contains_payload_checksum(monkeypatch, tmp_path: Path):
    _set_tmp_store(monkeypatch, tmp_path)
    cfg = HP()

    pred_path = rw.write_pred(cfg, {"weights": [1, 2, 3]}, artifact_tag="adfl")
    meta_path = pred_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["artefact"] == "PRED"
    assert meta["artifact_tag"] == "adfl"
    assert meta["hash_type"] == "pred"
    assert "hash" in meta
    assert "head_hash" in meta
    assert "canonical_cfg_subset" in meta


def test_read_cache_allows_trailing_bytes(monkeypatch, tmp_path: Path):
    _set_tmp_store(monkeypatch, tmp_path)
    cfg = HP()

    pred_path = rw.write_pred(cfg, {"weights": [1, 2, 3]}, artifact_tag="adfl")

    # Corrupt payload while keeping a readable pickle stream.
    with pred_path.open("ab") as f:
        f.write(b"tamper")

    # Current read path uses plain pickle loading and accepts trailing bytes.
    loaded = rw.read_cache(cfg, Artefacts.PRED, artifact_tag="adfl")
    assert loaded == {"weights": [1, 2, 3]}
