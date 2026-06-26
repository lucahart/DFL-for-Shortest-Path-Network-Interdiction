from pathlib import Path
import json
import pickle

import pytest

np = pytest.importorskip("numpy")

from dflintdpy.utils.read_write import (
    _unique_hash,
    repair_seed_sweep_hashes,
)


def _seed_triplet(seed: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(seed)
    vals = rng.randint(0, 150, 3).tolist()
    return int(vals[0]), int(vals[1]), int(vals[2])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_repair_seed_sweep_hashes_repairs_dataset_and_predictor(tmp_path: Path):
    root = tmp_path / "store_data"
    datasets_dir = root / "datasets"
    predictors_dir = root / "predictors"
    datasets_dir.mkdir(parents=True)
    predictors_dir.mkdir(parents=True)

    seed_0 = 105
    num_seeds = 4
    true_seed = seed_0 + 1
    wrong_seed = seed_0
    random_seed, intd_seed, loader_seed = _seed_triplet(true_seed)

    data_subset_wrong = {
        "num_features": 5,
        "num_train_samples": 1000,
        "num_val_samples": 250,
        "num_test_samples": 1000,
        "grid_size": [5, 5],
        "deg": 8,
        "noise_width": 0.5,
        "seed": wrong_seed,
        "random_seed": random_seed,
        "loader_seed": loader_seed,
    }
    data_hash_wrong = _unique_hash(data_subset_wrong, type="data")
    data_meta_old = datasets_dir / f"data_{data_hash_wrong}.meta.json"
    data_payload_old = datasets_dir / f"data_{data_hash_wrong}.npz"
    np.savez_compressed(data_payload_old, feats=np.zeros((2, 3)), costs=np.zeros((2, 4)))
    _write_json(
        data_meta_old,
        {
            "artefact": "DATA",
            "artifact_tag": None,
            "hash_type": "data",
            "hash": data_hash_wrong,
            "head_hash": "old-head-hash-data",
            "canonical_cfg_subset": data_subset_wrong,
        },
    )

    pred_subset_wrong = {
        **data_subset_wrong,
        "budget": 10,
        "num_scenarios": 3,
        "benders_max_count": 100,
        "benders_eps": 1e-3,
        "lsd": 1e-5,
        "intd_seed": intd_seed,
        "batch_size": 32,
        "pfl_epochs": 400,
        "dfl_epochs": 200,
        "pfl_lr": 2e-4,
        "dfl_lr": 3.5e-4,
    }
    pred_hash_wrong = _unique_hash(pred_subset_wrong, type="pred")
    pred_meta_old = predictors_dir / f"pred_mrdfl_{pred_hash_wrong}.meta.json"
    pred_payload_old = predictors_dir / f"pred_mrdfl_{pred_hash_wrong}.pkl"
    with pred_payload_old.open("wb") as f:
        pickle.dump({"model": "payload"}, f, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(
        pred_meta_old,
        {
            "artefact": "PRED",
            "artifact_tag": "mrdfl",
            "hash_type": "pred",
            "hash": pred_hash_wrong,
            "head_hash": "old-head-hash-pred",
            "canonical_cfg_subset": pred_subset_wrong,
        },
    )

    report = repair_seed_sweep_hashes(
        seed_0=seed_0,
        num_seeds=num_seeds,
        root_dir=root,
        dry_run=False,
        backup_originals=True,
    )

    assert report["counts"]["repaired"] == 2
    assert report["backup_dir"] is not None

    data_subset_fixed = dict(data_subset_wrong)
    data_subset_fixed["seed"] = true_seed
    data_hash_fixed = _unique_hash(data_subset_fixed, type="data")

    pred_subset_fixed = dict(pred_subset_wrong)
    pred_subset_fixed["seed"] = true_seed
    pred_hash_fixed = _unique_hash(pred_subset_fixed, type="pred")

    data_meta_new = datasets_dir / f"data_{data_hash_fixed}.meta.json"
    data_payload_new = datasets_dir / f"data_{data_hash_fixed}.npz"
    pred_meta_new = predictors_dir / f"pred_mrdfl_{pred_hash_fixed}.meta.json"
    pred_payload_new = predictors_dir / f"pred_mrdfl_{pred_hash_fixed}.pkl"

    assert data_meta_new.exists()
    assert data_payload_new.exists()
    assert pred_meta_new.exists()
    assert pred_payload_new.exists()

    data_meta_new_obj = json.loads(data_meta_new.read_text(encoding="utf-8"))
    pred_meta_new_obj = json.loads(pred_meta_new.read_text(encoding="utf-8"))
    assert data_meta_new_obj["canonical_cfg_subset"]["seed"] == true_seed
    assert pred_meta_new_obj["canonical_cfg_subset"]["seed"] == true_seed
    assert data_meta_new_obj["hash"] == data_hash_fixed
    assert pred_meta_new_obj["hash"] == pred_hash_fixed
    assert len(data_meta_new_obj.get("seed_repair_history", [])) == 1
    assert len(pred_meta_new_obj.get("seed_repair_history", [])) == 1

    backup_dir = Path(report["backup_dir"])
    assert (backup_dir / "datasets" / data_meta_old.name).exists()
    assert (backup_dir / "datasets" / data_payload_old.name).exists()
    assert (backup_dir / "predictors" / pred_meta_old.name).exists()
    assert (backup_dir / "predictors" / pred_payload_old.name).exists()

    assert not data_meta_old.exists()
    assert not data_payload_old.exists()
    assert not pred_meta_old.exists()
    assert not pred_payload_old.exists()


def test_repair_seed_sweep_hashes_dry_run_only_reports(tmp_path: Path):
    root = tmp_path / "store_data"
    predictors_dir = root / "predictors"
    predictors_dir.mkdir(parents=True)

    seed_0 = 50
    num_seeds = 3
    true_seed = seed_0 + 2
    wrong_seed = seed_0
    random_seed, intd_seed, loader_seed = _seed_triplet(true_seed)

    pred_subset_wrong = {
        "num_features": 5,
        "num_train_samples": 1000,
        "num_val_samples": 250,
        "num_test_samples": 1000,
        "grid_size": [5, 5],
        "deg": 8,
        "noise_width": 0.5,
        "seed": wrong_seed,
        "random_seed": random_seed,
        "loader_seed": loader_seed,
        "budget": 10,
        "num_scenarios": 3,
        "benders_max_count": 100,
        "benders_eps": 1e-3,
        "lsd": 1e-5,
        "intd_seed": intd_seed,
        "batch_size": 32,
        "pfl_epochs": 400,
        "dfl_epochs": 200,
        "pfl_lr": 2e-4,
        "dfl_lr": 3.5e-4,
    }
    pred_hash_wrong = _unique_hash(pred_subset_wrong, type="pred")
    pred_meta_old = predictors_dir / f"pred_adfl_{pred_hash_wrong}.meta.json"
    _write_json(
        pred_meta_old,
        {
            "artefact": "PRED",
            "artifact_tag": "adfl",
            "hash_type": "pred",
            "hash": pred_hash_wrong,
            "head_hash": "old-head",
            "canonical_cfg_subset": pred_subset_wrong,
        },
    )

    report = repair_seed_sweep_hashes(
        seed_0=seed_0,
        num_seeds=num_seeds,
        root_dir=root,
        dry_run=True,
        backup_originals=True,
    )

    assert report["counts"]["would_repair"] == 1
    assert pred_meta_old.exists()
    assert len(list(predictors_dir.glob("*.meta.json"))) == 1
