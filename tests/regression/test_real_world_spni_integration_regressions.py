import pytest

from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.config import build_run_config
from dflintdpy.utils.read_write import _unique_hash


pytestmark = pytest.mark.regression


############################
### Helper functionality ###
############################


def _run_cfg(graph_path: str):
    """Return a default SPNI run config pointing at one real graph."""
    return build_run_config(HP(), load_real_world_graph=graph_path)


#######################################
### test real-world cache identity ###
#######################################


def test_spni_real_world_cache_hashes_include_graph_identity():
    """Verify cache hashes differ for graph path and terminal changes."""
    # Arrange otherwise identical configs with different graph identities.
    my_graph_cfg = _run_cfg("real_world_spni_data/my_graph.csv")
    county_graph_cfg = _run_cfg("real_world_spni_data/county_level_arcs.csv")
    county_terminal_cfg = build_run_config(
        HP(),
        load_real_world_graph="real_world_spni_data/county_level_arcs.csv",
        source_node=1,
        target_node=14,
    )

    # Act by computing every cache hash used by the SPNI flow.
    hash_types = ["data", "intd", "pred", "result"]
    my_graph_hashes = {
        hash_type: _unique_hash(my_graph_cfg, type=hash_type)
        for hash_type in hash_types
    }
    county_graph_hashes = {
        hash_type: _unique_hash(county_graph_cfg, type=hash_type)
        for hash_type in hash_types
    }
    county_terminal_hashes = {
        hash_type: _unique_hash(county_terminal_cfg, type=hash_type)
        for hash_type in hash_types
    }

    # Assert graph identity participates in every topology-sensitive cache.
    for hash_type in hash_types:
        assert my_graph_hashes[hash_type] != county_graph_hashes[hash_type], \
            f"{hash_type} cache hash should include real graph identity."
        assert county_graph_hashes[hash_type] != \
            county_terminal_hashes[hash_type], \
            f"{hash_type} cache hash should include graph terminals."
    pass


#######################################
### test real-world terminal config ###
#######################################


def test_spni_real_world_run_config_accepts_source_and_target_nodes():
    """Verify real-world runs can configure non-default terminals."""
    # Arrange a one-indexed real graph that cannot use the default node 0.
    base_cfg = HP()

    # Act by trying to normalize explicit real-world graph terminals.
    run_cfg = build_run_config(
        base_cfg,
        load_real_world_graph="real_world_spni_data/county_level_arcs.csv",
        source_node=1,
        target_node=14,
    )

    # Assert that terminal configuration is preserved for graph construction.
    assert run_cfg.source_node == 1, \
        "Real-world run configs should preserve an explicit source node."
    assert run_cfg.target_node == 14, \
        "Real-world run configs should preserve an explicit target node."
    pass
