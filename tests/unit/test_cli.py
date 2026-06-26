import dflintdpy.cli.__main__ as cli_main
import dflintdpy.cli.spni as spni_cli
import dflintdpy.simulation.spni.pipeline as pipeline_module


#########################
### test dflintd CLI ###
#########################


def test_cli_main_dispatches_spni_subcommand(monkeypatch):
    """Verify that the package CLI forwards SPNI arguments."""
    # Arrange a SPNI command stub and a cache-configuration sentinel.
    recorded: dict[str, object] = {}

    def _fake_configure_cache_environment():
        recorded["configured"] = True

    def _fake_run_spni(argv):
        recorded["argv"] = argv
        return 0

    monkeypatch.setattr(
        cli_main,
        "configure_terminal_cache_environment",
        _fake_configure_cache_environment,
    )
    monkeypatch.setattr(cli_main, "_run_spni", _fake_run_spni)

    # Act by dispatching through the top-level command.
    exit_code = cli_main.main(["spni", "--mode", "seed_sweep"])

    # Assert that the subcommand receives the remaining argv unchanged.
    assert exit_code == 0, \
        "The top-level CLI should return the SPNI command exit code."
    assert recorded["configured"] is True, \
        "The top-level CLI should configure terminal cache defaults."
    assert recorded["argv"] == ["--mode", "seed_sweep"], \
        "The top-level CLI should forward SPNI arguments unchanged."
    pass


def test_cli_spni_main_returns_zero_after_pipeline_dispatch(monkeypatch):
    """Verify that the direct SPNI console script exits cleanly."""
    # Arrange a pipeline CLI stub and a cache-configuration sentinel.
    recorded: dict[str, object] = {}

    def _fake_configure_cache_environment():
        recorded["configured"] = True

    def _fake_pipeline_cli(argv, *, prog=None):
        recorded["argv"] = argv
        recorded["prog"] = prog
        return object()

    monkeypatch.setattr(
        spni_cli,
        "configure_terminal_cache_environment",
        _fake_configure_cache_environment,
    )
    monkeypatch.setattr(pipeline_module, "cli", _fake_pipeline_cli)

    # Act by running the direct SPNI console-script wrapper.
    exit_code = spni_cli.main(["--mode", "single"])

    # Assert that the wrapper uses a process-friendly return value.
    assert exit_code == 0, \
        "The direct SPNI console script should return a zero exit code."
    assert recorded["configured"] is True, \
        "The direct SPNI CLI should configure terminal cache defaults."
    assert recorded["argv"] == ["--mode", "single"], \
        "The direct SPNI CLI should forward arguments unchanged."
    assert recorded["prog"] is None, \
        "The direct SPNI CLI should leave argparse prog at its default."
    pass
