from pathlib import Path

import dflintdpy._runtime as runtime_module


##############################################################
### test configure_terminal_cache_environment(...) ###
##############################################################


def test_runtime_configure_terminal_cache_environment_sets_defaults(
    monkeypatch,
    tmp_path,
):
    """Verify that terminal runs get writable cache defaults."""
    # Arrange an environment without caller-provided cache settings.
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(
        runtime_module.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )

    # Act by applying the terminal cache defaults.
    runtime_module.configure_terminal_cache_environment()

    # Assert that both cache paths are set inside the temp root.
    xdg_cache_home = Path(runtime_module.os.environ["XDG_CACHE_HOME"])
    mpl_config_dir = Path(runtime_module.os.environ["MPLCONFIGDIR"])
    assert xdg_cache_home == tmp_path / "dflintdpy" / "xdg-cache", \
        "The XDG cache default should live under the temp directory."
    assert mpl_config_dir == xdg_cache_home / "matplotlib", \
        "The Matplotlib cache default should live under XDG_CACHE_HOME."
    assert xdg_cache_home.is_dir(), \
        "The XDG cache default should be created before use."
    assert mpl_config_dir.is_dir(), \
        "The Matplotlib cache default should be created before use."
    pass


def test_runtime_configure_terminal_cache_environment_preserves_user_values(
    monkeypatch,
    tmp_path,
):
    """Verify that explicit cache environment values are not overwritten."""
    # Arrange caller-provided cache settings.
    xdg_cache_home = tmp_path / "custom-xdg"
    mpl_config_dir = tmp_path / "custom-mpl"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache_home))
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config_dir))
    monkeypatch.setattr(
        runtime_module.tempfile,
        "gettempdir",
        lambda: str(tmp_path / "unused"),
    )

    # Act by applying the terminal cache defaults.
    runtime_module.configure_terminal_cache_environment()

    # Assert that user-provided values remain authoritative.
    assert runtime_module.os.environ["XDG_CACHE_HOME"] == str(xdg_cache_home), \
        "configure_terminal_cache_environment should preserve XDG_CACHE_HOME."
    assert runtime_module.os.environ["MPLCONFIGDIR"] == str(mpl_config_dir), \
        "configure_terminal_cache_environment should preserve MPLCONFIGDIR."
    pass
