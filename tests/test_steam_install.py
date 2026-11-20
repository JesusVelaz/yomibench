"""The public build runs only the player's own Steam install: a library's steamapps/common/<installdir> next to the
game's app manifest; build_run_dir (where every headless copy is made) refuses anything else when it's required."""
import pytest

from yomibench import game


def _library(tmp_path, installdir="YourOnlyMoveIsHUSTLE", manifest=True):
    d = tmp_path / "steamapps" / "common" / installdir
    d.mkdir(parents=True)
    (d / game.EXE).write_bytes(b"")
    if manifest:
        (tmp_path / "steamapps" / f"appmanifest_{game.STEAM_APP_ID}.acf").write_text(
            f'"AppState"\n{{\n\t"appid"\t\t"{game.STEAM_APP_ID}"\n\t"installdir"\t\t"{installdir}"\n}}\n')
    return d


def test_a_steam_install_is_recognized_and_a_copy_is_not(tmp_path):
    assert game.steam_install_ok(_library(tmp_path / "a"))
    assert not game.steam_install_ok(_library(tmp_path / "b", manifest=False))
    copy = tmp_path / "somewhere" / "YourOnlyMoveIsHUSTLE"
    copy.mkdir(parents=True)
    assert not game.steam_install_ok(copy)


def test_when_required_only_a_steam_install_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(game, "REQUIRE_STEAM_INSTALL", True)
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / game.EXE).write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="only your own Steam install"):
        game.find_game_dir(copy)
    with pytest.raises(FileNotFoundError, match="only your own Steam install"):
        game.build_run_dir(copy, tmp_path / "run")


def test_an_unsupported_game_build_warns_once(tmp_path, monkeypatch, capsys):
    from yomibench import game

    lib = tmp_path / "steamapps"
    d = lib / "common" / "YourOnlyMoveIsHUSTLE"
    d.mkdir(parents=True)
    manifest = lib / f"appmanifest_{game.STEAM_APP_ID}.acf"
    monkeypatch.delenv("YOMIBENCH_NO_BUILD_WARNING", raising=False)
    monkeypatch.setattr(game, "_warned_build", False)
    manifest.write_text(f'"AppState"\n{{\n\t"buildid"\t\t"{game.SUPPORTED_GAME["steam_build_id"]}"\n}}\n')
    assert game.warn_if_unsupported(d) is None
    manifest.write_text('"AppState"\n{\n\t"buildid"\t\t"1"\n}\n')
    assert "build 1" in game.warn_if_unsupported(d) and "supports 1.10.0" in capsys.readouterr().err
    game.warn_if_unsupported(d)
    assert capsys.readouterr().err == ""  # once per process
    monkeypatch.setenv("YOMIBENCH_NO_BUILD_WARNING", "1")
    assert game.warn_if_unsupported(d) is None
