"""The YomiBenchVsBot mod zip (tools/build_vsbot_mod.py): what the in-game "Multiplayer (vs bot)" needs from it."""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_vsbot_mod  # noqa: E402

MOD = ROOT / "mod" / "YomiBenchVsBot"


def _zip(tmp_path) -> zipfile.ZipFile:
    return zipfile.ZipFile(build_vsbot_mod.build(tmp_path))


def test_the_zip_carries_every_script_the_client_copy_runs(tmp_path):
    # VsBot.gd copies CLIENT_SCRIPTS out of res://YomiBenchVsBot/ into the headless copy's folder.
    listed = re.search(r"const CLIENT_SCRIPTS = \[([^\]]*)\]", (MOD / "VsBot.gd").read_text()).group(1)
    scripts = re.findall(r'"([^"]+)"', listed)
    assert {"MpClient.gd", "MpLink.gd", "YomiHuman.gd", "SteamStub.gd"} <= set(scripts)
    names = set(_zip(tmp_path).namelist())
    for s in scripts:
        assert f"YomiBenchVsBot/{s}" in names


def test_only_the_shared_link_script_declares_rpcs():
    # Godot 3.5 addresses an RPC by its index in the receiving script's remote methods: both games must
    # declare them in one shared script (MpLink.gd), never in VsBot.gd or MpClient.gd.
    for f in [*MOD.glob("*.gd"), ROOT / "mod" / "YomiBench" / "YomiHuman.gd"]:
        has = bool(re.search(r"^\s*(remote|remotesync|master|puppet)\s+func\b", f.read_text(), re.M))
        assert has == (f.name == "MpLink.gd"), f.name


def test_the_zip_is_deterministic(tmp_path):
    a = build_vsbot_mod.build(tmp_path / "a").read_bytes()
    b = build_vsbot_mod.build(tmp_path / "b").read_bytes()
    assert a == b
