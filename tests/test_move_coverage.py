"""Sweep generation in tools/move_coverage.py (no game needed)."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import move_coverage as mc  # noqa: E402


def xy(**kw):
    return {"kind": "xy", "name": "Dir", "visible": True, "radius": 25, "facing": 1, "snap_angles": 8, **kw}


def test_xy_values_stay_in_the_unit_disk_and_cover_32_angles():
    vals = mc.xy_values(xy())
    assert all(math.hypot(v["x"], v["y"]) <= 1.0001 for v in vals)
    full = {round(math.degrees(math.atan2(-v["y"], v["x"]))) % 360 for v in vals
            if not v.get("snap") and math.hypot(v["x"], v["y"]) > 0.99}
    assert len(full) >= 32
    assert any(v.get("snap") for v in vals)          # mouse input near snap angles
    assert {"x": 0.0, "y": 0.0} in vals              # minimum (the plot's min length/always-max apply)


def test_xy_limit_edges_are_probed_inside_and_just_outside():
    vals = mc.xy_values(xy(limit_angle=True, limit_center_degrees=0, limit_range_degrees=90))
    angles = {round(math.degrees(math.atan2(v["y"], v["x"]))) for v in vals if math.hypot(v["x"], v["y"]) > 0.99}
    assert {45, -45, 46, -46} <= angles


def test_ranges_every_step_plus_out_of_range():
    vals = mc.range_values({"kind": "slider", "min": 0, "max": 100, "step": 1})
    assert vals[:101] == list(range(101)) and -1 in vals and 101 in vals
    cont = mc.range_values({"kind": "slider", "min": 0, "max": 1, "step": 0})
    assert len(cont) == 13  # 11 points + 2 out of range


def test_8way_uses_only_enabled_directions_and_disabled_toggles_have_no_values():
    assert mc.widget_values({"kind": "8way", "enabled": ["E", "W"]}) == ["E", "W"]
    assert mc.widget_values({"kind": "toggle", "disabled": True}) == []


def test_small_widget_sets_are_a_full_product_large_ones_pairwise():
    small = [{"kind": "check", "visible": True}, {"kind": "8way", "enabled": ["E", "W", "N"], "visible": True}]
    combos = mc.widget_combos(small)
    assert len(combos) == 1 + 2 * 3
    big = [xy(), {"kind": "slider", "min": 0, "max": 100, "step": 1, "visible": True}]
    combos = mc.widget_combos(big)
    singles = len(mc.widget_values(big[0])) + len(mc.widget_values(big[1]))
    assert len(combos) == 1 + singles + 10 * 10


def test_seat_sweep_includes_gates_extras_di_and_predictions():
    desc = {
        "moves": [{"name": "Grab", "legal": True, "feintable": True, "reversible": True,
                   "widgets": [{"kind": "8way", "name": "Direction", "enabled": ["E", "W"], "visible": True}]},
                  {"name": "Taunt", "legal": False, "widgets": []}],
        "extra_widgets": [{"kind": "toggle", "name": "Hover", "visible": True, "disabled": False, "pressed": False}],
        "predictions": [-1, 1], "hurt": True,
    }
    sweep = mc.sweep_for_seat(desc, random.Random(0))
    actions = {spec["action"] for spec, _ in sweep}
    assert "Taunt" not in actions
    gates = [spec for spec, _ in sweep if spec.get("gate")]
    assert {spec["widgets"]["0"] for spec in gates} == set(mc.WAY_DIRS) - {"E", "W"}
    assert any(spec.get("feint") for spec, _ in sweep) and any(spec.get("reverse") for spec, _ in sweep)
    assert any("di" in spec for spec, _ in sweep) and any(spec.get("prediction") == 1 for spec, _ in sweep)
    assert any(spec.get("extra") == {"0": True} for spec, _ in sweep)
