"""Candidate rules hashes: set 4's and each set-5 switch combination's must equal the simulator plugin's
(`hello.candidate_rules`), or records and servers stop matching."""

import importlib.util
from pathlib import Path

MOD = Path(__file__).parents[1] / "mod" / "YomiBench"
spec = importlib.util.spec_from_file_location("rules_hash", Path(__file__).parents[1] / "tools/rules_hash.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

# the simulator plugin at its set-5 release, Handler(candidate_set="5", set5=...).hello()["candidate_rules"] by switch bits
# (bit 0 draw_pairs, 1 gun_aim_fine, 2 bullet_di, 3 draw_base).
SIM_SET5 = [
    "e903f07ec230", "d4893fe19a6c", "25d62064468e", "de5e91b86688",
    "c0dae1b29e4f", "cf444221f100", "b023bdedbac7", "443897562e49",
    "244a6e6fa3a5", "b1449be249a8", "3e7ed5f9720e", "724a47e71cea",
    "5cfaca0cb2d3", "33722b8fe83c", "6527652027cf", "7a624558f81b",
]


def test_set4_hash_is_frozen():
    assert rh.rules_hash(MOD) == "7b99f2785620"


def test_each_set5_combination_has_the_pinned_hash():
    assert [rh.rules_hash5(MOD, rh.set5_switches(b)) for b in range(16)] == SIM_SET5


def test_switches_not_given_stay_on():
    assert rh.rules_hash5(MOD, {}) == SIM_SET5[15]
    assert rh.rules_hash5(MOD, {"bullet_di": False}) == SIM_SET5[15 & ~4]


def test_set4_generator_is_frozen_apart_from_its_header():
    # The set-4 hash names Candidates4.gd; set 5 changes only Candidates.gd.
    four = (MOD / "Candidates4.gd").read_text(encoding="utf-8")
    assert four.startswith("extends Control\n# FROZEN: candidate set 4")
    assert "set5" not in rh.strip(four, True)
