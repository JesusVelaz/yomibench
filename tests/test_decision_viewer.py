"""tools/decision_viewer.py: one self-contained page from explain_game outputs."""

import json
import re
import sys
from pathlib import Path

from yomibench.bots import MatrixBot
from yomibench.engine import connect
from yomibench.selfplay import play_recorded, write_record

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decision_viewer as dv  # noqa: E402
import explain_game as eg  # noqa: E402


def test_the_page_inlines_every_game_and_loads_nothing_from_the_network(tmp_path):
    engine = connect("toy")
    paths = []
    for seed in (1, 2):
        rec = write_record(tmp_path, seed, play_recorded(engine, MatrixBot(), ("Toy", "Toy"), seed, engine.hello(),
                                                         max_turns=12))
        out = tmp_path / f"{seed}.explain.json"
        e = eg.explain(rec, net_dirs=[])
        e["decisions"][0]["seats"][0]["reading"] = "</script><b>x</b>"  # data can't close the script tag
        out.write_text(json.dumps(e))
        paths.append(out)
    page = tmp_path / "viewer.html"
    assert dv.main([str(p) for p in paths] + ["--out", str(page)]) == 0
    html = page.read_text(encoding="utf-8")
    assert "<title>Decision Viewer</title>" in html and "prefers-color-scheme: dark" in html
    assert not re.search(r"""(src|href)=["']https?:""", html)  # self-contained
    blob = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    games = json.loads(blob)
    assert [g["seed"] for g in games] == [1, 2] and all(g["decisions"] for g in games)
    assert games[0]["decisions"][0]["seats"][0]["reading"] == "</script><b>x</b>"
