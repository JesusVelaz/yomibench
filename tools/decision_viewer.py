"""Build the Decision Viewer: one self-contained HTML page from explain_game.py outputs.

    python tools/decision_viewer.py EXPLAIN.json [...] --out decision-viewer.html

The page lists the games; each game has a timeline strip of its decisions (one row per seat,
coloured by the decision's category: flat tie, confident, mixed, moderate, quick, explored;
marks below the strip for Wait, Taunt and idle Continue picks). Clicking a decision (or the
arrow keys) opens its card: both fighters' state, each acting seat's top candidates as HP-
equivalents behind its game value, the mix it played, the pick, the opponent mix it expected,
and a one-line reading. No network: the data and code are inlined. Works at 400px wide and in
dark mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KEEP_SEAT = ("value_hp", "support", "n_candidates", "n_known", "pick", "best_outside", "variant_gap", "flags",
             "category", "waste", "top", "opponent_mix", "reading", "reproduced")


def slim(e: dict[str, Any]) -> dict[str, Any]:
    """What the page reads of an explanation (drops nothing a card shows)."""
    return {
        "record": e["record"], "kind": e["kind"], "characters": e["characters"], "seed": e["seed"],
        "bots": [{k: b.get(k) for k in ("seat", "bot", "value", "hp_per_value")} for b in e["bots"]],
        "values": e["values"], "recompute": e.get("recompute"), "result": e.get("result"),
        "summary": e.get("summary"), "decisions_total": e.get("decisions_total"),
        "decisions": [{"turn": d["turn"], "tick": d["tick"], "search": d["search"], "distance": d["distance"],
                       "state": d["state"],
                       "seats": [None if x is None else {k: x.get(k) for k in KEEP_SEAT} for x in d["seats"]]}
                      for d in e["decisions"]],
    }


def build(explanations: list[dict[str, Any]]) -> str:
    data = json.dumps([slim(e) for e in explanations], separators=(",", ":")).replace("</", "<\\/")
    return PAGE.replace("/*DATA*/[]", data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("explanations", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    games = [json.loads(p.read_text()) for p in args.explanations]
    games.sort(key=lambda e: (e["kind"] != "gate", e["record"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(games), encoding="utf-8")
    print(f"{args.out}: {len(games)} games, {sum(len(e['decisions']) for e in games)} decisions")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decision Viewer</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f5f2; --surface: #fcfcfb; --surface-2: #f0efeb; --line: #dddcd6;
  --text: #0b0b0b; --text-2: #52514e; --muted: #7a7974;
  --c-confident: #2a78d6; --c-flat: #eb6834; --c-mixed: #1baf7a; --c-moderate: #a3a29b;
  --c-quick: #c9c8c1; --c-explored: #6250d6;
  --mix: #2a78d6; --gap: #b9b8b1; --pick: #0b0b0b; --hp: #1baf7a; --hp-low: #e34948; --meter: #eda100;
  --focus: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --surface-2: #242422; --line: #34342f;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e86;
    --c-confident: #3987e5; --c-flat: #d95926; --c-mixed: #199e70; --c-moderate: #6f6e68;
    --c-quick: #45443f; --c-explored: #9085e9;
    --mix: #3987e5; --gap: #5a5953; --pick: #f0efec; --hp: #199e70; --hp-low: #e66767; --meter: #c98500;
    --focus: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --surface-2: #242422; --line: #34342f;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e86;
  --c-confident: #3987e5; --c-flat: #d95926; --c-mixed: #199e70; --c-moderate: #6f6e68;
  --c-quick: #45443f; --c-explored: #9085e9;
  --mix: #3987e5; --gap: #5a5953; --pick: #f0efec; --hp: #199e70; --hp-low: #e66767; --meter: #c98500;
  --focus: #3987e5;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 16px; }
header h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
header p { margin: 0 0 12px; color: var(--text-2); max-width: 72ch; }
.layout { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 760px) { .layout { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
.games { padding: 6px; max-height: calc(100vh - 120px); overflow: auto; position: sticky; top: 12px; }
@media (max-width: 760px) { .games { position: static; max-height: 260px; } }
.games h2, .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 8px 8px 6px; font-weight: 600; }
.game { display: block; width: 100%; text-align: left; border: 0; background: transparent; color: inherit; font: inherit;
        padding: 8px; border-radius: 8px; cursor: pointer; }
.game:hover { background: var(--surface-2); }
.game[aria-current="true"] { background: var(--surface-2); box-shadow: inset 3px 0 0 var(--focus); }
.game .t { font-weight: 600; }
.game .s { color: var(--text-2); font-size: 12px; }
.main { min-width: 0; display: grid; gap: 12px; }
.gamehead { padding: 12px 14px; }
.gamehead h2 { margin: 0; font-size: 16px; }
.gamehead .meta { color: var(--text-2); font-size: 13px; margin-top: 2px; }
.counts { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; font-size: 12px; color: var(--text-2); }
.counts b { color: var(--text); font-variant-numeric: tabular-nums; }
.strip { padding: 10px 14px 12px; }
.strip svg { display: block; width: 100%; height: 72px; touch-action: manipulation; cursor: pointer; }
.strip svg:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--text-2); margin-top: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.sw { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
.controls button, .controls select { font: inherit; font-size: 13px; color: var(--text); background: var(--surface-2);
  border: 1px solid var(--line); border-radius: 7px; padding: 5px 10px; cursor: pointer; }
.controls button:focus-visible, .controls select:focus-visible, .game:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }
.controls .pos { color: var(--text-2); font-size: 13px; font-variant-numeric: tabular-nums; }
.card { padding: 12px 14px 14px; }
.card h3 { margin: 0; font-size: 15px; }
.card .sub { color: var(--text-2); font-size: 13px; }
.seats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
@media (max-width: 900px) { .seats { grid-template-columns: minmax(0, 1fr); } }
.seat { border: 1px solid var(--line); border-radius: 9px; padding: 10px 12px; background: var(--surface); min-width: 0; }
.seat h4 { margin: 0; font-size: 14px; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
.seat h4 .role { color: var(--text-2); font-weight: 400; font-size: 12px; }
.chip { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 999px;
        border: 1px solid var(--line); color: var(--text-2); white-space: nowrap; }
.chip .sw { width: 8px; height: 8px; }
.state { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 3px 10px; margin: 8px 0; font-size: 12px; color: var(--text-2); align-items: center; }
.bar { height: 6px; background: var(--surface-2); border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 3px; }
.reading { margin: 8px 0; padding: 7px 9px; background: var(--surface-2); border-radius: 7px; font-size: 13px; }
.flags { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
table.cands { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
table.cands th { text-align: left; font-weight: 600; color: var(--muted); font-size: 11px; padding: 3px 4px; border-bottom: 1px solid var(--line); }
table.cands td { padding: 4px; border-bottom: 1px solid var(--line); vertical-align: middle; }
table.cands tr.picked td { font-weight: 650; }
table.cands td.lab { overflow-wrap: anywhere; line-height: 1.3; }
table.cands td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.gapbar { height: 8px; position: relative; }
.gapbar i { position: absolute; right: 0; top: 0; height: 8px; border-radius: 4px 0 0 4px; background: var(--gap); }
.gapbar b { position: absolute; right: -1px; top: -1px; width: 2px; height: 10px; background: var(--text-2); border-radius: 1px; }
.mixbar { height: 8px; background: var(--surface-2); border-radius: 4px; overflow: hidden; }
.mixbar i { display: block; height: 100%; background: var(--mix); border-radius: 0 4px 4px 0; }
.pickmark { display: inline-block; font-size: 10px; font-weight: 700; padding: 0 5px; border-radius: 4px; background: var(--pick); color: var(--surface); margin-right: 4px; vertical-align: 1px; }
.opp { font-size: 12px; color: var(--text-2); margin-top: 8px; }
.empty { color: var(--muted); font-size: 13px; padding: 10px 12px; border: 1px dashed var(--line); border-radius: 9px; }
.note { color: var(--muted); font-size: 12px; margin-top: 10px; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--line);
       border-radius: 7px; padding: 5px 8px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); display: none; z-index: 5; max-width: 260px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Decision Viewer</h1>
    <p>Why the bot picked each move. Every decision, the bot previews its candidates one decision ahead, scores them with its value net, and solves for a mix the opponent can't exploit. Values are shown as HP-equivalents (value 0.01 ≈ 12 HP), measured against the seat's game value: the moves in the mix sit at 0, everything else behind.</p>
  </header>
  <div class="layout">
    <nav class="panel games" aria-label="Games"><h2>Games</h2><div id="games"></div></nav>
    <main class="main" id="main"></main>
  </div>
</div>
<div id="tip" role="status"></div>
<script id="data" type="application/json">/*DATA*/[]</script>
<script>
"use strict";
const GAMES = JSON.parse(document.getElementById("data").textContent);
const CATS = [
  ["flat_tie", "Flat tie", "--c-flat", "the best other move is within 6 HP: the net can't tell them apart"],
  ["confident", "Confident", "--c-confident", "everything outside the mix is 24+ HP worse"],
  ["mixed", "Mixed", "--c-mixed", "several moves balanced against the opponent"],
  ["moderate", "Moderate", "--c-moderate", "one move, 6–24 HP ahead of the next"],
  ["quick", "Quick", "--c-quick", "capped self-play search (no refinement): a weaker read"],
  ["explored", "Explored", "--c-explored", "picked for exploration, not from the mix"],
];
const CAT = Object.fromEntries(CATS.map(c => [c[0], c]));
const WASTE = {wait: "Wait", taunt: "Taunt", idle_continue: "idle Continue"};
const STEP = [["all", "every decision"], ["waste", "Wait / Taunt / idle Continue"], ["flat_tie", "flat ties"],
              ["confident", "confident picks"], ["explored", "explored picks"]];
let gi = 0, di = 0, stepBy = "all";
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmtHp = x => x == null ? "?" : (Math.abs(x) < 0.05 ? "0" : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(Math.abs(x) < 10 ? 1 : 0));
const pct = p => (p * 100 < 1 && p > 0) ? "<1%" : Math.round(p * 100) + "%";
const role = (g, s) => g.bots[s].bot;
const net = (g, s) => (g.bots[s].value || "").replace("valuenet:", "");
const cssVar = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function gameTitle(g) { return `${g.characters[0]} vs ${g.characters[1]}`; }
function gameSub(g) {
  const r = g.result || {};
  const who = r.winner == null ? "draw" : `p${r.winner + 1} won`;
  const kind = g.kind === "gate" ? `gate: ${role(g, 0)} vs ${role(g, 1)}` : "self-play";
  return `${kind} · ${who} · ${r.turns ?? "?"} turns`;
}

function renderGames() {
  $("#games").innerHTML = GAMES.map((g, i) =>
    `<button class="game" data-i="${i}" aria-current="${i === gi}"><div class="t">${esc(gameTitle(g))}</div>
     <div class="s">${esc(gameSub(g))}</div></button>`).join("");
  document.querySelectorAll(".game").forEach(b => b.onclick = () => { gi = +b.dataset.i; di = 0; render(); });
}

function matches(d, by) {
  if (by === "all") return true;
  return d.seats.some(x => x && (by === "waste" ? x.waste : by === "flat_tie" ? x.flags.flat_tie :
    by === "confident" ? x.flags.confident : x.flags.explored));
}

function counts(g) {
  const out = [];
  g.summary.forEach((s, seat) => {
    const w = Object.entries(s.waste || {}).map(([k, v]) => `${WASTE[k]} ${v.n}`).join(", ") || "none";
    out.push(`<span>p${seat + 1} (${esc(role(g, seat))}): <b>${s.decisions}</b> decisions · flat ties <b>${s.flat_tie}</b> · confident <b>${s.confident}</b> · ${esc(w)}</span>`);
  });
  return out.join("");
}

function strip(g) {
  const n = g.decisions.length, W = 1000, rowH = 22, gap = 6, top = 4, wasteY = top + 2 * rowH + gap + 6;
  const H = wasteY + 12, bw = W / Math.max(n, 1);
  let s = `<svg id="strip" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" tabindex="0" role="img"
    aria-label="Timeline of ${n} decisions; arrow keys move between them">`;
  for (let seat = 0; seat < 2; seat++) {
    const y = top + seat * (rowH + gap);
    s += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="var(--surface-2)"/>`;
    g.decisions.forEach((d, i) => {
      const x = d.seats[seat];
      if (!x) return;
      const c = CAT[x.category];
      const h = x.category === "quick" ? rowH / 2 : rowH;
      s += `<rect x="${(i * bw).toFixed(2)}" y="${y + rowH - h}" width="${Math.max(bw - (bw > 3 ? 1 : 0), 0.6).toFixed(2)}" height="${h}" fill="var(${c[2]})"/>`;
      const mw = Math.max(bw, 3), mx = ((i + .5) * bw - mw / 2).toFixed(2);
      if (x.category === "explored") s += `<rect x="${mx}" y="${y - 4}" width="${mw}" height="3" fill="var(${c[2]})"/>`;
      if (x.waste) s += `<rect x="${mx}" y="${wasteY + (seat ? 7 : 0)}" width="${mw}" height="5" fill="var(--text-2)"/>`;
    });
  }
  s += `<rect id="cursor" x="0" y="0" width="${Math.max(bw, 2)}" height="${2 * rowH + gap + top + 2}" fill="none" stroke="var(--text)" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  return s + "</svg>";
}

function seatState(st, g, seat) {
  const hpPct = 100 * st.hp / st.max_hp;
  const gun = st.character === "Cowboy" ? ` · gun ${st.has_gun == null ? "?" : st.has_gun ? "held" : "not held"}${st.bullets != null ? ` · ${st.bullets} bullets` : ""}` : "";
  return `<div class="state">
    <span>HP ${st.hp}/${st.max_hp}</span><div class="bar"><i style="width:${hpPct}%;background:var(${hpPct < 30 ? "--hp-low" : "--hp"})"></i></div>
    <span>Meter ${st.meter}</span><div class="bar"><i style="width:${Math.min(100, ((st.meter || 0) * 100 + (st.meter_partial || 0)) / 5)}%;background:var(--meter)"></i></div>
    <span>State</span><span>${esc(st.state)}${st.hitstun ? " (in hitstun)" : ""} · burst ${st.burst ?? "?"} · combo ${st.combo ?? 0}${gun}</span>
  </div>`;
}

function seatCard(g, d, seat) {
  const st = d.state[seat], x = d.seats[seat];
  let h = `<section class="seat"><h4>p${seat + 1} ${esc(st.character)} <span class="role">${esc(role(g, seat))} · ${esc(net(g, seat))}</span></h4>`;
  h += seatState(st, g, seat);
  if (!x) return h + `<div class="empty">${st.actionable ? "Only one option this decision." : "Not actionable: mid-move or being hit."}</div></section>`;
  const c = CAT[x.category], f = x.flags;
  h += `<div class="flags"><span class="chip"><span class="sw" style="background:var(${c[2]})"></span>${c[1]}</span>
    <span class="chip">${f.search} search</span>${f.in_mix ? `<span class="chip">pick in mix</span>` : `<span class="chip">pick outside mix</span>`}
    ${f.explored ? `<span class="chip">explored: ${esc(f.explore_mode)}</span>` : ""}${f.opening ? `<span class="chip">opening</span>` : ""}
    ${x.waste ? `<span class="chip">${WASTE[x.waste]}</span>` : ""}${x.reproduced === false ? `<span class="chip">re-search differed</span>` : ""}</div>`;
  h += `<div class="reading">${esc(x.reading)}</div>`;
  const bo = x.best_outside;
  h += `<div class="sub">Game value ${fmtHp(x.value_hp)} HP for p${seat + 1} · picked <b>${esc(x.pick.label)}</b> (${pct(x.pick.p)})${bo ? ` · best other move ${esc(bo.label)} ${fmtHp(-bo.gap_hp)} HP` : ""}</div>`;
  const shown = x.top, worst = Math.max(1, ...shown.map(t => -(t.hp ?? 0)));
  h += `<table class="cands"><colgroup><col><col style="width:54px"><col style="width:22%"><col style="width:46px"><col style="width:18%"></colgroup>
    <thead><tr><th>Candidate (top by value)</th><th style="text-align:right">vs game</th><th>behind</th><th style="text-align:right">mix</th><th>played</th></tr></thead><tbody>`;
  for (const t of shown) {
    const behind = t.hp == null ? 0 : Math.max(0, -t.hp);
    h += `<tr class="${t.picked ? "picked" : ""}"><td class="lab" title="${esc(t.label)}">${t.picked ? `<span class="pickmark">PICK</span>` : ""}${esc(t.label)}</td>
      <td class="num">${fmtHp(t.hp)}</td>
      <td><div class="gapbar" title="${behind.toFixed(1)} HP behind the game value"><i style="width:${(100 * behind / worst).toFixed(1)}%"></i><b></b></div></td>
      <td class="num">${t.p > 0 ? pct(t.p) : "–"}</td>
      <td><div class="mixbar"><i style="width:${(100 * t.p).toFixed(1)}%"></i></div></td></tr>`;
  }
  h += `</tbody></table>`;
  h += `<div class="note">Differences under about 1 HP are inside the solver's tolerance. ${x.n_known} of ${x.n_candidates} candidates evaluated${x.n_known < x.n_candidates ? " (double oracle previews only what it needs)" : ""}${x.variant_gap != null ? ` · nearest unmixed variant of a mixed move: ${fmtHp(-x.variant_gap * g.bots[seat].hp_per_value)} HP` : ""}</div>`;
  h += `<div class="opp">Expected opponent mix: ${x.opponent_mix.map(o => `${esc(o.label)} ${pct(o.p)}`).join(" · ") || "?"}</div>`;
  return h + `</section>`;
}

function card(g) {
  const d = g.decisions[di];
  if (!d) return `<section class="panel card"><div class="empty">No decisions with a choice in this game.</div></section>`;
  return `<section class="panel card" aria-live="polite"><h3>Turn ${d.turn} · tick ${d.tick}</h3>
    <div class="sub">${d.search === "quick" ? "Quick (capped) search" : "Full search"} · distance ${d.distance} px${g.values === "recomputed" ? " · values re-computed offline from the recorded state" : ""}</div>
    <div class="seats">${seatCard(g, d, 0)}${seatCard(g, d, 1)}</div></section>`;
}

function render() {
  renderGames();
  const g = GAMES[gi];
  if (!g) { $("#main").innerHTML = `<div class="panel card empty">No games.</div>`; return; }
  const rc = g.recompute;
  $("#main").innerHTML = `
    <section class="panel gamehead"><h2>${esc(gameTitle(g))}</h2>
      <div class="meta">${esc(gameSub(g))} · seed ${g.seed} · ${g.values === "recomputed" ? `values re-computed (${rc.picks_reproduced}/${rc.seat_decisions} picks reproduced)` : "values recorded during play"}</div>
      <div class="counts">${counts(g)}</div></section>
    <section class="panel strip"><div class="section-title" style="margin:0 0 6px">Timeline · top row p1, bottom row p2 · ticks below: Wait / Taunt / idle Continue (upper p1, lower p2)</div>
      ${strip(g)}
      <div class="legend">${CATS.map(c => `<span title="${esc(c[3])}"><span class="sw" style="background:var(${c[2]})"></span>${c[1]}</span>`).join("")}<span>half-height = quick · mark above = explored</span></div>
      <div class="controls"><button id="prev" aria-label="Previous decision">← Prev</button><button id="next" aria-label="Next decision">Next →</button>
        <label>Step through <select id="by">${STEP.map(s => `<option value="${s[0]}"${s[0] === stepBy ? " selected" : ""}>${s[1]}</option>`).join("")}</select></label>
        <span class="pos" id="pos"></span></div></section>
    <div id="card"></div>`;
  const svg = $("#strip");
  const pick = e => {
    const r = svg.getBoundingClientRect();
    return Math.max(0, Math.min(g.decisions.length - 1, Math.floor((e.clientX - r.left) / r.width * g.decisions.length)));
  };
  svg.addEventListener("click", e => { di = pick(e); update(); svg.focus(); });
  svg.addEventListener("mousemove", e => {
    const d = g.decisions[pick(e)], tip = $("#tip");
    tip.innerHTML = `Turn ${d.turn}: ` + d.seats.map((x, s) => x ? `p${s + 1} ${esc(x.pick.label)} (${CAT[x.category][1]})` : null).filter(Boolean).join("<br>");
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 12, innerWidth - 270) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  svg.addEventListener("mouseleave", () => $("#tip").style.display = "none");
  svg.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
    if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
  });
  $("#prev").onclick = () => step(-1);
  $("#next").onclick = () => step(1);
  $("#by").onchange = e => { stepBy = e.target.value; const k = next(0, 1, true); if (k != null) { di = k; } update(); };
  update();
}

function next(from, dir, inclusive) {
  const ds = GAMES[gi].decisions;
  for (let k = inclusive ? from : from + dir; k >= 0 && k < ds.length; k += dir) if (matches(ds[k], stepBy)) return k;
  return null;
}
function step(dir) { const k = next(di, dir, false); if (k != null) { di = k; update(); } }

function update() {
  const g = GAMES[gi], n = g.decisions.length;
  $("#card").innerHTML = card(g);
  const cur = $("#cursor");
  if (cur) cur.setAttribute("x", (di * 1000 / Math.max(n, 1)).toFixed(2));
  const hits = g.decisions.filter(d => matches(d, stepBy)).length;
  $("#pos").textContent = `decision ${n ? di + 1 : 0} of ${n}` + (stepBy === "all" ? "" : ` · ${hits} match`);
}

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
