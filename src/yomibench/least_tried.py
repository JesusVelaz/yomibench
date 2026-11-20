"""Least-tried exploration for the exploration arm, at every level of a move: its type, then each
of its options. Generic: no move, character, widget or category is named anywhere.

In an exploration-arm game, at each full-search decision outside the opening window with no other
exploration, with probability `p` (and at most `k` times per game per seat) one seat explores:
1. it takes the legal **move type** (a candidate's action, as `explore_uniform_moves` groups them)
   that its character has been seen picking least, fleet-wide, over a rolling window of games;
2. within that type, for each **option** that has more than one value among the type's candidates
   (`candidate_options`: every key of the candidate's `data` widgets, a bare XY `data` as one
   option, and every key of its `extra`: toggles, checkboxes, DI, ... whatever the engine exposes),
   it takes the value least picked for that character, type and option;
3. it plays a candidate of the type matching those values; when none matches them all, one of those
   matching the most (uniform among them).
Ties break at random (seeded per game). The record marks the pick like any other uniform
exploration (`explore: [bool, bool]`, `explore_mode: "least-tried"`), so training's value-only rule
(`--explore-targets uniform-only`) puts the positions before it on search-value targets and keeps
outcome targets after it, plus `explore_pick: {"kind": "least-tried", "seat", "move_type",
"options": {option: value}, "counts_at_pick": {"move_type": n, "options": {option: n}},
"matched": m, "of": n}` (m of the n chosen option values matched by the candidate played).

"Tried" comes from a **counts file** that the round-3 worker takes once per chunk from
the fleet's shared file (`refresh_counts`; one writer, tools/round3_loop.py least-counts, PUTs it to the
hub as COUNTS_HUB_NAME and every worker downloads it): every round-3 chunk tar on the hub, production
and exploration-arm alike, is counted once, and each seat's pick on a decision where that seat had more than one candidate is
counted per character x move type (`counts`) and per character x move type x option x value
(`options`, every option of the candidate played); the newest tars that hold `window_games` games
are summed. The per-tar counts are cached, so a refresh downloads only the tars it hasn't seen.

Games (production's slice): `games` G (0 < G <= 1, default 1: every game, as in the exploration
arm) limits least-tried picks to a share G of games: a game takes part only when a seeded draw from
its seed alone (`game_selected`; never the process, chunk or machine) is below G, and inside such a
game p and k apply as above. The other games are played exactly as without least-tried exploration
(identical decision and result lines); only the header's `explore_least_tried` differs, with
"games": G and "selected": false. G = 1 is never written (settings, tags, headers), so the arm's
records are unchanged.

Taper: when every legal move type and option value of a character has caught up
(tools/move_type_coverage.py: enough positions after it in the training window and a held-out value
error near that after common moves), the coverage report lists the character in `taper` with a
floor F (about 0.015). The worker copies that into the counts file, and a tapered character's seat
is then eligible in only a seeded share F of all games (`seat_on`), never zero: in a production
slice of G games, in a share F / G of the selected ones (all of them when F >= G).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import time
import gzip
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

DEFAULTS = {"p": 0.03, "k": 2, "window_games": 20000}
GAMES_DEFAULT = 1.0  # `games`: the share of games with least-tried picks (1: all, the exploration arm)
# The taper rule's defaults (tools/move_type_coverage.py): a move type has caught up with at least
# `min_positions` positions after it and a held-out error at most `error_ratio` x common moves' (on
# at least `min_error_positions` valued positions); a character whose legal types all caught up keeps
# least-tried exploration in a share `floor` of its games.
TAPER_DEFAULTS = {"min_positions": 2000, "error_ratio": 1.2, "floor": 0.015, "min_error_positions": 50}
# Round-3 self-play chunk tars: production machines and exploration-arm machines (NAME.explore).
COUNT_TAR = re.compile(r"^r3-([A-Za-z0-9_-]+(?:\.explore)?)-g(\d+)-c(\d+)(?:p(\d+))?\.tar$")
COVERAGE_HUB_NAME = "r3-move-coverage.json"  # the coverage report's JSON on the hub (its `taper`)
COUNTS_FORMAT = 2  # 2: per-option counts (`options`)
MODE = "least-tried"
STATE_BOOST_MAX = 10.0  # the state-rarity boost's cap on p's multiplier (user decision)
STATE_DECAY_FLOOR = 0.1  # a class with enough positions but not yet a normal error keeps this share of its boost
EXPAND_MAX = 8  # with "expand": at most this many of the chosen type's candidates get their factors expanded


def parse_settings(spec: str | dict | None) -> dict | None:
    """--explore-least-tried / manifest flag: a JSON object with p (0 < p <= 1), k (an integer >= 1),
    window_games (an integer >= 1), each defaulting to DEFAULTS, and games (0 < games <= 1, default
    1: every game). `games` is kept only below 1, so {"games": 1} normalizes to the arm's settings.
    None, "", "{}" or {} -> None (off)."""
    if spec is None or spec == "" or spec == {}:
        return None
    d = json.loads(spec) if isinstance(spec, str) else spec
    if not isinstance(d, dict):
        raise ValueError(f"least-tried settings must be a JSON object, got {spec!r}")
    if not d:
        return None
    unknown = set(d) - set(DEFAULTS) - {"games", "expand", "state_boost"}
    if unknown:
        raise ValueError(f"least-tried settings: unknown keys {sorted(unknown)} "
                         f"(known: {sorted([*DEFAULTS, 'games', 'expand', 'state_boost'])})")
    sb = d.get("state_boost")
    if sb is not None and (isinstance(sb, bool) or not isinstance(sb, (int, float)) or not 1 <= sb <= STATE_BOOST_MAX):
        raise ValueError(f"least-tried state_boost must be a number from 1 to {STATE_BOOST_MAX}, got {sb!r}")
    x = d.get("expand", False)
    if not isinstance(x, bool):
        raise ValueError(f"least-tried expand must be true or false, got {x!r}")
    out = {**DEFAULTS, **d}
    p, k, w = out["p"], out["k"], out["window_games"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 < p <= 1:
        raise ValueError(f"least-tried p must be in (0, 1], got {p!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"least-tried k must be an integer >= 1, got {k!r}")
    if isinstance(w, bool) or not isinstance(w, int) or w < 1:
        raise ValueError(f"least-tried window_games must be an integer >= 1, got {w!r}")
    g = d.get("games", GAMES_DEFAULT)
    if isinstance(g, bool) or not isinstance(g, (int, float)) or not 0 < g <= 1:
        raise ValueError(f"least-tried games must be in (0, 1], got {g!r}")
    return {"p": float(p), "k": k, "window_games": w, **({"games": float(g)} if g < 1 else {}),
            **({"expand": True} if x else {}), **({"state_boost": float(sb)} if sb and sb > 1 else {})}


def settings_tag(settings: dict) -> str:
    """A ledger tag value: p:0.03,k:2,window_games:20000 (then ,games:G for a share of games)."""
    return ",".join(f"{k}:{settings[k]}" for k in (*DEFAULTS, "games", "expand", "state_boost") if k in settings)


# --- counting picks -------------------------------------------------------------------------------

_ACTION = re.compile(r'\{"action":"([^"\\]*)"')
_PICKS = re.compile(r'"picks":\[(\d+),(\d+)\]')
# Each player's own engine state name, read from a record line without parsing it: in the state's
# players list, each player object starts with "character" and names its "state" before any nested data.
_SEAT_STATE = re.compile(r'\{"character":"[^"\\]*",[^{]*?"state":"([^"\\]*)"')


def seat_states(line: str) -> list:
    """The deciding seats' own engine state names [p1, p2] from a record line (None where unread):
    the state-rarity boost's state classes (generic: whatever names the engine uses)."""
    i = line.find('"players":[')
    found = _SEAT_STATE.findall(line, i) if i >= 0 else []
    return (found[:2] + [None, None])[:2]


_CANON: dict[tuple, str] = {}


def _canon(value: Any) -> str:
    """Canonical JSON of an option value (memoized for flat dicts such as XY vectors: hot in counting)."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if type(value) is int:
        return str(value)
    if type(value) is dict and len(value) <= 4:
        try:
            key = tuple(sorted(value.items()))
            hit = _CANON.get(key)
        except TypeError:  # unhashable (nested) values
            key, hit = None, None
        if hit is None:
            hit = json.dumps(value, sort_keys=True, separators=(",", ":"))
            if key is not None and len(_CANON) < 100_000:
                _CANON[key] = hit
        return hit
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def candidate_options(c: Any) -> dict[str, str]:
    """A candidate's options as {option: canonical JSON value}, generic over every widget: each key of
    a `data` dict ("data.<key>"), or `data` itself when it is a bare XY dict, a string or a flag
    ("data"), and each key of `extra` ("extra.<key>"). `c` is a record's candidate dict or an
    engine Candidate."""
    data, extra = (c.get("data"), c.get("extra")) if isinstance(c, dict) else (c.data, c.extra)
    out: dict[str, str] = {}
    if isinstance(data, dict) and data and not set(data) <= {"x", "y"}:
        out.update((f"data.{k}", _canon(v)) for k, v in data.items())
    elif data is not None:
        out["data"] = _canon(data)
    if isinstance(extra, dict):
        out.update((f"extra.{k}", _canon(v)) for k, v in extra.items())
    elif extra is not None:
        out["extra"] = _canon(extra)
    return out


def decision(line: str, middle: bool = False, picked: bool = False) -> dict | None:
    """A record line's decision without parsing its (large) state and candidate lists: {"actions":
    [[action of each candidate] per seat], "picks": [i, j]}; with `picked` also "picked": each
    seat's played candidate (dict); with `middle` also the keys between the state and the candidates
    (search, value, explore, explore_mode, explore_pick, opening, strategies, ...). The actions are
    read with a regex (4x faster than json on set-4 records); a line the fast path can't read is
    parsed whole. None for a line with no decision (the header or the result)."""
    i = line.rfind('"candidates":')
    if i < 0:
        return None
    m = _PICKS.search(line, i)
    out: dict | None = None
    if m:
        body = line[i:m.start()]
        k = body.find('],[{"action":"')
        if k >= 0:
            hits = list(_ACTION.finditer(body))
            seats = [[h for h in hits if h.start() <= k], [h for h in hits if h.start() > k]]
            actions = [[h.group(1) for h in seat] for seat in seats]
            picks = [int(m.group(1)), int(m.group(2))]
            if all(0 <= picks[s] < len(actions[s]) for s in (0, 1)):
                out = {"actions": actions, "picks": picks}
                if picked:
                    try:
                        dec = json.JSONDecoder()
                        out["picked"] = [dec.raw_decode(body, seats[s][picks[s]].start())[0] for s in (0, 1)]
                    except ValueError:
                        out = None
    if out is not None and middle:
        j = line.find(',"actionable":')
        try:
            mid = json.loads("{" + line[j + 1:i].rstrip(", ") + "}") if 0 <= j < i else None
        except ValueError:
            mid = None
        out = None if mid is None else {**mid, **out}
    if out is None:
        d = json.loads(line)
        if "candidates" not in d or "picks" not in d:
            return None
        d.pop("state", None)
        out = {**{k: v for k, v in d.items() if middle and k != "candidates"},
               "actions": [[c["action"] for c in seat] for seat in d["candidates"]], "picks": list(d["picks"])}
        if picked:
            out["picked"] = [d["candidates"][s][d["picks"][s]] for s in (0, 1)]
    return out


def count_record_lines(lines: Iterable[str], counts: dict, options: dict | None = None,
                       states: dict | None = None) -> None:
    """Add one record's picks (its JSON lines) to counts[character][move type] and, with `options`,
    options[character][move type][option][value] (every option of the candidate played): each seat's
    pick on a decision where that seat had more than one candidate. With `states`,
    states[character][engine state name] counts those decisions by the seat's own state (seat_states)."""
    it = iter(lines)
    try:
        chars = json.loads(next(it))["characters"]
    except (StopIteration, ValueError, KeyError):
        return
    for line in it:
        d = decision(line, picked=options is not None)
        if d is None:
            continue
        sts = seat_states(line) if states is not None else (None, None)
        for seat in (0, 1):
            acts = d["actions"][seat]
            if len(acts) <= 1:
                continue
            if states is not None and sts[seat]:
                by_st = states.setdefault(chars[seat], {})
                by_st[sts[seat]] = by_st.get(sts[seat], 0) + 1
            move = acts[d["picks"][seat]]
            by_char = counts.setdefault(chars[seat], {})
            by_char[move] = by_char.get(move, 0) + 1
            if options is not None:
                by_move = options.setdefault(chars[seat], {}).setdefault(move, {})
                for opt, val in candidate_options(d["picked"][seat]).items():
                    by_val = by_move.setdefault(opt, {})
                    by_val[val] = by_val.get(val, 0) + 1


def count_tar(data: bytes) -> dict:
    """{"games": N, "counts": {character: {move type: picks}}, "options": {character: {move type:
    {option: {value: picks}}}}} for one chunk tar's records."""
    counts: dict = {}
    options: dict = {}
    states: dict = {}
    games = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not (m.isfile() and re.match(r"^g\d{12}-[0-9a-f]+\.jsonl\.gz$", Path(m.name).name)):
                continue
            with tf.extractfile(m) as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
                count_record_lines(f, counts, options, states)
            games += 1
    return {"games": games, "counts": counts, "options": options, "states": states}


def _tar_order(name: str) -> tuple:
    m = COUNT_TAR.match(name)
    return (int(m.group(2)), int(m.group(3)), int(m.group(4) or -1), m.group(1)) if m else (-1, -1, -1, name)


def window_tars(cache: dict, window_games: int) -> list[str]:
    """The cached tars in the window: newest first (by the refresh that counted them, then
    generation, chunk and part) until they hold `window_games` games."""
    out, games = [], 0
    for name in sorted(cache["tars"], key=lambda n: (cache["tars"][n]["seq"], _tar_order(n)), reverse=True):
        if games >= window_games:
            break
        out.append(name)
        games += cache["tars"][name]["games"]
    return out


def merge_counts(parts: Iterable[dict]) -> dict:
    """Nested count dicts (any depth, ints at the leaves) summed, keys sorted."""
    total: dict = {}

    def add(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict):
                add(dst.setdefault(k, {}), v)
            else:
                dst[k] = dst.get(k, 0) + v
    for p in parts:
        add(total, p)

    def ordered(d: dict) -> dict:
        return {k: ordered(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}
    return ordered(total)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def refresh_counts(names: list[str], get: Callable[[str], bytes | None], cache_path: Path, window_games: int,
                   taper: dict[str, float] | None = None, log: Callable[[str], None] = print,
                   state_taper: dict | None = None) -> dict:
    """Count every hub chunk tar in `names` not counted yet (COUNT_TAR, with its .sha256 sidecar;
    `get(name)` -> bytes or None), newest first, stopping once the new tars alone fill the window
    (older ones are then never needed: they are marked skipped). A tar whose download fails or
    disagrees with its sidecar is retried at the next refresh. The cache (per-tar counts) is pruned
    to the window and saved; returns the counts document (write it with `write_counts`)."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache.get("format") != COUNTS_FORMAT:  # a cache of another layout is recounted
        cache = {"format": COUNTS_FORMAT}
    cache.setdefault("seq", 0)
    cache.setdefault("tars", {})
    cache.setdefault("skipped", [])
    have, skipped, listed = set(cache["tars"]), set(cache["skipped"]), set(names)
    new = sorted((n for n in names if COUNT_TAR.match(n) and n not in have and n not in skipped
                  and n + ".sha256" in listed), key=_tar_order, reverse=True)
    seq = cache["seq"] + 1
    games_new = 0
    t0 = time.monotonic()
    counted = 0
    for i, name in enumerate(new):
        if games_new >= window_games:
            cache["skipped"] = sorted(skipped | set(new[i:]))
            break
        want = (get(name + ".sha256") or b"").decode("utf-8", "replace").strip().split()[:1]
        data = get(name)
        if data is None or not want or hashlib.sha256(data).hexdigest() != want[0]:
            log(f"least-tried counts: {name} missing or its sha256 differs; retried next refresh")
            continue
        entry = count_tar(data)
        cache["tars"][name] = {"seq": seq, **entry}
        games_new += entry["games"]
        counted += 1
    cache["seq"] = seq
    keep = set(window_tars(cache, window_games))
    cache["tars"] = {n: v for n, v in cache["tars"].items() if n in keep}
    cache["skipped"] = sorted(set(cache["skipped"]) & listed)  # names the hub has dropped are forgotten
    _atomic_write(cache_path, json.dumps(cache, sort_keys=True).encode("utf-8"))
    tars = window_tars(cache, window_games)
    doc = {"format": COUNTS_FORMAT, "window_games": window_games,
           "games": sum(cache["tars"][n]["games"] for n in tars), "tars": tars,
           "counts": merge_counts(cache["tars"][n]["counts"] for n in tars),
           "options": merge_counts(cache["tars"][n].get("options") or {} for n in tars),
           "states": merge_counts(cache["tars"][n].get("states") or {} for n in tars),
           "taper": dict(sorted((taper or {}).items())), "created": round(time.time(), 3)}
    if state_taper:
        doc["state_taper"] = {c: state_taper[c] for c in sorted(state_taper)}
    log(f"least-tried counts: {counted} new tars counted ({games_new} games, {time.monotonic() - t0:.0f}s); "
        f"window {doc['games']} games from {len(tars)} tars"
        + (f"; tapered: {', '.join(f'{c} {f:g}' for c, f in doc['taper'].items())}" if doc["taper"] else ""))
    return doc


def write_counts(doc: dict, path: Path) -> str:
    """Write a counts document; returns the file's sha256 (recorded in each record's header)."""
    data = json.dumps(doc, indent=1, sort_keys=True).encode("utf-8")
    _atomic_write(path, data)
    return hashlib.sha256(data).hexdigest()


# --- the fleet's shared counts file -------------------------------------------------------------------
# One writer (tools/round3_loop.py least-counts) refreshes the counts incrementally from new tars and PUTs
# the counts document to the hub as COUNTS_HUB_NAME with a sha256 sidecar; every worker downloads it at
# chunk start instead of counting the hub's tars itself.
COUNTS_HUB_NAME = "r3-least-tried-counts.json"


def check_shared(data: bytes | None, sidecar: bytes | None, window_games: int, max_age_s: float,
                 now: float | None = None) -> str | None:
    """Why a downloaded shared counts file can't be used (missing, sha256 differing from its sidecar,
    another format or window, older than `max_age_s`), or None when it can."""
    if not data:
        return "no shared counts file on the hub"
    want = (sidecar or b"").decode("utf-8", "replace").strip().split()[:1]
    if not want or hashlib.sha256(data).hexdigest() != want[0]:
        return "the shared counts file's sha256 differs from its sidecar"
    try:
        doc = json.loads(data)
    except ValueError:
        return "the shared counts file isn't JSON"
    if not isinstance(doc, dict) or doc.get("format") != COUNTS_FORMAT:
        return f"the shared counts file has format {doc.get('format') if isinstance(doc, dict) else None!r}"
    if doc.get("window_games") != window_games:
        return f"the shared counts file covers a window of {doc.get('window_games')} games, not {window_games}"
    age = (time.time() if now is None else now) - float(doc.get("created") or 0)
    if age > max_age_s:
        return f"the shared counts file is {age / 3600:.1f} h old (limit {max_age_s / 3600:g} h)"
    return None


def state_taper_from_coverage(doc: dict | None) -> dict:
    """{character: {"caught": [state classes caught up], "retired": bool}} from a coverage report's
    `state_taper` (tools/move_type_coverage.py); {} without one."""
    out = {}
    for char, t in ((doc or {}).get("state_taper") or {}).items():
        if isinstance(t, dict):
            out[char] = {"caught": sorted(str(x) for x in t.get("caught") or []), "retired": bool(t.get("retired"))}
    return out


def taper_from_coverage(doc: dict | None) -> dict[str, float]:
    """{character: floor} from a coverage report's JSON (tools/move_type_coverage.py `taper`): the
    characters whose legal move types and option values have all caught up. Floors outside (0, 1] are ignored."""
    out = {}
    for char, t in ((doc or {}).get("taper") or {}).items():
        f = t.get("floor") if isinstance(t, dict) else None
        if isinstance(t, dict) and t.get("tapered") and isinstance(f, (int, float)) and 0 < f <= 1:
            out[str(char)] = float(f)
    return out


# --- playing ----------------------------------------------------------------------------------------

def _unit(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16) / 16 ** 15


class LeastTried(NamedTuple):
    """The arm's settings and the counts file, as selfplay.play_recorded uses them."""
    p: float
    k: int
    window_games: int
    counts: dict[str, dict[str, int]]
    taper: dict[str, float]
    counts_sha256: str
    counts_games: int
    options: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    games: float = GAMES_DEFAULT
    expand: bool = False  # expand the chosen type's factors before choosing its option values
    state_boost: float = 0.0  # >1: the state-rarity boost's cap (settings "state_boost"); 0: off
    states: dict = {}  # counts file: decisions per character x own engine state
    state_taper: dict = {}  # counts file: per character {"caught": [...], "retired": bool}

    @classmethod
    def load(cls, settings: dict, counts_path: Path) -> "LeastTried":
        data = Path(counts_path).read_bytes()
        doc = json.loads(data)
        if doc.get("format") != COUNTS_FORMAT:
            raise ValueError(f"{counts_path}: counts format {doc.get('format')!r}, want {COUNTS_FORMAT}")
        return cls(settings["p"], settings["k"], settings["window_games"], doc.get("counts") or {},
                   doc.get("taper") or {}, hashlib.sha256(data).hexdigest(), int(doc.get("games") or 0),
                   doc.get("options") or {}, float(settings.get("games", GAMES_DEFAULT)),
                   bool(settings.get("expand", False)), float(settings.get("state_boost") or 0.0),
                   doc.get("states") or {}, doc.get("state_taper") or {})

    def header(self, selected: bool = True) -> dict:
        """The record header's `explore_least_tried`; with a share of games below 1, also "games" and
        whether this game is one of them ("selected")."""
        out = {"p": self.p, "k": self.k, "window_games": self.window_games, "counts_sha256": self.counts_sha256,
               "counts_games": self.counts_games, "taper": dict(sorted(self.taper.items()))}
        if self.games < 1:
            out.update(games=self.games, selected=bool(selected))
        if self.expand:
            out["expand"] = True
        if self.state_boost > 1:
            out["state_boost"] = self.state_boost
        return out

    def boost(self, character: str, state) -> float:
        """p's multiplier for a seat in engine state `state` (the state-rarity boost, generic): 1 when
        off, without counts, for a caught-up state class or a retired character (every class caught
        up: tools/move_type_coverage.py `state_taper`); else min(cap, sqrt(median count / its count)),
        shrinking continuously as its count nears the taper's minimum positions (to a floor share
        STATE_DECAY_FLOOR of the excess until its error is normal too)."""
        if self.state_boost <= 1 or not state:
            return 1.0
        st = self.state_taper.get(character) or {}
        if st.get("retired") or state in (st.get("caught") or ()):
            return 1.0
        counts = self.states.get(character) or {}
        if not counts:
            return 1.0
        vals = sorted(counts.values())
        med = vals[len(vals) // 2]
        n = int(counts.get(state, 0))
        raw = min(self.state_boost, (med / max(n, 1)) ** 0.5)
        if raw <= 1:
            return 1.0
        decay = max(STATE_DECAY_FLOOR, 1 - n / TAPER_DEFAULTS["min_positions"])
        return 1 + (raw - 1) * decay

    def game_selected(self, seed: int) -> bool:
        """Whether the game with this seed takes part (a seeded draw below `games`; every game at 1)."""
        return self.games >= 1 or _unit(f"least-tried-games:{seed}") < self.games

    def seat_on(self, seed: int, seat: int, character: str) -> bool:
        """Whether this game's `seat` may explore: in every selected game, unless its character is
        tapered; then in a seeded share (the floor) of all games: floor / games of the selected ones."""
        if not self.game_selected(seed):
            return False
        floor = self.taper.get(character)
        return floor is None or _unit(f"least-tried-taper:{seed}:{seat}") < min(1.0, floor / self.games)

    def choose(self, cands: list[list[Any]], characters: tuple[str, str] | list[str], used: list[int],
               on: list[bool], rng: Any,
               expand: Callable[[int, list[int]], int] | None = None,
               states=None) -> tuple[int, int, str, dict] | None:
        """One decision's draw: (seat, candidate index, move type, explore_pick fields) or None.
        Eligible seats are on, under k picks this game and with more than one move type; with
        probability p one of them (at random) explores: its least-picked legal type, then per option
        with a choice among that type's candidates its least-picked value, then a candidate matching
        the most of those values (uniform among them; see the module docstring).

        With `expand` (settings "expand": true; set 6's expand-only option values): once the type is
        chosen, `expand(seat, indices)` asks the engine for that type's factor variants (appended to
        `cands[seat]` in place, returning how many were added), so the option values are chosen over
        the base candidates and every variant. At most EXPAND_MAX of the type's candidates are
        expanded (a seeded sample when there are more). Without it, nothing changes (no extra draw)."""
        seats = [s for s in (0, 1) if on[s] and used[s] < self.k and len({c.action for c in cands[s]}) > 1]
        boosts = None
        if self.state_boost > 1:  # per-seat p (the state-rarity boost)
            boosts = {s: self.boost(characters[s], (states or (None, None))[s]) for s in seats}
            u = rng.random() if seats else 1.0
            seats = [s for s in seats if u < min(1.0, self.p * boosts[s])]
            if not seats:
                return None
        elif not seats or rng.random() >= self.p:
            return None
        seat = seats[rng.randrange(len(seats))]
        char = characters[seat]
        by_move: dict[str, list[int]] = {}
        for i, c in enumerate(cands[seat]):
            by_move.setdefault(c.action, []).append(i)
        tried = self.counts.get(char) or {}
        low = min(int(tried.get(m, 0)) for m in by_move)
        tied = sorted(m for m in by_move if int(tried.get(m, 0)) == low)
        move = tied[rng.randrange(len(tied))]
        idx = by_move[move]
        added = None
        if self.expand and expand is not None:
            base = idx if len(idx) <= EXPAND_MAX else sorted(rng.sample(idx, EXPAND_MAX))
            added = int(expand(seat, base))
            idx = [i for i, c in enumerate(cands[seat]) if c.action == move]
        opts = [candidate_options(cands[seat][i]) for i in idx]
        tried_opts = (self.options.get(char) or {}).get(move) or {}
        chosen: dict[str, str] = {}
        counts_at: dict[str, int] = {}
        for key in sorted(set().union(*opts)):
            values = sorted({o[key] for o in opts if key in o})
            if len(values) < 2:
                continue
            seen = tried_opts.get(key) or {}
            least = min(int(seen.get(v, 0)) for v in values)
            ties = [v for v in values if int(seen.get(v, 0)) == least]
            chosen[key] = ties[rng.randrange(len(ties))]
            counts_at[key] = least
        score = [sum(o.get(key) == v for key, v in chosen.items()) for o in opts]
        best = max(score)
        pool = [i for i, sc in zip(idx, score) if sc == best]
        return seat, pool[rng.randrange(len(pool))], move, {
            "options": chosen, "counts_at_pick": {"move_type": low, "options": counts_at},
            "matched": best, "of": len(chosen), **({"expanded": added} if added is not None else {}),
            **({"state": (states or (None, None))[seat], "boost": round(boosts[seat], 4)} if boosts is not None else {})}
