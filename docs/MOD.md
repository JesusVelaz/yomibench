# The game mod

Two pieces of GDScript run inside the game's own executable. Nothing from the game is in this repository: only
these scripts, which call the game's own scenes and methods.

| Folder | What it is |
|---|---|
| `mod/YomiBench/` | the **engine server**: turns a headless copy of the game into a protocol engine (`docs/PROTOCOL.md`) on 127.0.0.1 |
| `mod/YomiBenchVsBot/` | the **vs-bot mod**: adds Multiplayer (vs bot) to the menu (`docs/PLAYING.md`) |

## How the engine server runs

`yomibench serve`, `game` and `game-pool` engines build a run folder from your Steam install: the game's exe, pack
and libraries, these scripts, and an `override.cfg`. It adds `YomiServer` as an autoload, gives the copy its own user
folder, and replaces the Steam autoload with `SteamStub.gd`, because a headless copy needs no Steam session; it's
made from your own install, on your PC. The copy starts with `--no-window --disable-render-loop --audio-driver Dummy`.

## How a turn is answered

- **Candidates** (`Candidates*.gd`): for each legal menu move, its data widgets (aim, timing, direction) are set to
  a grid of values through the widgets' own methods, and the extra panel (DI, feint, reverse, per-character toggles)
  gives more variants. Sets 2 and 4 are frozen. Set 5 adds Cowboy's draw pairs, finer gun aims and bullet DI; set 6
  adds panel sets (a move's single extra-panel changes made as a player makes them, and pairs of them).
  - Later sets are **layers** (`Candidates.gd` `LAYERS`): each one adds `extra_set` values after the earlier ones',
    so every earlier index keeps its meaning.
- **Legality** (`Legality.gd`): a hidden pair of the game's own action panels answers which moves are legal, so
  legality is always the game's.
- **Previews** (`YomiMatch.gd`): each candidate pair is played out in a reused preview game that copies the real
  match's state first (fighters, objects, the RNG), then reports the resulting state.
- **Raw inputs** (`step_raw`): a recorded game's exact inputs, played as given. Inputs a player couldn't have sent
  are refused.

## Rules hashes

Every candidate set's rules have a hash, reported by `hello` as `candidate_rules`: the rules file plus the frozen
generator, reduced to tokens (comments and whitespace don't count). Records carry it, so a game always replays on
the rules it was played with. `rules_aliases.json` maps a computed hash to the frozen one when a file changes
without changing any list. `tools/candidates_replay.py` checks that an engine reproduces a record's candidate
lists, position by position.

## Extending it

- **A new candidate set:** add its rules file, add a layer to `Candidates.gd` (`_<key>_slots` and `_<key>_extras`),
  give it a rules hash in `YomiServer.gd`, and check it with `tools/candidates_replay.py` on recorded games.
- **A new engine command:** add it to `YomiServer.gd`'s dispatch and document it in `docs/PROTOCOL.md`.
