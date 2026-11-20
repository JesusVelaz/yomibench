# Training a bot on one PC

```
uv run python tools/train_loop.py --out ~/.yomibench/train
```

Each cycle:

1. **Self-play.** The current best plays itself (plain search before the first promotion) on `game-pool:auto`,
   headless copies of your own game, for `--games-per-cycle` games (default 200).
2. **Train.** A small value network learns from every cycle's records so far.
3. **Gate.** The new net plays the current best in seat-swapped pairs over every pairing of `--characters`
   (`--pairs` each, a fixed number). The decision rule is fixed before any game is played (`tools/pool_eval.py`
   prints it first): the new net is promoted only on a clear win over the current best.
4. **Promote.** If the new net wins clearly, it becomes the best: `OUT/best.json` names it (generation, model,
   sha256) and the model is kept as `OUT/models/<sha256>.npz`.

Every stage is skipped when its output exists, so stopping and rerunning the same command resumes. `--cycles N` stops
after N cycles; the default keeps going.

## Starting from our weights

The real game is slow: one PC plays tens to low hundreds of games an hour. Starting from scratch takes days before
the first promotions. To continue from HustleBot's published weights (each release's `gen-N.npz`) instead:

```
uv run python tools/train_loop.py --out ~/.yomibench/train --start-from gen-N.npz
```

## Settings worth knowing

| flag | default | |
|---|---|---|
| `--engine` | `game-pool:auto` | any engine spec (`docs/SETUP.md`) |
| `--candidate-set` | `6` | how finely the bot's options are discretized; 6 matches our published nets |
| `--characters` | all five | the gate plays every ordered pairing of these |
| `--games-per-cycle` | 200 | more games per cycle: steadier training, slower cycles |
| `--pairs` | 30 | gate pairs per pairing |
| `--procs` | 1 | parallel self-play and gate processes |
| `--below-normal` | off | run every stage at below-normal priority (Windows), to keep the PC responsive |

## Reading the results

- `OUT/best.json`: the current best (generation and model).
- `OUT/cycle-K/gate.jsonl`: the gate's games and its summary line (`pool_summary`): the win rate per pairing with a
  95% interval, and the decision.
- `OUT/ledger.jsonl`: one row per stage with its timing and counters.

Play your best net with the mod: `--bot current-best` reads `~/.yomibench/best.json`, or set `YOMIBENCH_BEST` to your
`OUT/best.json` ([PLAYING.md](PLAYING.md)).
