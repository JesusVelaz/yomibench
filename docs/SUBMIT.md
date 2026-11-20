# Submit a bot to the Arena

> **Status: the checks are live, rated matches open soon.** You can submit now and the checks will tell you whether
> your bot is ready. Valid bots are played once rated matches start, and their results then appear on the site.

The Arena rates bots against each other under one fixed set of ranked settings, and every bot gets the same
thinking budget per move. In phase 1 you submit **data only**: your trained nets and a few search settings for our
existing search. Nothing you send is ever run as code.

**[Open the submission form](https://github.com/Stixels/yomibench/issues/new?template=submit-bot.yml)**

## What you send

### A model bundle

Your nets, packed with the harness. The one-PC training loop names its current best net in `OUT/best.json`
(its `model` field):

```
yomibench bundle pack --name my-bot-1 --owner your-github-username --value PATH/TO/BEST_MODEL.npz
```

The Arena's search needs a value net (`--value`). A policy net (`--policy policy.npz`) is optional; with one, your
bot can prune its search (see Settings). Nets are arrays only and load with `allow_pickle=False`.

### An entry

A small JSON object:

```json
{"name": "my-bot-1", "owner": "your-github-username", "bundle_sha256": "<sha256 of the zip>", "candidate_set": 6,
 "settings": {}, "notes": "trained 2 days on one PC"}
```

| Key | What it is |
| --- | --- |
| `name` | The bundle's name: 1-40 of a-z 0-9 . _ -, starting and ending with a letter or digit. A new version needs a new name (`my-bot-2`). Names of house generations (`gen-13`, `g13`) and built-in bots are reserved. |
| `owner` | Your GitHub username: the name in your profile's address (`github.com/USERNAME`), not your display name. It must be the account you open the submission from; any other name is refused. |
| `bundle_sha256` | The zip's sha256: `sha256sum my-bot-1.zip`, or on Windows `(Get-FileHash my-bot-1.zip).Hash.ToLower()`. |
| `candidate_set` | The candidate set your nets were trained on. It must be the Arena's (6 today, in `arena/spec.json`). |
| `settings` | Optional. See below. |
| `notes` | Optional, up to 500 characters, shown on your bot's page. |

### Settings

The budget per move is the Arena's and is the same for everyone. Settings can only narrow how your bot spends it.
Both need a policy net in the bundle:

| Setting | Values | Effect |
| --- | --- | --- |
| `prune_k` | 4 to 32 | Your policy net keeps its top K candidate moves before the value net's search. |
| `prune_scope` | `own` or `both` | With `prune_k`: prune only your own moves (the default), or both sides' moves in your search. |

Leave `settings` empty to play the full search with your value net, like HustleBot, the house bot.

## Check it on your PC first

```
python tools/validate_submission.py entry.json my-bot-1.zip
```

These are the same checks the form runs:

- the bundle's size (25 MB at most) and contents, and every net's size, with nothing loaded before it's measured;
- its format and file hashes, and that each net loads;
- that every weight is a finite number and the value net can score a position;
- that your nets aren't HustleBot's own (checked by their arrays against the house models, published or not:
  weights are published two generations behind the current best, and entering them as yours is refused);
- that the nets' features version and your candidate set are the ones the Arena plays;
- the entry's keys, types and ranges.

## Submit

Open the **[submission form](https://github.com/Stixels/yomibench/issues/new?template=submit-bot.yml)**, paste
your entry and drag the zip in. A check runs on the issue and comments the result; edit the issue to fix anything
it finds, and the check runs again.

## What happens next

Once rated matches open, valid bots join the queue. Our machines play the rated matches at low priority, on spare
capacity, under the Arena's settings against the house generations and every other entry. Results are committed as
data (`arena/results/`), and the site rebuilds with your bot's rating, its record by matchup and your notes.

Rules: submit nets you trained or have the right to enter. No game files, and no data from other players' replays.
Bots never play online; the Arena runs locally on our machines.

## Versions

Every rated game runs on one pinned set of versions, in `arena/spec.json`: the game (1.10.0, its Steam build), the
engine build, the mod, the play code, the candidate set and its rules, the net features version and the match
settings. Every result records the versions it actually ran on, and the leaderboard says which spec it rates. When
the spec changes, a new rating era starts.

## Later: phase 2

Bots with their own code (a custom search, not just nets) will come later, run in a sandbox. Phase 1 is nets and
settings only.
