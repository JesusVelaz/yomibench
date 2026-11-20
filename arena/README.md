# Arena data

The Arena is the site's home page: a leaderboard of bots under one fixed set of ranked settings, with the same
per-move thinking budget for every bot. Everything here is data. No code from a submission is ever run.

- `spec.json`: the versions every rated game runs on: the game (version, Steam build, the sha256 of its exe and
  pck), the engine build, the mod fingerprint, the play code's sha256, the candidate set and its rules hash, the net
  features version, the match settings, the characters and the turn cap. Machines that play rated games check all
  of it before playing and refuse on any difference. Its hash (the first 12 hex digits of the sha256 of its
  canonical JSON, without `notes`) names the rating era: when the spec changes, a new era starts.
- `house.json`: the house bot's generations and each one's model sha256.
- `bots.json`: the entered bots other than the house generations. Each is
  `{"name", "owner", "digest", "settings", "entered_utc"}`, from a submission that passed
  `tools/validate_submission.py`.
- `results/ranked.jsonl`: rated games, one row per microshard (a few seat-swapped pairs of one pairing of bots):
  `{"a", "b", "score", "games", "pairs", "matchups": {"Cowboy vs Ninja": {"score", "games"}}, "spec", "actual",
  "seed", "block", "shard", "utc", "games_detail"}`. `score` is `a`'s share of the points (a draw counts half).
  `spec` is the spec hash it was rated under and `actual` the versions the machine really ran.

Until every bot has 100 games in the current era, the house generations' promotion tests (from the releases) fill
in the ratings. `site/build.py` fits Bradley–Terry ratings over it, and when a results file changes on `main`, the
site rebuilds.
