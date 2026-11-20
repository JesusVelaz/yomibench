# Playing against the bot

The vs-bot mod adds **Multiplayer (vs bot)** to the game's main menu. You play a normal multiplayer match: you
both choose at the same time and see only your own panel. The bot is a second, invisible copy of your game that
joins your match through 127.0.0.1 only, so nobody else can see or join it.

## What you need

- The game through Steam (Windows, version 1.10.0), **started from Steam**. The mod only runs in your own, owned
  copy.
- This repository set up (`docs/SETUP.md`), and a bot: your own trained net (`docs/TRAINING.md`) or our published
  weights.
- About 1 GB of free memory while you play.

## Install

1. Build the mod: `uv run python tools/build_vsbot_mod.py --out dist` gives `dist/YomiBenchVsBot.zip`.
2. In Steam: right-click the game → **Manage** → **Browse local files**. Put the zip, still zipped, in the `mods`
   folder there (create it if it's missing).
3. Start the game from Steam. If **Multiplayer (vs bot)** isn't in the main menu, turn mods on in the game's mod
   settings and restart (the game turns them off once after an update).
4. Click **Multiplayer (vs bot)** once. Since no bot server is set yet, it tells you where its settings file is
    (`yomibench_vsbot.json`). Write the bot server command there:

        {"server": "C:/path/to/yomibench/.venv/Scripts/python.exe",
         "args": ["-m", "yomibench.human_play", "--engine", "game-pool:auto", "--bot", "current-best"]}

    `current-best` plays the net named by `~/.yomibench/best.json`. Set `YOMIBENCH_BEST` to another `best.json` to
   choose a different one; the training loop writes one in its output folder.

## Play

1. Click **Multiplayer (vs bot)**. The first start takes a few seconds while the bot's copy of the game is set up.
2. In the panel, pick your character (P1) and the bot's (P2), the settings (*ranked* or *standard*), and how the
   bot picks: *mixed* plays its solved strategy, the way it trained; *strongest* always plays its top move.
3. Press **Start game**. The turn timer is the game's own multiplayer timer, and the bot always answers inside
   it. It also never sees your move before it answers.

## Good to know

- **Local only:** the mod won't start while you're in an online session or a Steam lobby, or while gameplay mods
  are on. Online and ranked play are untouched.
- **Your games:** each match is saved on your PC, in the folder the bot server was given (`--out`), with no
  player names. That folder is marked so training never uses it.
- **Sync:** the bottom bar should say the bot is **in sync**. A game that ends **out of sync** doesn't count;
  please report it with the game's log.
