# Setup

## What you need

- *Your Only Move Is HUSTLE* installed **through Steam** (Windows; version 1.10.0, the current one).
  yomibench only runs your own Steam install: a folder Steam put under a library's `steamapps/common`, next to
  the game's app manifest. Copies elsewhere are refused.
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- About 250 MB of RAM per headless game copy, and a few cores. `game-pool:auto` sizes itself to your PC.

## Install

```
git clone https://github.com/Stixels/yomibench && cd yomibench
uv sync --extra train --extra test
uv run pytest -q                       # tests that need the game skip without it
```

## Finding the game

yomibench looks in your Steam libraries (the default Steam folder and every library it lists). To point it at a
specific library's install, pass `--game-dir` or set `YOMIBENCH_GAME_DIR`; it must still be a Steam install.

Each headless copy is built in `~/.yomibench/run` (`YOMIBENCH_RUN_DIR`) from your install: the game's exe, pack and
libraries, these scripts, and a small `override.cfg`. Your install is only read. Copies start with no window
(on Windows they run on a separate, never-shown desktop) and with sound off.

## Engines

| spec | what |
|---|---|
| `toy` | a tiny built-in game, for trying things without the game |
| `game` | one headless copy of your game |
| `game-pool:N` / `game-pool:auto` | N headless copies acting as one engine (sized to this PC with `auto`) |
| `tcp:HOST:PORT` | an engine server already running (e.g. `yomibench serve`) |
| `cmd:...` | any program that speaks the protocol on stdin/stdout |

## Linux and macOS

The game's Linux build runs in Docker (`docker/`). Install the Linux build with your own Steam account (Steam or
SteamCMD), mount that library's `steamapps` folder into the container, and set `YOMIBENCH_GAME_DIR` to
`/steamapps/common/YourOnlyMoveIsHUSTLE` inside it. On Apple Silicon the image runs under x86 emulation, slower.

## Troubleshooting

- **"yomibench runs only your own Steam install"**: point `--game-dir` at the folder Steam installed
  (Steam → the game → Manage → Browse local files).
- **"engine 'sim' isn't available in this build"**: that engine isn't part of this repository; use
  `game-pool:auto`.
- **Windows, Microsoft Store Python:** keep the run folder outside `%LOCALAPPDATA%` (the default is fine); the Store
  Python redirects writes there and the game then can't find its libraries.
