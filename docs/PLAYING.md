# Play against HustleBot

Play a match against **HustleBot g12** on your own PC. It runs inside your own copy of the game, through a mod that
adds **Multiplayer (vs bot)** to the main menu. Everything stays on your PC.

> **Today this takes a few commands.** There's no one-click installer yet. Setup takes about 10 minutes the first
> time. You type the commands in PowerShell, and you can copy and paste every one of them.

## What you need

- **Windows** (10 or 11).
- ***Your Only Move Is HUSTLE* from Steam**, version 1.10.0 (the current one), started at least once. The mod
  only runs in your own copy, started from Steam. We never share game files.
- About **1 GB of free memory** while you play, and about 300 MB of disk space.
- An internet connection for setup. Playing works offline.

You don't need to install Python yourself: the `uv` tool in step 2 gets it for you.

The newest public bot is **HustleBot g12**. Newer generations (g13, g14) are on the site but their weights aren't
public yet: they're published two generations behind the current best.

## Set up (once)

1. **Make a folder** called `C:\HustleBot`. The steps below use this exact path, so the commands work as written.

2. **Install uv.** Open **PowerShell** (Start menu → type `PowerShell` → Enter), paste this and press Enter:

        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    Then **close PowerShell and open it again**, so it finds `uv`.

3. **Download the code:** [yomibench-main.zip](https://github.com/Stixels/yomibench/archive/refs/heads/main.zip).
    Right-click it → **Extract All…** → change the folder to `C:\HustleBot` → **Extract**. You should now have
    `C:\HustleBot\yomibench-main`.

4. **Download the bot:** [gen-12.npz](https://github.com/Stixels/yomibench/releases/download/gen-12/gen-12.npz)
    (about 400 KB, from the [HustleBot g12 release](https://github.com/Stixels/yomibench/releases/tag/gen-12)).
    Move it into `C:\HustleBot`, so you have `C:\HustleBot\gen-12.npz`.

5. **Build the mod.** In PowerShell, paste:

        cd C:\HustleBot\yomibench-main
        uv sync --extra net
        uv run python tools/build_vsbot_mod.py --out dist

    The last line should end with `dist\YomiBenchVsBot.zip ... bytes`. The first command can take a minute.

6. **Put the mod in the game.** In Steam, right-click the game → **Manage** → **Browse local files**. Open the
    `mods` folder there (create it if it's missing) and copy `C:\HustleBot\yomibench-main\dist\YomiBenchVsBot.zip`
    into it. **Don't unzip it.**

7. **Tell the mod where the bot is.** In PowerShell, paste this as one line:

        Set-Content -Encoding ascii "$env:APPDATA\YourOnlyMoveIsHUSTLE\yomibench_vsbot.json" '{"server": "C:/HustleBot/yomibench-main/.venv/Scripts/python.exe", "args": ["-m", "yomibench.human_play", "--engine", "game-pool:auto", "--bot", "matrix-do:C:/HustleBot/gen-12.npz@5"]}'

    It prints nothing when it works.

## Play

1. **Start the game from Steam.** The main menu now shows **Multiplayer (vs bot)**.
2. **Click Multiplayer (vs bot).** A panel opens. The first time, the bot takes up to a minute to get ready.
3. **Pick the characters:** **You (P1)** and the bot (P2), each one of Ninja, Cowboy, Wizard, Mutant or Robot.
   HustleBot is strongest as Cowboy.
4. **Leave the rest as it is** (*ranked* settings, *mixed* picks) and press **Start game**.
5. **Play.** It works like an online match: you both choose at the same time and see only your own panel. The bot
   always answers within the game's turn timer, and it never sees your move first.
6. **When the match ends**, play again from the panel, or press **Back to main menu**. Quitting the game closes the
   bot too.

## If something goes wrong

| What you see | What to do |
|---|---|
| No **Multiplayer (vs bot)** in the menu | Mods are off: turn them on in the game's mod settings and restart the game. Check that the zip sits directly in `mods`, still zipped. |
| "The YomiBench bot server isn't installed" | The settings file from step 7 is missing. Run step 7 again. |
| "Couldn't start the bot server" | Check that `C:\HustleBot\yomibench-main\.venv\Scripts\python.exe` exists. If not, run step 5 again. |
| "Lost the bot server" | Check that `C:\HustleBot\gen-12.npz` exists, then restart the game. |
| "needs the game started through Steam" | Close the game and start it from Steam. |
| "off while ..." | Leave the online lobby or turn off gameplay mods (character mods are fine). |
| `uv` is not recognized | Close PowerShell and open it again after step 2. |
| Step 7 says it can't find part of the path | Start the game once from Steam, quit it, and run step 7 again. |
| The bar at the bottom says **out of sync** | That game doesn't count. Please [report it](https://github.com/Stixels/yomibench/issues) with the file `yomibench_vsbot.log` from `%APPDATA%\YourOnlyMoveIsHUSTLE`. |

## Good to know

- **Local only:** the bot's side is a second, invisible copy of your game that joins your match through
  127.0.0.1, so nobody else can see or join it. The mod won't start while you're in an online session or a Steam
  lobby. Online and ranked play are untouched.
- **Your games** are saved on your PC in `Documents\YomiBench\games-root`, with no player names. Nothing is
  uploaded, and training never uses them.
- **Bot picks:** *mixed* plays its solved strategy, the way it trained, so it can't be read. *strongest* always
  plays its top move.

## Remove it

1. Delete `YomiBenchVsBot.zip` from the game's `mods` folder.
2. Delete the folder `C:\HustleBot`.
3. Optional: in `%APPDATA%\YourOnlyMoveIsHUSTLE`, delete `yomibench_vsbot.json`, `yomibench_vsbot.log` and the
   folders `yomibench-mp-client` and `yomibench-server` if they're there. Also delete the folders
   `%APPDATA%\YourOnlyMoveIsHUSTLE-YomiBenchClient` and `%USERPROFILE%\.yomibench`, and `Documents\YomiBench` if you
   don't want your saved games.
4. Optional, if you don't use uv for anything else: delete `uv.exe` and `uvx.exe` from `%USERPROFILE%\.local\bin`,
   and the folders `%APPDATA%\uv` and `%LOCALAPPDATA%\uv`.

## For developers

To play your own trained net, point `--bot` at it instead of `gen-12.npz` (`matrix-do:PATH/TO/MODEL.npz@5`), or use
`--bot current-best`, which plays the net named by `~/.yomibench/best.json` (set `YOMIBENCH_BEST` to another
`best.json`; the training loop writes one in its output folder). See [Setup](SETUP.md) and [Training](TRAINING.md).
