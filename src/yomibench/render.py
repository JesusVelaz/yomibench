"""Render replay files to video with the real game.

    yomibench render REPLAY [REPLAY ...] --out DIR

The game plays each replay exactly as its replay browser would (mod/YomiBench/YomiRender.gd),
saving every frame and logging every sound it starts; ffmpeg then encodes them to MP4, with an
audio track rebuilt from those sounds (render_audio.py; game music off by default). `--fixed-fps 60`
gives every frame the same game time however slowly it draws, so a slow software renderer still
yields a normal-speed video.

Needs a display: on Windows the game opens a window while it renders. On Linux (or Docker on a
Mac, see docker/render.sh) run it under Xvfb.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from yomibench.game import EXE, LINUX, build_run_dir, find_game_dir, popen_hidden
from yomibench.render_audio import mix

RUN_DIR_NAME = "render"  # separate from the servers' run dir, so rendering never touches them


def render(replays: list[Path], out_dir: Path, game_dir: str | None = None, run_dir: Path | None = None,
           tail: int = 150, scale: int = 2, keep_frames: bool = False, timeout: float = 1800,
           music: bool = False, states: bool = False) -> list[Path]:
    """Render each replay to OUT_DIR/<replay stem>.mp4 and return the video paths. Without ffmpeg on
    PATH, frames are kept in OUT_DIR/<stem>.frames and no video is made. With `states`, no video:
    the game logs every tick it plays to OUT_DIR/<stem>.states.jsonl (see YomiRender.gd), and those
    paths are returned."""
    if LINUX and not os.environ.get("DISPLAY"):
        raise SystemExit("rendering needs a display; run under Xvfb (e.g. docker/render.sh or xvfb-run)")
    run = build_run_dir(find_game_dir(game_dir), run_dir or _default_render_dir(),
                        user_dir_name="YomiBenchRender", main_autoload=("YomiRender", "YomiRender.gd"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("YOMIBENCH_FFMPEG") or shutil.which("ffmpeg")
    # Native Windows renders run on the never-shown desktop (yomibench.hidden_desktop; the GPU still draws), muted,
    # at below-normal priority. Linux renders run under Xvfb as before.
    win_flags = ["--no-window"] if os.name == "nt" else []
    popen_kw = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS}
                if os.name == "nt" else {})
    videos = []
    for replay in replays:
        if states:
            log = out_dir / f"{replay.stem}.states.jsonl"
            log.unlink(missing_ok=True)
            args = [str(run / EXE), "--audio-driver", "Dummy", "--resolution", "320x180", "--fixed-fps", "60",
                    "--disable-vsync", f"--yomi-replay={replay.resolve().as_posix()}",
                    f"--yomi-frames={out_dir.resolve().as_posix()}", f"--yomi-states={log.resolve().as_posix()}"]
            _run_hidden(args, run, timeout, popen_kw)
            if not log.exists() or not log.stat().st_size:
                raise SystemExit(f"{replay}: the game logged no states (is it a replay from game version 1.10?)")
            print(f"{replay.name}: states -> {log}")
            videos.append(log)
            continue
        frames = out_dir / f"{replay.stem}.frames"
        shutil.rmtree(frames, ignore_errors=True)
        frames.mkdir()
        args = [str(run / EXE), "--audio-driver", "Dummy", "--resolution", "640x360", "--fixed-fps", "60",
                "--disable-vsync", *win_flags, f"--yomi-replay={replay.resolve().as_posix()}",
                f"--yomi-frames={frames.resolve().as_posix()}", f"--yomi-tail={tail}"]
        t_render, start = time.monotonic(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _run_hidden(args, run, timeout, popen_kw)
        count = len(list(frames.glob("f*.png")))
        if not count:
            raise SystemExit(f"{replay}: the game saved no frames (is it a replay from game version 1.10?)")
        if not ffmpeg:
            print(f"{replay.name}: {count} frames in {frames} (install ffmpeg to encode them)")
            continue
        video = out_dir / f"{replay.stem}.mp4"
        # The game's sound, rebuilt from the sound events YomiRender.gd logged (see render_audio).
        log, sounds = Path(f"{frames}.sounds.jsonl"), Path(f"{frames}.sounds")
        audio, placed = Path(f"{frames}.wav"), 0
        if log.exists():
            placed = mix(log, sounds, audio, count, ffmpeg, music=music)
        audio_in = ["-i", str(audio)] if placed else []
        audio_out = ["-c:a", "aac", "-b:a", "128k"] if placed else []
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-framerate", "60", "-i", str(frames / "f%06d.png"),
                        *audio_in, "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", *audio_out, "-movflags", "+faststart", str(video)],
                       check=True)
        if not keep_frames:
            shutil.rmtree(frames)
            shutil.rmtree(sounds, ignore_errors=True)  # game assets: never kept beside the videos
            for p in (log, audio):
                p.unlink(missing_ok=True)
        print(f"{replay.name}: {count} frames, {placed} sounds -> {video}")
        print(json.dumps({"render": replay.stem, "render_start_utc": start,
                          "render_s": round(time.monotonic() - t_render, 1)}), flush=True)
        videos.append(video)
    return videos


def _run_hidden(args, cwd, timeout, popen_kw) -> None:
    """Run one game instance to completion with no window on the desktop (yomibench.game.popen_hidden)."""
    p = popen_hidden(args, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **popen_kw)
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def _default_render_dir() -> Path:
    base = os.environ.get("YOMIBENCH_RUN_DIR")
    return (Path(base).parent if base else Path.home() / ".yomibench") / RUN_DIR_NAME
