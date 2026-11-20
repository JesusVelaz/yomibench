"""Audio track for a rendered replay, rebuilt from the game's own sound events.

A render runs slower than real time (`--fixed-fps 60` stamps every frame 1/60 s however long it takes
to draw), while the game's mixer runs in real time, so the sound can't be recorded live. Instead
mod/YomiBench/YomiRender.gd logs every sound the game starts, on the video frame it starts, and saves
each sound file once; `mix` plays them back at those frames, the way the game's mixer would:

- player volume, plus the bus chain's volume (bus -> Master), muted buses dropped
- pitch_scale as a resample (it changes speed and pitch together, as in the game)
- AudioStreamPlayer2D: Godot 3's distance attenuation and left/right pan for a 640-wide screen
- looping streams (the music) play until the player stops or the video ends; one-shot sounds play to
  their end. A one-shot's logged "stop" is ignored: the real-time mixer finishes sounds early
  relative to the slow render, so its stops don't mark game time.

Pure Python (the render container has no numpy); ffmpeg decodes the saved files.
"""

from __future__ import annotations

import json
import math
import subprocess
import wave
from array import array
from pathlib import Path

RATE = 44100
FPS = 60
SCREEN_WIDTH = 640  # the game's viewport width, for 2D panning


def _db(x: float) -> float:
    return 10 ** (x / 20)


def _decode(ffmpeg: str, path: Path) -> array:
    raw = subprocess.run([ffmpeg, "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "2",
                          "-ar", str(RATE), "-"], capture_output=True, check=True).stdout
    pcm = array("h")
    pcm.frombytes(raw)
    return pcm


def _bus_gain(buses: dict, name: str) -> float:
    gain, seen = 1.0, set()
    while name and name in buses and name not in seen:
        seen.add(name)
        b = buses[name]
        if b.get("mute"):
            return 0.0
        gain *= _db(b.get("volume_db", 0.0))
        name = b.get("send", "")
    return gain


def mix(log: Path, sounds: Path, out_wav: Path, frames: int, ffmpeg: str = "ffmpeg", music: bool = False) -> int:
    """Mix the logged sounds into OUT_WAV (16-bit stereo, RATE), FRAMES/60 s long. Returns the
    number of sounds placed. MUSIC off (the default) drops the Music bus: the in-game replays we
    match were watched with music off, so their sound is effects only."""
    events = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    buses = next((e["buses"] for e in events if e["ev"] == "buses"), {})
    if not music and "Music" in buses:
        buses["Music"] = dict(buses["Music"], mute=True)
    streams = {e["file"]: e for e in events if e["ev"] == "stream" and e.get("file")}
    total = int(frames / FPS * RATE)
    out = array("f", bytes(4 * 2 * total))
    stop_at: dict[int, list[int]] = {}
    for e in events:
        if e["ev"] == "stop":
            stop_at.setdefault(e["id"], []).append(e["f"])
    cache: dict[str, array] = {}
    placed = 0
    for e in events:
        if e["ev"] != "start" or not e.get("file") or e["file"] not in streams:
            continue
        info = streams[e["file"]]
        if e["file"] not in cache:
            cache[e["file"]] = _decode(ffmpeg, sounds / e["file"])
        pcm = cache[e["file"]]
        n_src = len(pcm) // 2
        if not n_src:
            continue
        gain = _db(e.get("volume_db", 0.0)) * _bus_gain(buses, e.get("bus", "Master")) / 32768
        gl = gr = gain
        if "dx" in e:  # AudioStreamPlayer2D, as Godot 3 mixes it
            dist = math.hypot(e["dx"], e["dy"])
            md = e.get("max_distance", 2000.0) or 2000.0
            if dist >= md:
                continue
            g = (1.0 - dist / md) ** e.get("attenuation", 1.0)
            pan = min(1.0, max(0.0, (e["dx"] + SCREEN_WIDTH * 0.5) / SCREEN_WIDTH))
            gl, gr = gain * g * (1.0 - pan), gain * g * pan
        pitch = e.get("pitch", 1.0) or 1.0
        start = int(e["f"] / FPS * RATE)
        loop = bool(info.get("loop"))
        end = total
        if loop:  # a looping stream lasts until its player stops (or the video ends)
            later = [f for f in stop_at.get(e["id"], []) if f > e["f"]]
            if later:
                end = min(end, int(min(later) / FPS * RATE))
        pos = e.get("from", 0.0) * RATE if loop else 0.0  # the music starts mid-song, as logged
        i = start
        while i < end:
            if pos >= n_src:
                if not loop:
                    break
                pos -= n_src
            j = int(pos)
            k = j + 1 if j + 1 < n_src else (0 if loop else j)
            t = pos - j
            o = 2 * i
            out[o] += (pcm[2 * j] * (1 - t) + pcm[2 * k] * t) * gl
            out[o + 1] += (pcm[2 * j + 1] * (1 - t) + pcm[2 * k + 1] * t) * gr
            pos += pitch
            i += 1
        placed += 1
    pcm_out = array("h", (int(max(-1.0, min(1.0, x)) * 32767) for x in out))
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm_out.tobytes())
    return placed
