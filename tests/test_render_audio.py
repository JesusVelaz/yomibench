import json
import shutil
import wave
from array import array

import pytest

from yomibench import render_audio as ra

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def tone(path, seconds, value=8000):
    """A constant-level stereo 'sound' (a DC block), easy to find in the mix."""
    n = int(ra.RATE * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(ra.RATE)
        w.writeframes(array("h", [value] * (2 * n)).tobytes())


def read(path):
    with wave.open(str(path)) as w:
        pcm = array("h")
        pcm.frombytes(w.readframes(w.getnframes()))
    return pcm


def at(pcm, seconds, ch=0):
    return pcm[2 * int(seconds * ra.RATE) + ch]


def write_log(tmp_path, events):
    buses = {"Master": {"volume_db": 0, "mute": False, "send": ""},
             "Fx": {"volume_db": 0, "mute": False, "send": "Master"},
             "Music": {"volume_db": 0, "mute": False, "send": "Master"}}
    log = tmp_path / "x.sounds.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in [{"ev": "buses", "buses": buses}, *events]))
    return log


def test_sound_starts_on_its_frame_and_plays_to_its_end_despite_an_early_stop(tmp_path):
    (tmp_path / "s").mkdir()
    tone(tmp_path / "s" / "hit.wav", 0.5)
    log = write_log(tmp_path, [
        {"ev": "stream", "file": "hit.wav", "loop": False},
        {"ev": "start", "f": 60, "id": 1, "file": "hit.wav", "volume_db": 0, "pitch": 1, "bus": "Fx"},
        {"ev": "stop", "f": 62, "id": 1},  # the real-time mixer finished it early: not game time
    ])
    out = tmp_path / "out.wav"
    assert ra.mix(log, tmp_path / "s", out, 180) == 1
    pcm = read(out)
    assert len(pcm) == 2 * 3 * ra.RATE
    assert at(pcm, 0.99) == 0 and abs(at(pcm, 1.01) - 8000) <= 1 and abs(at(pcm, 1.45) - 8000) <= 1
    assert at(pcm, 1.55) == 0


def test_2d_sound_pans_and_music_is_off_by_default(tmp_path):
    (tmp_path / "s").mkdir()
    tone(tmp_path / "s" / "hit.wav", 0.2)
    tone(tmp_path / "s" / "song.wav", 0.3)
    log = write_log(tmp_path, [
        {"ev": "stream", "file": "hit.wav", "loop": False},
        {"ev": "stream", "file": "song.wav", "loop": True},
        {"ev": "start", "f": 0, "id": 9, "file": "song.wav", "volume_db": 0, "pitch": 1, "bus": "Music"},
        {"ev": "start", "f": 60, "id": 2, "file": "hit.wav", "volume_db": 0, "pitch": 1, "bus": "Fx",
         "dx": -320, "dy": 0, "max_distance": 2000, "attenuation": 1},  # far left edge of the screen
    ])
    out = tmp_path / "out.wav"
    ra.mix(log, tmp_path / "s", out, 120)
    pcm = read(out)
    assert at(pcm, 0.5) == 0 and at(pcm, 0.5, 1) == 0  # no music
    assert at(pcm, 1.05) > 6000 and at(pcm, 1.05, 1) == 0  # all left, attenuated by distance


def test_music_loops_until_its_player_stops_when_enabled(tmp_path):
    (tmp_path / "s").mkdir()
    tone(tmp_path / "s" / "song.wav", 0.3)
    log = write_log(tmp_path, [
        {"ev": "stream", "file": "song.wav", "loop": True},
        {"ev": "start", "f": 0, "id": 9, "file": "song.wav", "volume_db": -6, "pitch": 1, "bus": "Music"},
        {"ev": "stop", "f": 90, "id": 9},
    ])
    out = tmp_path / "out.wav"
    ra.mix(log, tmp_path / "s", out, 120, music=True)
    pcm = read(out)
    assert abs(at(pcm, 1.2) - 8000 * 10 ** (-6 / 20)) < 30  # still looping past the 0.3 s file
    assert at(pcm, 1.6) == 0  # stopped at frame 90
