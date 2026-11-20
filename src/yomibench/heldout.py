"""A frozen held-out split per record folder: games validated once are never trained on later.

Each folder gets a manifest, `<folder>/.heldout.json` (or, with a root, `<root>/<name>-<hash of
the folder's path>.heldout.json` for folders that must stay read-only), listing which game files
are held out and which train:

    {"format": 1, "seed": 0, "fraction": 0.15, "heldout": [names], "training": [names],
     "created_utc": ..., "updated_utc": ...}

A game file is assigned once, when a run first sees it, by a hash of its name and the manifest's
seed (held out if the hash falls under the manifest's fraction), and is never reassigned: the
seed and fraction are those the manifest was created with, whatever a later run asks for. Games
added later are assigned the same way and appended. If a folder's first assignment holds out
nothing although it has at least two games, the game with the lowest hash is held out, so a
small folder still validates. Files listed but no longer present are kept in the manifest (they
stay out of training if they come back). No torch here: training's record workers import it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

FORMAT = 1
NAME = ".heldout.json"
PATTERN = "g*.jsonl.gz"


def manifest_path(folder: Path, root: Path | None = None) -> Path:
    folder = Path(folder)
    if root is None:
        return folder / NAME
    tag = hashlib.sha1(str(folder.resolve()).lower().encode()).hexdigest()[:10]
    return Path(root) / f"{folder.resolve().name}-{tag}{NAME}"


def _bucket(name: str, seed: int) -> float:
    """A fixed pseudo-random number in [0, 1) for a game file name."""
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:15], 16) / 16 ** 15


def update(folder: Path, fraction: float = 0.15, seed: int = 0, root: Path | None = None) -> dict[str, Any]:
    """Load (or create) `folder`'s manifest and assign the game files it doesn't list yet; written
    back only if something changed. Returns the manifest plus `path`, `sha256` (of the file as
    written) and `added` (files assigned by this call)."""
    folder = Path(folder)
    path = manifest_path(folder, root)
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("format") != FORMAT:
            raise SystemExit(f"{path}: unknown held-out manifest format {m.get('format')!r}")
    else:
        if not 0 < fraction < 1:
            raise SystemExit(f"held-out fraction must be between 0 and 1, not {fraction}")
        m = {"format": FORMAT, "seed": seed, "fraction": fraction, "heldout": [], "training": [],
             "created_utc": _now(), "updated_utc": None}
    known = set(m["heldout"]) | set(m["training"])
    new = sorted(p.name for p in folder.glob(PATTERN) if p.name not in known)
    if new:
        held = [n for n in new if _bucket(n, m["seed"]) < m["fraction"]]
        if not m["heldout"] and not held and len(known) + len(new) >= 2:
            held = [min(new, key=lambda n: _bucket(n, m["seed"]))]
        m["heldout"] = sorted(set(m["heldout"]) | set(held))
        m["training"] = sorted(set(m["training"]) | (set(new) - set(held)))
        m["updated_utc"] = _now()
        body = json.dumps(m, indent=1, sort_keys=True) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            raise SystemExit(f"can't write the held-out manifest {path} ({e}); pass --heldout-root DIR to keep "
                             "manifests outside read-only record folders") from e
    body = path.read_bytes() if path.exists() else b""
    return {**m, "path": str(path), "sha256": hashlib.sha256(body).hexdigest() if body else None, "added": len(new)}


def split(folder: Path, fraction: float = 0.15, seed: int = 0,
          root: Path | None = None) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """(training files, held-out files, manifest info) for the game files now in `folder`."""
    m = update(folder, fraction, seed, root)
    held = set(m["heldout"])
    files = sorted(Path(folder).glob(PATTERN))
    info = {"manifest": m["path"], "sha256": m["sha256"], "seed": m["seed"], "fraction": m["fraction"],
            "added": m["added"], "heldout": sum(p.name in held for p in files),
            "training": sum(p.name not in held for p in files)}
    return [p for p in files if p.name not in held], [p for p in files if p.name in held], info


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
