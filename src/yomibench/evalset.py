"""A frozen evaluation set: game records copied once into a folder, with a manifest of their sha256s
(`.evalset.json`), and never changed afterwards.

    freeze(tars, out, games)                 # build it once; later calls only check it
    verify(out)                              # every listed record there with its sha256, nothing else
    check_disjoint(eval_files, train_files)  # refuse a seed or record sha256 shared with training

Round 3's trainer freezes one from gate games (tools/round3_loop.py --eval-from-gates) and passes it
to `yomibench.train --eval-dir`. Training verifies an eval folder that has a manifest and checks it
against every record in its training folders, so those games are never trained on (see
yomibench.train.select_files).

Selection is deterministic: every record in the given tars (flat `g<seed>-....jsonl.gz` members),
duplicates by sha256 dropped, sorted by sha256, the first `games`. Sorting by a hash spreads the
pick across gates, pairings and seats without a seed. Nothing is frozen until that many records
are available.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable

from yomibench import compat

MANIFEST = ".evalset.json"
RECORD = re.compile(r"^g(\d{12,})-[A-Za-z0-9._-]*\.jsonl\.gz$")


def record_seed(path: Path | str) -> int | None:
    """The seed in a record's file name (`g<12 or more digits>-...`; a gate game's seed is
    seed x 1,000,003 + n, 14 digits with --seed 20261116), None for another name."""
    m = RECORD.match(Path(path).name)
    return int(m.group(1)) if m else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _members(tf: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for m in tf.getmembers():
        if m.isfile() and Path(m.name).name == m.name and RECORD.match(m.name):  # flat records only, no paths
            yield m


def freeze(tars: list[Path], out: Path, games: int, exclude_seeds: Iterable[int] = ()) -> dict[str, Any] | None:
    """Freeze `games` records from `tars` into `out` (see the module docstring) and return the
    manifest. If `out` is already frozen, only verify it. If fewer than `games` records are
    available (seeds in `exclude_seeds` don't count), freeze nothing and return None."""
    out = Path(out)
    if (out / MANIFEST).exists():
        return verify(out)
    excluded = set(exclude_seeds)
    found: dict[str, tuple[str, str]] = {}  # record sha256 -> (member name, tar name)
    sources = {}
    for tar in sorted(Path(t) for t in tars):
        sources[tar.name] = file_sha256(tar)
        with tarfile.open(tar) as tf:
            for m in _members(tf):
                if record_seed(m.name) in excluded:
                    continue
                with tf.extractfile(m) as f:
                    data = f.read()
                if compat.held_out_gz(data):  # a human-play game is a held-out human test: never in a set
                    continue
                found.setdefault(hashlib.sha256(data).hexdigest(), (m.name, tar.name))
    if len(found) < games:
        return None
    picked = sorted(found)[:games]
    by_tar: dict[str, dict[str, str]] = {}
    for digest in picked:
        name, tar = found[digest]
        by_tar.setdefault(tar, {})[name] = digest
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("g*.jsonl.gz"):  # a freeze interrupted before its manifest
        stale.unlink()
    records = {}
    for tar in sorted(Path(t) for t in tars):
        want = by_tar.get(tar.name)
        if not want:
            continue
        with tarfile.open(tar) as tf:
            for m in _members(tf):
                if m.name in want:
                    with tf.extractfile(m) as f:
                        data = f.read()
                    if hashlib.sha256(data).hexdigest() != want[m.name]:
                        continue  # a same-named member with other content: the first one is listed
                    (out / m.name).write_bytes(data)
                    records[m.name] = {"sha256": want[m.name], "seed": record_seed(m.name), "source": tar.name}
    manifest = {"format": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "games": len(records), "sources": {k: v for k, v in sources.items() if k in by_tar},
                "records": dict(sorted(records.items()))}
    tmp = out / f"{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, out / MANIFEST)  # the manifest is written last: its presence means frozen
    return manifest


def verify(out: Path) -> dict[str, Any]:
    """The manifest of a frozen folder, after checking that the folder holds exactly its records
    with their sha256s. Raises ValueError otherwise."""
    out = Path(out)
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    listed = manifest["records"]
    present = {p.name for p in out.glob("g*.jsonl.gz")}
    problems = [f"{n} is missing" for n in sorted(set(listed) - present)]
    problems += [f"{n} is not in the manifest" for n in sorted(present - set(listed))]
    problems += [f"{n} has changed (sha256)" for n in sorted(set(listed) & present)
                 if file_sha256(out / n) != listed[n]["sha256"]]
    if problems:
        raise ValueError(f"frozen eval set {out} doesn't match its manifest: " + "; ".join(problems[:5])
                         + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))
    return manifest


def check_disjoint(eval_files: list[Path], train_files: list[Path]) -> None:
    """Refuse (ValueError) if any evaluation record shares a seed (from file names) or a sha256 with
    a training record. Only training files the same size as some evaluation file are hashed."""
    train_seeds: dict[int, Path] = {}
    for p in train_files:
        s = record_seed(p)
        if s is not None:
            train_seeds.setdefault(s, p)
    clashes = [f"{p.name} has the seed of training record {train_seeds[s].name}"
               for p in eval_files if (s := record_seed(p)) is not None and s in train_seeds]
    sizes: dict[int, list[Path]] = {}
    for p in eval_files:
        sizes.setdefault(p.stat().st_size, []).append(p)
    eval_sha: dict[Path, str] = {}
    for p in train_files:
        same_size = sizes.get(p.stat().st_size)
        if not same_size:
            continue
        digest = file_sha256(p)
        for q in same_size:
            if eval_sha.setdefault(q, file_sha256(q)) == digest:
                clashes.append(f"{q.name} is the same record as training record {p.name}")
    if clashes:
        raise ValueError("evaluation games overlap the training data: " + "; ".join(clashes[:5])
                         + (f" (and {len(clashes) - 5} more)" if len(clashes) > 5 else ""))
