"""Check an Arena submission (phase 1: data only, no code): a model bundle plus an entry.

    python tools/validate_submission.py ENTRY.json BUNDLE.zip [--json]
    python tools/validate_submission.py --issue-body BODY.md --bundle BUNDLE.zip --issue-author LOGIN [--json]
    python tools/validate_submission.py --issue-body BODY.md --bundle-url      # print the bundle's link
    python tools/validate_submission.py --render-comment RESULT.json            # the issue comment, made safe

A submission is a bundle (`yomibench bundle pack`: NAME/bundle.json plus value.npz and optionally policy.npz,
arrays only) and an entry: who it is, the candidate set its nets were trained on, and search settings for our
existing search. The Arena fixes the match settings and the per-move thinking budget for every bot
(arena/spec.json); an entry's settings may only narrow what its bot searches inside that budget, never widen it.

Checks, in order, each bounded before anything is allocated:
- the zip: at most 25 MB, exactly NAME/bundle.json plus value.npz and/or policy.npz, no duplicate or other entries,
  at most 64 MB unpacked (counted while unpacking, not trusted from the zip's headers);
- each net file: at most 256 arrays and 64 MB unpacked, numeric arrays only (plus the metadata string), shapes read
  from the array headers before any array is loaded;
- the bundle's manifest, file hashes and loading (yomibench.ladder), with allow_pickle=False;
- every weight finite, and one forward pass of the value net on a zero position of the right width, so a net that
  is missing a tensor or has mismatched shapes can't pass;
- the nets' features version and the entry's candidate set are the ones arena/spec.json pins;
- no net is one of HustleBot's own (its arrays' hash isn't in the house list, published or not);
- the entry's schema (allowed keys only, types and ranges); its name isn't reserved (house generations and their
  page names, built-in bots, device names) or already entered with other files (arena/bots.json); on an issue, the
  owner is the issue's author.

Exit 0 when the submission is valid, 1 with the reasons when it isn't. Nothing in a submission is ever run: a net is
arrays, and settings are numbers and names from a fixed list. Messages quote submitted text only in a bounded,
quoted form, and --render-comment strips anything that could break out of the comment's code block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BUNDLE = 25 * 1024 * 1024    # GitHub's attachment limit for a zip; today's nets are well under 1 MB
MAX_UNPACKED = 64 * 1024 * 1024  # the bundle's files together, unpacked
MAX_ARRAYS = 256                 # per net file
MAX_NET = 64 * 1024 * 1024       # one net file's arrays, unpacked
NAME = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,38}[a-z0-9])?$")
RESERVED = re.compile(r"^(?:g|gen)[-_.]?\d+$"                                   # house generations and their pages
                      r"|^(?:house|yomibench|random|matrix|matrix-do|policy)(?:[-_.].*)?$"  # house and built-in bots
                      r"|^(?:con|prn|aux|nul|com\d|lpt\d)(?:\..*)?$")          # names Windows can't store
OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")         # a GitHub login
ENTRY_KEYS = {"name", "owner", "bundle_sha256", "candidate_set", "settings", "notes"}
SPEC = ROOT / "arena" / "spec.json"  # the Arena's pinned versions: candidate set, net features
BOTS = ROOT / "arena" / "bots.json"  # bots already entered
# Search settings a phase-1 entry may set, each of which can only narrow the Arena's fixed search:
# a policy net shortlists moves before the value net's search (yomibench.bots.PolicyPrune).
SETTINGS = {
    "prune_k": (int, 4, 32),                 # keep the policy's top K candidates
    "prune_scope": (str, ("own", "both")),   # prune only this bot's own moves, or both seats' in its search
}
NEEDS = {"prune_k": "policy.npz", "prune_scope": "policy.npz"}


def q(text, limit: int = 60) -> str:
    """Submitted text, quoted and bounded for messages."""
    s = str(text)
    return repr(s[:limit] + ("…" if len(s) > limit else ""))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def arena_spec() -> dict | None:
    return _json(SPEC)


def check_entry(entry: object, author: str | None = None, entered: list | None = None) -> list[str]:
    """Schema problems in an entry (empty when it's well formed)."""
    if not isinstance(entry, dict):
        return ["the entry must be a JSON object"]
    bad = []
    extra = set(entry) - ENTRY_KEYS
    if extra:
        bad.append(f"unknown entry keys {', '.join(q(k, 30) for k in sorted(map(str, extra))[:5])} "
                   f"(allowed: {', '.join(sorted(ENTRY_KEYS))})")
    name, owner = entry.get("name"), entry.get("owner")
    if not isinstance(name, str) or not NAME.match(name):
        bad.append("name: 1-40 of a-z 0-9 . _ -, starting and ending with a letter or digit")
    elif RESERVED.match(name):
        bad.append(f"name: {q(name)} is reserved (house generations, built-in bots and device names)")
    if not isinstance(owner, str) or not OWNER.match(owner):
        bad.append("owner: your GitHub login")
    elif author is not None and owner.casefold() != author.casefold():
        bad.append(f"owner: {q(owner)} isn't this issue's author ({q(author)}); a bot is entered by its owner")
    sha = entry.get("bundle_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        bad.append("bundle_sha256: the bundle zip's sha256, 64 hex digits")
    spec = arena_spec()
    cset = entry.get("candidate_set")
    if type(cset) is not int:
        bad.append("candidate_set: the candidate set your nets were trained on (a whole number)")
    elif spec and cset != spec["candidate_set"]:
        bad.append(f"candidate_set: the Arena plays candidate set {spec['candidate_set']}; nets trained on set {cset} "
                   f"aren't compatible")
    notes = entry.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 500:
        bad.append("notes: text, at most 500 characters")
    settings = entry.get("settings", {})
    if not isinstance(settings, dict):
        return bad + ["settings: a JSON object"]
    for k, v in settings.items():
        rule = SETTINGS.get(k)
        if rule is None:
            bad.append(f"settings: {q(k, 30)} isn't a phase-1 setting (allowed: {', '.join(sorted(SETTINGS))})")
        elif rule[0] is int and not (type(v) is int and rule[1] <= v <= rule[2]):
            bad.append(f"settings.{k}: a whole number from {rule[1]} to {rule[2]}")
        elif rule[0] is str and v not in rule[1]:
            bad.append(f"settings.{k}: one of {', '.join(rule[1])}")
    if "prune_scope" in settings and "prune_k" not in settings:
        bad.append("settings.prune_scope needs settings.prune_k")
    for e in entered or []:
        if isinstance(name, str) and e.get("name") == name:
            bad.append(f"name: {q(name)} is already entered; a new version needs a new name (like {name}-2)")
    return bad


def unpack(bundle: Path, into: Path) -> tuple[Path | None, list[str]]:
    """Unpacks a bundle zip into `into` with every limit enforced while reading; (its folder, problems)."""
    try:
        z = zipfile.ZipFile(bundle)
    except (zipfile.BadZipFile, OSError) as e:
        return None, [f"bundle: not a zip file ({type(e).__name__})"]
    with z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(infos) > 4 or len(set(names)) != len(names):
            return None, ["bundle: a bundle holds one NAME/ folder with bundle.json and its net files, once each"]
        tops = {n.split("/", 1)[0] for n in names}
        top = next(iter(tops)) if len(tops) == 1 else None
        allowed = {f"{top}/", f"{top}/bundle.json", f"{top}/value.npz", f"{top}/policy.npz"}
        if top is None or not NAME.match(top) or set(names) - allowed or f"{top}/bundle.json" not in names:
            return None, [f"bundle: unexpected entries {', '.join(q(n, 40) for n in sorted(set(names) - allowed)[:5])}"
                          if top else "bundle: a bundle holds exactly one NAME/ folder"]
        if any(i.flag_bits & 0x1 for i in infos):
            return None, ["bundle: encrypted entries aren't allowed"]
        d = into / top
        d.mkdir(parents=True)
        total = 0
        for i in infos:
            if i.filename.endswith("/"):
                continue
            with z.open(i) as src, (d / i.filename.split("/", 1)[1]).open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UNPACKED:
                        return None, [f"bundle: more than {MAX_UNPACKED >> 20} MB unpacked"]
                    dst.write(chunk)
    return d, []


def check_net_file(path: Path) -> list[str]:
    """Bounds on one .npz before numpy loads it: array count, dtypes and unpacked size from the array headers."""
    import numpy as np
    from numpy.lib import format as npf

    name = path.name
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return [f"{name}: not a numpy .npz file"]
    with z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(infos) > MAX_ARRAYS or len(set(names)) != len(names) or not all(n.endswith(".npy") for n in names):
            return [f"{name}: at most {MAX_ARRAYS} uniquely named arrays, nothing else"]
        total = 0
        for i in infos:
            try:
                with z.open(i) as f:
                    version = npf.read_magic(f)
                    shape, _, dtype = (npf.read_array_header_1_0(f) if version == (1, 0)
                                       else npf.read_array_header_2_0(f))
            except Exception:
                return [f"{name}: {q(i.filename, 40)} isn't a numpy array"]
            if dtype.hasobject or dtype.kind not in "fiubU":
                return [f"{name}: {q(i.filename, 40)} has dtype {dtype}; nets hold numbers only"]
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize if shape else dtype.itemsize
            total += size
            if total > MAX_NET:
                return [f"{name}: more than {MAX_NET >> 20} MB of arrays"]
    return []


def check_value_net(path: Path) -> list[str]:
    """Every weight finite and one forward pass on a zero position: the net can score a position."""
    import numpy as np

    from yomibench.valuenet import ENCODERS, ValueNet, forward

    try:
        net = ValueNet.load(path)
        enc = ENCODERS[net.version]
        if any(a.dtype.kind == "f" and not np.isfinite(a).all() for a in net.p.values()):
            return ["value.npz: a weight is NaN or infinite"]
        nums = np.zeros((1, enc.N_NUMERIC), np.float32)
        cats = np.zeros((1, len(enc.CATEGORY_OF_SLOT)), np.int64)
        out = forward(net.p, nums, cats, enc.CATEGORY_OF_SLOT)
    except Exception as e:  # missing tensors, mismatched shapes, bad metadata
        return [f"value.npz: the net can't score a position ({type(e).__name__}: {q(e, 120)})"]
    if out.shape != (1,) or not np.isfinite(out).all():
        return ["value.npz: the net's output isn't one finite value per position"]
    return []


def array_hash(path: Path) -> str:
    """A net's identity by its arrays alone: sha256 over every array but `meta`, in name order (name, dtype, shape,
    bytes). Repacking the file or changing its metadata doesn't change it; a house model can't pass as an entry."""
    import numpy as np

    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(n for n in z.files if n != "meta"):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def load_house_hashes(path: Path | None) -> set[str]:
    """The house models' array hashes (a list of hashes only, published as the `house-hashes` release's asset)."""
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {h for h in data.get("hashes", []) if isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)}


def check_policy_net(path: Path) -> list[str]:
    import numpy as np

    from yomibench.policy import PolicyNet

    try:
        net = PolicyNet.load(path)
    except Exception as e:
        return [f"policy.npz: doesn't load ({type(e).__name__}: {q(e, 120)})"]
    if any(a.dtype.kind == "f" and not np.isfinite(a).all() for a in net.p.values()):
        return ["policy.npz: a weight is NaN or infinite"]
    return []


def validate(entry: object, bundle: Path, author: str | None = None,
             house: set[str] | None = None) -> tuple[list[str], dict | None]:
    """(problems, the bundle's manifest when it checked out)."""
    from yomibench.ladder import LadderError, _verify_dir

    bad = check_entry(entry, author, _json(BOTS) or [])
    if not bundle.is_file():
        return bad + ["no bundle file"], None
    size = bundle.stat().st_size
    if size > MAX_BUNDLE:
        return bad + [f"the bundle is {size >> 20} MB; the limit is {MAX_BUNDLE >> 20} MB"], None
    tmp = Path(tempfile.mkdtemp(prefix="yomibench-submission-"))
    try:
        d, problems = unpack(bundle, tmp)
        if problems:
            return bad + problems, None
        for f in ("value.npz", "policy.npz"):
            if (d / f).exists():
                problems += check_net_file(d / f)
        if problems:
            return bad + problems, None
        try:
            manifest = _verify_dir(d)
        except (LadderError, ValueError, KeyError, TypeError) as e:
            return bad + [f"bundle: {q(e, 160)}"], None
        if "value.npz" not in manifest["files"]:
            return bad + ["the Arena's search needs a value net (value.npz) in the bundle"], manifest
        bad += check_value_net(d / "value.npz")
        for f in ("value.npz", "policy.npz"):  # HustleBot's own nets, published or not, can't be entered
            if house and (d / f).exists() and array_hash(d / f) in house:
                bad.append(f"{f}: these are HustleBot's weights (a house model); submit nets you trained")
        if "policy.npz" in manifest["files"]:
            bad += check_policy_net(d / "policy.npz")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if isinstance(entry, dict):
        if entry.get("name") and manifest["name"] != entry["name"]:
            bad.append(f"the entry's name {q(entry['name'])} isn't the bundle's ({q(manifest['name'])})")
        sha = entry.get("bundle_sha256")
        if isinstance(sha, str) and _sha256(bundle) != sha.lower():
            bad.append("bundle_sha256 doesn't match the bundle zip")
        spec = arena_spec()
        for f, info in manifest["files"].items():
            if spec and info.get("features") not in spec["features"]:
                bad.append(f"{f}: features version {q(info.get('features'), 10)}; the Arena plays {spec['features']}")
        for k in (entry.get("settings") or {}) if isinstance(entry.get("settings"), dict) else ():
            if k in NEEDS and NEEDS[k] not in manifest["files"]:
                bad.append(f"settings.{k} needs {NEEDS[k]} in the bundle")
    return bad, manifest


def entry_from_issue(body: str) -> object:
    """The entry JSON from an issue form's body: the fenced block under the "Entry" heading."""
    m = re.search(r"###\s*Entry\s*\n+```(?:json)?\s*\n(.*?)\n```", body, re.S) or \
        re.search(r"###\s*Entry\s*\n+(\{.*?\})\s*(?:\n###|\Z)", body, re.S)
    if not m:
        raise ValueError("no entry JSON under the form's Entry field")
    if len(m.group(1)) > 4000:
        raise ValueError("the entry is longer than 4000 characters")
    return json.loads(m.group(1))


# Where an issue's bundle may come from: a file attached to the issue, or a release asset on GitHub.
BUNDLE_URL = re.compile(r"https://github\.com/(?:user-attachments/files/\d+/[\w.-]+\.zip"
                        r"|[\w.-]+/[\w.-]+/releases/download/[\w.%+-]+/[\w.-]+\.zip)")


def bundle_url_from_issue(body: str) -> str:
    """The bundle link under the form's "Bundle" heading (GitHub attachments and release assets only)."""
    m = re.search(r"###\s*Bundle\s*\n(.*?)(?:\n###|\Z)", body, re.S)
    found = BUNDLE_URL.findall(m.group(1)) if m else []
    if len(found) != 1:
        raise ValueError("the Bundle field needs exactly one zip attached to the issue or one GitHub release asset link")
    return found[0]


def render_comment(result: dict) -> str:
    """The issue comment for a check's result. Every problem sits in a code block with backticks removed, so
    submitted text can't close the block, mention anyone or add links."""
    problems = [re.sub(r"[`\r\x00-\x08\x0b-\x1f]", "'", str(p))[:300] for p in result.get("problems", [])[:20]]
    if result.get("valid"):
        b = result.get("bundle") or {}
        name = re.sub(r"[^a-z0-9._-]", "", str(b.get("name", "")))[:40]
        return (f"**Checks passed** for `{name}` (digest `{str(b.get('digest', ''))[:12]}`).\n\n"
                "Rated matches open soon: valid bots will be played then, and their results appear on the site.\n")
    return ("**Checks failed.** Edit the issue to fix these, and the checks run again.\n\n```text\n"
            + "\n".join(f"- {p}" for p in problems) + "\n```\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("entry", nargs="?", type=Path)
    ap.add_argument("bundle_pos", nargs="?", type=Path, metavar="bundle")
    ap.add_argument("--issue-body", type=Path, help="read the entry from an issue form's body instead")
    ap.add_argument("--issue-author", help="the issue's author: the entry's owner must be this login")
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--bundle-url", action="store_true", help="with --issue-body: print the bundle's link and stop")
    ap.add_argument("--render-comment", type=Path, metavar="RESULT.json", help="print the issue comment for a result")
    ap.add_argument("--house-hashes", type=Path, help="the house models' array hashes (house-hashes.json)")
    a = ap.parse_args(argv)
    if a.render_comment:
        print(render_comment(json.loads(a.render_comment.read_text(encoding="utf-8"))), end="")
        return 0
    if a.bundle_url:
        try:
            print(bundle_url_from_issue(a.issue_body.read_text(encoding="utf-8")))
            return 0
        except (OSError, ValueError, AttributeError) as e:
            print(f"not valid:\n- {e}")
            return 1
    bundle = a.bundle or a.bundle_pos
    body_sha = None
    try:
        if a.issue_body:
            body = a.issue_body.read_text(encoding="utf-8")
            body_sha = hashlib.sha256(body.rstrip("\n").encode()).hexdigest()  # the report step compares the same
            entry = entry_from_issue(body)
        elif a.entry:
            entry = json.loads(a.entry.read_text(encoding="utf-8"))
        else:
            ap.error("give ENTRY.json or --issue-body")
    except (OSError, ValueError) as e:
        entry, bad, manifest = None, [f"entry: {q(e, 160)}"], None
    else:
        bad, manifest = (validate(entry, bundle, a.issue_author, load_house_hashes(a.house_hashes)) if bundle else
                         (check_entry(entry, a.issue_author, _json(BOTS) or []) + ["no bundle given"], None))
    out = {"valid": not bad, "problems": bad, "body_sha256": body_sha,
           "bundle": {k: manifest.get(k) for k in ("name", "digest")} if manifest else None}
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        print("valid" if not bad else "not valid:\n" + "\n".join(f"- {b}" for b in bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
