import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from yomibench import ladder

from test_ladder import nets  # noqa: F401  (the toy value and policy nets fixture)

spec = importlib.util.spec_from_file_location("validate_submission",
                                              Path(__file__).resolve().parents[1] / "tools" / "validate_submission.py")
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)

GOOD = {"name": "ours-v1", "owner": "Someone", "bundle_sha256": "0" * 64, "candidate_set": 6,
        "settings": {"prune_k": 8}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_the_entry_schema_allows_only_narrowing_settings_and_real_owners():
    assert vs.check_entry(GOOD) == []
    assert vs.check_entry({**GOOD, "settings": {"prune_k": 8, "prune_scope": "both"}}) == []
    assert any("isn't a phase-1 setting" in b for b in vs.check_entry({**GOOD, "settings": {"iterations": 10_000}}))
    assert any("prune_k" in b for b in vs.check_entry({**GOOD, "settings": {"prune_k": 500}}))
    assert any("prune_k" in b for b in vs.check_entry({**GOOD, "settings": {"prune_k": True}}))
    for reserved in ("gen-14", "g13", "gen13", "random", "matrix-do", "house-v2", "con", "nul.x"):
        assert any("reserved" in b for b in vs.check_entry({**GOOD, "name": reserved})), reserved
    assert any("name:" in b for b in vs.check_entry({**GOOD, "name": "trailing."}))
    assert any("unknown entry keys" in b for b in vs.check_entry({**GOOD, "code": "import os"}))
    assert vs.check_entry([1, 2]) == ["the entry must be a JSON object"]
    assert any("candidate set 6" in b for b in vs.check_entry({**GOOD, "candidate_set": 5}))
    assert any("candidate_set" in b for b in vs.check_entry({k: v for k, v in GOOD.items() if k != "candidate_set"}))
    assert vs.check_entry(GOOD, author="someone") == []  # GitHub logins compare case-insensitively
    assert any("isn't this issue's author" in b for b in vs.check_entry(GOOD, author="mallory"))
    assert any("already entered" in b for b in vs.check_entry(GOOD, entered=[{"name": "ours-v1"}]))
    assert vs.check_entry({**GOOD, "bundle_sha256": "A" * 64}) == []  # Get-FileHash prints uppercase


def test_submitted_text_cant_escape_the_comment():
    evil = "x\n```\n@everyone [click](https://evil.example) ::error::pwned"
    problems = vs.check_entry({**GOOD, "settings": {evil: 1}})
    comment = vs.render_comment({"valid": False, "problems": problems})
    body = comment.split("```text\n", 1)[1]
    assert body.count("```") == 1 and comment.count("```") == 2  # only the block's own fences
    assert "\n::error" not in comment and "\n@" not in comment
    ok = vs.render_comment({"valid": True, "bundle": {"name": "ours-v1`<b>", "digest": "ab" * 32}})
    assert "`ours-v1b`" in ok and "open soon" in ok


def test_the_entry_is_read_from_an_issue_form_body():
    body = ("### Bot name\n\nours-v1\n\n### Entry\n\n```json\n" + json.dumps(GOOD) + "\n```\n\n### Bundle\n\n"
            "https://github.com/user-attachments/files/1/ours-v1.zip\n")
    assert vs.entry_from_issue(body) == GOOD
    assert vs.bundle_url_from_issue(body) == "https://github.com/user-attachments/files/1/ours-v1.zip"
    rel = body.replace("https://github.com/user-attachments/files/1/ours-v1.zip",
                       "https://github.com/someone/bots/releases/download/v1/ours-v1.zip")
    assert vs.bundle_url_from_issue(rel).endswith("/releases/download/v1/ours-v1.zip")
    for elsewhere in ("https://example.com/ours-v1.zip", "https://github.com/user-attachments/files/1/x.exe"):
        with pytest.raises(ValueError):
            vs.bundle_url_from_issue(body.replace("https://github.com/user-attachments/files/1/ours-v1.zip", elsewhere))


def test_a_real_bundle_validates_and_mismatches_are_refused(nets, tmp_path):  # noqa: F811
    value, policy = nets
    zip_ = tmp_path / "ours-v1.zip"
    ladder.pack("ours-v1", "Someone", value, policy, zip_)
    entry = {**GOOD, "bundle_sha256": sha(zip_)}
    bad, manifest = vs.validate(entry, zip_)
    assert bad == [] and manifest["name"] == "ours-v1"
    bad, _ = vs.validate({**entry, "name": "other-v1"}, zip_)
    assert any("isn't the bundle's" in b for b in bad)
    bad, _ = vs.validate({**entry, "bundle_sha256": "1" * 64}, zip_)
    assert any("doesn't match" in b for b in bad)
    # pruning needs a policy net; the Arena's search needs a value net
    vonly = tmp_path / "v.zip"
    ladder.pack("ours-v1", "Someone", value, None, vonly)
    bad, _ = vs.validate({**entry, "bundle_sha256": sha(vonly)}, vonly)
    assert bad == ["settings.prune_k needs policy.npz in the bundle"]
    ponly = tmp_path / "p.zip"
    ladder.pack("ours-v1", "Someone", None, policy, ponly)
    bad, _ = vs.validate({**entry, "settings": {}, "bundle_sha256": sha(ponly)}, ponly)
    assert bad == ["the Arena's search needs a value net (value.npz) in the bundle"]
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    bad, manifest = vs.validate(entry, junk)
    assert manifest is None and any(b.startswith("bundle:") for b in bad)


def test_bundles_are_bounded_and_nets_must_be_able_to_play(nets, tmp_path):  # noqa: F811
    import numpy as np

    value, _ = nets
    # extra or duplicate entries
    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(extra, "w") as z:
        z.writestr("ours-v1/bundle.json", "{}")
        z.writestr("ours-v1/run.py", "print(1)")
    assert any("unexpected entries" in b for b in vs.validate({**GOOD, "settings": {}}, extra)[0])
    # a net with a tensor missing loads, but can't score a position
    with np.load(value, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    broken = tmp_path / "broken.npz"
    np.savez(broken, **{k: v for k, v in arrays.items() if k != "w2"})
    zb = tmp_path / "broken.zip"
    ladder.pack("ours-v1", "Someone", broken, None, zb)
    bad, _ = vs.validate({**GOOD, "settings": {}, "bundle_sha256": sha(zb)}, zb)
    assert any("can't score a position" in b for b in bad)
    # a NaN weight
    nan = tmp_path / "nan.npz"
    np.savez(nan, **{k: (np.full_like(v, np.nan) if k == "w1" else v) for k, v in arrays.items()})
    zn = tmp_path / "nan.zip"
    ladder.pack("ours-v1", "Someone", nan, None, zn)
    bad, _ = vs.validate({**GOOD, "settings": {}, "bundle_sha256": sha(zn)}, zn)
    assert any("NaN or infinite" in b for b in bad)
    # an array whose header claims more than the limit is refused before it's loaded
    big = tmp_path / "big.npz"
    with zipfile.ZipFile(big, "w") as z:
        header = np.lib.format.header_data_from_array_1_0(np.zeros(1, np.float32))
        header["shape"] = (vs.MAX_NET,)
        import io
        buf = io.BytesIO()
        np.lib.format.write_array_header_1_0(buf, header)
        z.writestr("w1.npy", buf.getvalue())
    assert any("MB of arrays" in b for b in vs.check_net_file(big))


def test_a_house_models_weights_are_refused_whatever_its_metadata(nets, tmp_path):  # noqa: F811
    import numpy as np

    value, _ = nets
    house = tmp_path / "house-hashes.json"
    house.write_text(json.dumps({"hashes": [vs.array_hash(value)]}))
    # the same arrays with other metadata (a "trimmed" copy) still match
    with np.load(value, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads(str(arrays.pop("meta")))
    meta["note"] = "repacked"
    copy = tmp_path / "copy.npz"
    np.savez(copy, meta=np.array(json.dumps(meta)), **arrays)
    assert vs.array_hash(copy) == vs.array_hash(value)
    z = tmp_path / "ours-v1.zip"
    ladder.pack("ours-v1", "Someone", copy, None, z)
    entry = {**GOOD, "settings": {}, "bundle_sha256": sha(z)}
    bad, _ = vs.validate(entry, z, house=vs.load_house_hashes(house))
    assert any("HustleBot's weights" in b for b in bad)
    assert vs.validate(entry, z, house=set())[0] == []  # without the list, the same bundle is fine
    other = tmp_path / "other.npz"  # one weight changed: a different net
    np.savez(other, meta=np.array(json.dumps(meta)), **{k: (v + 1e-3 if k == "w1" else v) for k, v in arrays.items()})
    assert vs.array_hash(other) != vs.array_hash(value)
