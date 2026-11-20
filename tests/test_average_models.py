"""tools/average_models.py: weight averaging of one lineage's fits, embedding rows matched by name."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("average_models", Path(__file__).resolve().parents[1] / "tools" / "average_models.py")
avg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avg)


def _model(path, vocab, emb, w1, mean=None, extra=None, meta_extra=None):
    meta = {"name": path.stem, "features": 3, "raw_defaults": {}, "vocab": {"state": vocab}, **(meta_extra or {})}
    params = {"emb_state": emb, "w1": w1, "b1": np.zeros(2, np.float32), "w3": np.ones((2, 1), np.float32),
              "b3": np.zeros(1, np.float32), "mean": np.zeros(3, np.float32) if mean is None else mean,
              "std": np.ones(3, np.float32), **(extra or {})}
    np.savez(path, meta=np.array(json.dumps(meta)), **params)
    return path


def test_a_model_averaged_with_a_renumbered_copy_of_itself_is_itself(tmp_path):
    emb = np.array([[0, 0], [1, 2], [3, 4], [5, 6]], np.float32)  # row 0 unknown; Wait, Jump, Dash
    w1 = np.arange(6, dtype=np.float32).reshape(3, 2)
    a = _model(tmp_path / "a.npz", {"Wait": 1, "Jump": 2, "Dash": 3}, emb, w1)
    # the same weights, but this fit numbered its vocabulary differently
    b = _model(tmp_path / "b.npz", {"Dash": 1, "Wait": 2, "Jump": 3}, emb[[0, 3, 1, 2]], w1)
    params, meta = avg.average([a, b])
    np.testing.assert_allclose(params["emb_state"], emb)
    np.testing.assert_allclose(params["w1"], w1)
    assert meta["vocab"]["state"] == {"Wait": 1, "Jump": 2, "Dash": 3}
    assert [s["name"] for s in meta["averaged_from"]] == ["a", "b"] and "val_mse" not in meta


def test_weights_are_the_mean(tmp_path):
    v = {"Wait": 1}
    e1, e2 = np.array([[0, 0], [2, 2]], np.float32), np.array([[0, 0], [4, 0]], np.float32)
    a = _model(tmp_path / "a.npz", v, e1, np.ones((3, 2), np.float32))
    b = _model(tmp_path / "b.npz", v, e2, 3 * np.ones((3, 2), np.float32))
    params, _ = avg.average([a, b])
    np.testing.assert_allclose(params["w1"], 2 * np.ones((3, 2)))
    np.testing.assert_allclose(params["emb_state"], [[0, 0], [3, 1]])


def test_refuses_what_is_not_one_lineage(tmp_path):
    v = {"Wait": 1}
    e = np.zeros((2, 2), np.float32)
    a = _model(tmp_path / "a.npz", v, e, np.ones((3, 2), np.float32))
    other_std = _model(tmp_path / "b.npz", v, e, np.ones((3, 2), np.float32), mean=np.ones(3, np.float32))
    with pytest.raises(ValueError, match="standardization"):
        avg.average([a, other_std])
    wider = _model(tmp_path / "c.npz", v, e, np.ones((3, 3), np.float32))
    with pytest.raises(ValueError, match="shapes"):
        avg.average([a, wider])
    with pytest.raises(ValueError, match="at least two"):
        avg.average([a])
    assert avg.main([str(tmp_path / "out.npz"), str(a), str(wider)]) == 2 and not (tmp_path / "out.npz").exists()


def test_side_heads_are_kept_only_when_every_model_has_them(tmp_path):
    v = {"Wait": 1}
    e = np.zeros((2, 2), np.float32)
    head = {"aux_search_w": np.ones((2, 3), np.float32), "aux_search_b": np.zeros(3, np.float32)}
    a = _model(tmp_path / "a.npz", v, e, np.ones((3, 2), np.float32), extra=head, meta_extra={"aux_search_heads": [1, 4, 16]})
    b = _model(tmp_path / "b.npz", v, e, np.ones((3, 2), np.float32))
    params, meta = avg.average([a, b])
    assert "aux_search_w" not in params and "aux_search_heads" not in meta
    c = _model(tmp_path / "c.npz", v, e, np.ones((3, 2), np.float32), extra=head)
    params, _ = avg.average([a, c])
    assert "aux_search_w" in params
    assert avg.main([str(tmp_path / "out.npz"), str(a), str(c)]) == 0 and (tmp_path / "out.npz").exists()
