import pytest

from slam.common.config import Config


def test_load_and_dotted_access(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("""
dataset:
  type: tum_rgbd
  root: /data/foo
camera:
  depth_scale: 5000.0
""")
    cfg = Config.load(cfg_file)
    assert cfg.dataset.type == "tum_rgbd"
    assert cfg.dataset.root == "/data/foo"
    assert cfg.camera.depth_scale == 5000.0


def test_require_passes_when_present(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("a:\n  b: 1\n")
    cfg = Config.load(cfg_file)
    cfg.require("a.b")  # should not raise


def test_require_raises_when_missing(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("a:\n  b: 1\n")
    cfg = Config.load(cfg_file)
    with pytest.raises(KeyError):
        cfg.require("a.c")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "does_not_exist.yaml")


def test_get_with_default(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("a: 1\n")
    cfg = Config.load(cfg_file)
    assert cfg.get("missing", "fallback") == "fallback"
