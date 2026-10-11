"""Config loading for SLAM pipelines.

A config file is plain YAML. We don't force a rigid schema (different SLAM
backends/adapters need different fields) — we just load it into a dict-like
`Config` object with dotted attribute access, and let each module validate
the keys it actually needs via `Config.require(...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a nested dict with attribute-style access.

    Example:
        cfg = Config.load("configs/visual_slam.yaml")
        cfg.dataset.root
        cfg.camera.depth_scale
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def require(self, *keys: str) -> None:
        """Raise a clear error if any dotted key path is missing.

        Example: cfg.require("dataset.root", "camera.intrinsics_file")
        """
        missing = [k for k in keys if self._get(k) is None]
        if missing:
            raise KeyError(f"Config is missing required key(s): {missing}")

    def _get(self, dotted_key: str) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            raise AttributeError(f"Config has no key '{name}'")
        value = self._data[name]
        return Config(value) if isinstance(value, dict) else value

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"
