"""Configuration loading with explicit, side-effect-free semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping.

    Environment expansion and implicit includes are deliberately unsupported: a
    submitted specification must serialize to the same bytes on every machine.
    """

    source = path.expanduser().resolve(strict=True)
    suffix = source.suffix.lower()
    with source.open("r", encoding="utf-8") as stream:
        if suffix == ".json":
            value = json.load(stream)
        elif suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(stream)
        else:
            raise ValueError(f"unsupported configuration format: {source.suffix}")
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return value
