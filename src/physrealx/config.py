"""Configuration loading with explicit, side-effect-free semantics."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def _unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"configuration contains duplicate key: {key!r}")
        mapping[key] = value
    return mapping


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("configuration mapping keys must be scalar values") from error
        if duplicate:
            raise ValueError(f"configuration contains duplicate key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping.

    Environment expansion and implicit includes are deliberately unsupported: a
    submitted specification must serialize to the same bytes on every machine.
    """

    source = path.expanduser().resolve(strict=True)
    suffix = source.suffix.lower()
    try:
        with source.open("r", encoding="utf-8") as stream:
            if suffix == ".json":
                value = json.load(stream, object_pairs_hook=_unique_pairs)
            elif suffix in {".yaml", ".yml"}:
                value = yaml.load(stream, Loader=_UniqueKeyLoader)
            else:
                raise ValueError(f"unsupported configuration format: {source.suffix}")
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid configuration syntax: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return value
