"""Built distributions contain algorithm sources, not machine-local binaries or weights."""

import zipfile
from pathlib import Path

import pytest


def test_wheel_keeps_sources_and_excludes_native_build_products(tmp_path):
    wheel_module = pytest.importorskip("hatchling.builders.wheel")
    root = Path(__file__).resolve().parents[2]
    builder = wheel_module.WheelBuilder(str(root))
    filename = next(builder.build(directory=str(tmp_path), versions=["standard"]))
    with zipfile.ZipFile(tmp_path / filename) as archive:
        names = archive.namelist()
    for source in (
        "artgs/train.py",
        "dtart/main.py",
        "act/model.py",
        "dinov2/SOURCE.json",
        "dino_wm/SOURCE.json",
    ):
        assert "physrealx/_embedded/" + source in names
    assert "physrealx/platform.py" in names
    assert not any(
        n.endswith((".so", ".o", ".pyc", ".ckpt")) or "/weights/" in n or "/build/" in n
        for n in names
    )
