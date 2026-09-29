"""Built distributions contain algorithm sources, not machine-local binaries or weights."""

import tarfile
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
        "scenesmith/SOURCE.json",
    ):
        assert "physrealx/_embedded/" + source in names
    assert "physrealx/platform.py" in names
    assert not any(
        n.endswith((".so", ".o", ".pyc", ".ckpt")) or "/weights/" in n or "/build/" in n
        for n in names
    )


def test_source_distribution_contains_demo_and_website_sources_without_local_data(tmp_path):
    sdist_module = pytest.importorskip("hatchling.builders.sdist")
    root = Path(__file__).resolve().parents[2]
    builder = sdist_module.SdistBuilder(str(root))
    filename = next(builder.build(directory=str(tmp_path), versions=["standard"]))
    with tarfile.open(tmp_path / filename) as archive:
        names = [name.split("/", 1)[-1] for name in archive.getnames()]
    assert "examples/catalog.py" in names
    assert "examples/run_all.py" in names
    assert "examples/physical_asset_building/demo.ipynb" in names
    assert "docs/_hooks/demo_gallery.py" in names
    assert "src/physrealx/services/assets.py" in names
    assert "src/physrealx/services.py" not in names
    assert not any(
        name.startswith("examples/") and ("/input/" in name or "/output/" in name) for name in names
    )
