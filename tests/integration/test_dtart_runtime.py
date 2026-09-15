"""Regression for loading DTArt's native extension once and exiting normally."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def test_native_extensions_have_one_import_identity():
    root = Path(__file__).resolve().parents[2] / "algorithms/dtart"
    if importlib.util.find_spec("kaolin") is None or not list((root / "mycuda").glob("common*.so")):
        pytest.skip("DTArt native runtime is not built in this environment")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import main, sys; "
            "assert 'mycuda.common' in sys.modules; "
            "assert 'mycuda.gridencoder' in sys.modules; "
            "assert 'common' not in sys.modules; "
            "assert 'gridencoder' not in sys.modules",
        ],
        cwd=root,
        env={**os.environ, "PYOPENGL_PLATFORM": "egl", "WANDB_MODE": "disabled"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.real_data
@pytest.mark.skipif(
    os.environ.get("PHYSREALX_REAL_DATA_TESTS") != "1",
    reason="requires an actually completed DTArt reconstruction",
)
def test_export_contains_native_checkpoint_and_both_state_meshes():
    import numpy as np
    import torch
    import trimesh

    root = Path(__file__).resolve().parents[2]
    output = root / "examples/real_assets_and_digital_twin/output"
    report = json.loads((output / "reconstruction-report.json").read_text())
    assert report["method"]["name"] == "dtart"
    with zipfile.ZipFile(output / "reconstruction.zip") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["prediction"] == report["prediction"]
        config = json.loads(archive.read("run.json"))
        checkpoint = torch.load(
            io.BytesIO(archive.read("model/checkpoint.pth")),
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["global_step"] + 1 == config["iterations"]
        for state in ("init", "last"):
            for identifier in report["prediction"]["parts"]:
                mesh = trimesh.load(
                    io.BytesIO(archive.read(f"model/{state}_part_{identifier}.obj")),
                    file_type="obj",
                    force="mesh",
                )
                assert mesh.area > 0 and np.isfinite(mesh.vertices).all()
            for name in (f"{state}_model", f"{state}_embed_fn"):
                assert checkpoint[name]
                assert all(torch.isfinite(value).all() for value in checkpoint[name].values())
