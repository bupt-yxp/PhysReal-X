from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from physrealx import PhysRealX
from physrealx.errors import DomainValidationError
from physrealx.reference import reference_asset


def test_physical_asset_preview_animates_visual_and_collision_geometry(tmp_path) -> None:
    px = PhysRealX.open(tmp_path / "workspace")
    assembly = px.artifacts.put_json(
        {
            "schema": "physrealx.asset-assembly-bundle/1",
            "asset_spec": reference_asset().model_dump(mode="json"),
        }
    )

    animation = px.assets.preview_physical_asset(assembly, frames=3, fps=10)

    with Image.open(io.BytesIO(px.artifacts.get_bytes(animation))) as gif:
        assert gif.size == (900, 360)
        assert gif.n_frames == 3
        assert gif.info["duration"] == 100
        metadata = json.loads(gif.info["comment"])
        assert metadata["assembly"]["artifact_id"] == assembly.artifact_id
        assert metadata["joints"] == ["drawer.slide"]
        first = np.asarray(gif.convert("RGB"))
        gif.seek(2)
        last = np.asarray(gif.convert("RGB"))
        assert np.any(first[:330, :450] != last[:330, :450])
        assert np.any(first[:330, 450:] != last[:330, 450:])


def test_physical_asset_preview_rejects_invalid_requests(tmp_path) -> None:
    px = PhysRealX.open(tmp_path / "workspace")
    assembly = px.artifacts.put_json(
        {
            "schema": "physrealx.asset-assembly-bundle/1",
            "asset_spec": reference_asset().model_dump(mode="json"),
        }
    )

    with pytest.raises(DomainValidationError, match="frames"):
        px.assets.preview_physical_asset(assembly, frames=True)
    with pytest.raises(DomainValidationError, match="fps"):
        px.assets.preview_physical_asset(assembly, fps=0)
    with pytest.raises(DomainValidationError, match="asset specification"):
        px.assets.preview_physical_asset(
            px.artifacts.put_json({"schema": "physrealx.asset-assembly-bundle/1"})
        )
