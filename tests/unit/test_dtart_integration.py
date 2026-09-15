"""DTArt input isolation and metric output normalization."""

import json

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from physrealx import PhysRealX
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations import dtart_capture
from physrealx.integrations.dtart import DTArtRunConfig, normalized_dtart_prediction
from physrealx.integrations.rgbd import MetricRGBDInput, RGBDView


def test_dtart_capture_excludes_validation_and_uses_millimetres(tmp_path, monkeypatch):
    import cv2
    import yaml

    px = PhysRealX.open(tmp_path / "store")
    ref = px.artifacts.put_json({"fixture": True})
    source = MetricRGBDInput(camera_rig=ref, enhanced_depth=ref, reference_model=ref)
    views = []
    for state in ("closed", "open"):
        for i in range(4):
            views.append(
                RGBDView(
                    f"{state}.{i}",
                    state,
                    {"model": "PINHOLE", "width": 16, "height": 16, "params": [20, 20, 8, 8]},
                    np.c_[np.eye(3), np.zeros(3)],
                    np.full((16, 16, 3), 127, dtype=np.uint8),
                    np.ones((16, 16)),
                    np.ones((16, 16, 3)),
                    np.ones((16, 16), bool),
                )
            )
    monkeypatch.setattr(
        dtart_capture, "load_rgbd", lambda *_: ({"coordinate_frame": "capture_metric"}, {}, views)
    )
    target = tmp_path / "data"
    metadata = dtart_capture.prepare_dtart_capture(
        source, px.artifacts, target, state_streams=("closed", "open"), training_frames_per_state=2
    )
    train = set(metadata["training_frame_ids"])
    validation = set(metadata["validation_frame_ids"])
    assert len(train) == len(validation) == 4
    assert train.isdisjoint(validation)
    assert train | validation == {v.frame_id for v in views}
    assert len(list((target / "color_segmented").glob("*.png"))) == 4
    assert len(list((target / "validation/color_segmented").glob("*.png"))) == 4
    depth = cv2.imread(str(target / "depth_filtered/0_000001.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and np.all(depth == 1000)
    poses = yaml.safe_load((target / "init_keyframes.yml").read_text())
    np.testing.assert_allclose(poses["frame_0_000001"]["cam_in_ob"], np.diag([1, -1, -1, 1]))
    assert not (target / "gt").exists()
    with pytest.raises(DomainValidationError, match="requires 6"):
        dtart_capture.prepare_dtart_capture(
            source,
            px.artifacts,
            tmp_path / "too-few",
            state_streams=("closed", "open"),
            training_frames_per_state=4,
        )


@pytest.mark.parametrize("kind", ["slide", "hinge"])
def test_dtart_motion_normalizes_metric_transform(tmp_path, kind):
    px = PhysRealX.open(tmp_path / "store")
    config = DTArtRunConfig(joint_types=(kind,), state_streams=("closed", "open"))
    origin = np.array([0.2, 0.3, 0])
    rotation = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix() if kind == "hinge" else np.eye(3)
    translation = (np.eye(3) - rotation) @ origin if kind == "hinge" else np.array([0.125, 0, 0])
    filename = "init_revolute_motion.json" if kind == "hinge" else "init_prismatic_motion.json"
    (tmp_path / filename).write_text(
        json.dumps([{"rotation": rotation.tolist(), "translation": translation.tolist()}])
    )
    for state in ("init", "last"):
        for i in range(2):
            trimesh.creation.box().export(tmp_path / f"{state}_part_{i}.obj")
    prediction = normalized_dtart_prediction(tmp_path, px.artifacts, config)
    joint = prediction.parts["1"]
    assert joint.span == pytest.approx(np.pi / 2 if kind == "hinge" else 0.125)
    np.testing.assert_allclose(joint.axis, [0, 0, 1] if kind == "hinge" else [1, 0, 0], atol=1e-8)
    if kind == "hinge":
        np.testing.assert_allclose(joint.origin, origin, atol=1e-8)
    assert prediction.motion_pair == ("closed", "open")
    for refs in prediction.states.values():
        assert all(px.artifacts.verify(ref) for ref in refs.values())
    (tmp_path / filename).write_text("[]")
    with pytest.raises(MethodExecutionError, match="motion count"):
        normalized_dtart_prediction(tmp_path, px.artifacts, config)
