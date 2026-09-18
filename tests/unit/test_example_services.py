"""Application behavior extracted from examples: placement and encoder ownership."""

import pytest

from physrealx import PhysRealX, SceneSpec
from physrealx.domain import SceneEntitySpec
from physrealx.errors import DomainValidationError
from physrealx.integrations.world_model import EncoderSpec


def test_encoder_cache_is_verified_and_does_not_download_when_valid(tmp_path, monkeypatch):
    import urllib.request

    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    weights = px.artifacts.put_bytes(b"test weights")
    encoder = px.artifacts.put_json(EncoderSpec(weights=weights))
    cache = tmp_path / "encoder.json"
    px.artifacts.materialize(encoder, cache)

    def unexpected(*args, **kwargs):
        raise AssertionError("valid cache must not download")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    result = px.world_models.prepare_encoder(cache=cache)
    assert px.artifacts.get_bytes(result) == cache.read_bytes()


def test_encoder_import_failure_preserves_existing_cache(tmp_path, monkeypatch):
    from physrealx.integrations import world_model

    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    cache = tmp_path / "encoder.json"
    cache.write_text("previous cache")

    def fail(*args, **kwargs):
        raise DomainValidationError("invalid checkpoint")

    monkeypatch.setattr(world_model, "import_dinov2_weights", fail)
    with pytest.raises(DomainValidationError, match="invalid checkpoint"):
        px.world_models.prepare_encoder(cache=cache, weights_path=tmp_path / "supplied.pth")
    assert cache.read_text() == "previous cache"
    assert not list(tmp_path.glob(".encoder-*"))


def test_sensor_service_derives_body_count_and_preserves_task_contract(tmp_path):
    from physrealx import TaskSpec
    from physrealx.domain import ActionSpec, Predicate, SignalSpec
    from physrealx.integrations.sensors import CameraSpec

    px = PhysRealX.open(tmp_path, discover_plugins=False)
    ref = px.assets.create_room_shell(
        resource_id="robot.sensor-test", width_m=0.5, depth_m=0.5, height_m=0.5
    )
    asset = px.assets.get(ref)
    scene = px.scenes.register(
        SceneSpec(
            resource_id="sensor.scene",
            version="1.0.0",
            display_name="Sensor fixture",
            entities=(SceneEntitySpec(name="robot", asset=ref, role="robot"),),
        )
    )
    signal = SignalSpec(name="position", shape=(1,), dtype="float32", unit="m", frame="world")
    task = TaskSpec(
        resource_id="sensor.task",
        version="1.0.0",
        scene=scene,
        subject_entity="robot",
        instruction="Move",
        initial_conditions=(),
        success_conditions=(Predicate(signal="position", operator="ge", value=0.1),),
        action=ActionSpec(
            kind="joint_position",
            signal=signal.model_copy(update={"name": "command"}),
            lower=(0,),
            upper=(1,),
        ),
        observations=(signal,),
        max_steps=100,
    )
    cameras = {"front": CameraSpec.look_at([1, 0, 1], [0, 0, 0], width=64, height=48)}
    updated = px.tasks.with_sensors(
        task, scene, cameras, resource_id="sensor.rendered", frequency_hz=20
    )
    signals = {s.name: s for s in updated.observations}
    assert signals["body.position"].shape == (len(asset.links), 3)
    assert signals["camera.front.rgb"].shape == (48, 64, 3)
    assert updated.action == task.action and updated.success_conditions == task.success_conditions
    assert len(task.observations) == 1
    with pytest.raises(ValueError, match="unique"):
        px.tasks.with_sensors(
            updated, scene, cameras, resource_id="sensor.duplicate", frequency_hz=20
        )
