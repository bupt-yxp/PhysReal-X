from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx import PhysRealX
from physrealx.domain import MethodCapability, RunStatus
from physrealx.errors import DomainValidationError, MethodUnavailableError
from physrealx.integrations.isaac_sim import IsaacSimBackend, IsaacSimConfig, isaac_installation
from physrealx.reference import reference_asset, reference_scene, reference_task


def test_simulation_facade_uses_the_episode_executor(tmp_path: Path) -> None:
    px = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    px.assets.register(asset)
    scene = reference_scene(asset)
    scene_ref = px.scenes.register(scene)
    task_ref = px.tasks.register(reference_task(scene))
    policy = px.methods.bind(MethodCapability.POLICY_INFERENCE, "scripted_drawer")
    record = px.simulations.run(
        scene_ref,
        task_ref,
        backend="reference_drawer",
        policy=policy,
        dataset_id="facade.dataset",
        episode_count=2,
    )
    assert record.status is RunStatus.SUCCEEDED
    assert record.result is not None and record.result.episode_count == 2
    assert all(episode.success for episode in px.runs.episodes(record))
    with pytest.raises(DomainValidationError, match="already bound"):
        px.simulations.run(
            scene_ref,
            task_ref,
            backend=px.methods.bind(MethodCapability.SIMULATION_RUN, "reference_drawer"),
            config={},
            policy=policy,
            dataset_id="invalid",
        )
    with pytest.raises(TypeError, match="method"):
        px.simulations.run(scene_ref, task_ref, method="reference_drawer")


def test_isaac_is_registered_without_importing_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.delenv("PHYSREALX_ISAAC_SIM_ROOT", raising=False)
    px = PhysRealX.open(tmp_path, discover_plugins=False)
    installation = px.methods.inspect(MethodCapability.SIMULATION_RUN, "isaac_sim")
    assert not installation.ready
    assert "PHYSREALX_ISAAC_SIM_ROOT" in installation.detail
    assert "isaacsim" not in sys.modules
    assert not any(name.startswith("omni.") for name in sys.modules)
    with pytest.raises(MethodUnavailableError):
        px.plugins.backend("isaac_sim")
    capabilities = IsaacSimBackend(px.artifacts).capabilities()
    assert capabilities.contact_force
    assert any("camera" in pattern for pattern in capabilities.observation_patterns)


@pytest.mark.parametrize(
    "overrides",
    [
        {"controlled_joints": ["x", "x"]},
        {"initial_positions": []},
        {"effort_limits": [-1]},
        {"effort_limits": [float("inf")]},
        {"passive_positions": {"x": 1}},
        {"gravity_m_s2": [0, 0, float("nan")]},
        {"timestep_seconds": 0},
        {"cameras": {"bad.name": {}}},
        {"randomization": {"mass_scale": [2, 1]}},
    ],
)
def test_isaac_config_rejects_invalid_or_unsupported_settings(overrides: dict) -> None:
    config = {"controlled_joints": ["x"], "initial_positions": [0], "effort_limits": [10]}
    with pytest.raises(ValidationError):
        IsaacSimConfig.model_validate({**config, **overrides})


def test_isaac_requires_the_tested_runtime_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "python.sh").write_text("#!/bin/sh\n")
    (tmp_path / "VERSION").write_text("5.1.0")
    monkeypatch.setenv("PHYSREALX_ISAAC_SIM_ROOT", str(tmp_path))
    with pytest.raises(MethodUnavailableError, match=r"4\.5\.0"):
        isaac_installation()


def test_usd_shared_author_preserves_box_assets(tmp_path: Path) -> None:
    pytest.importorskip("pxr.Usd")
    pytest.importorskip("trimesh")
    from physrealx.integrations.usd import author_asset_stage, validate_usd_stage

    px = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    stage, textures, count = author_asset_stage(asset, px.artifacts)
    validate_usd_stage(stage, asset, px.artifacts)
    assert count == len(asset.geometries)
    assert not textures


@pytest.mark.parametrize("change", ["unknown_joint", "joint_limit", "units", "unsupported_signal"])
def test_isaac_rejects_invalid_contract_before_starting_kit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from physrealx.domain import SignalDType, SignalSpec
    from physrealx.integrations import isaac_sim

    px = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    scene = reference_scene(asset)
    task = reference_task(scene)
    task = task.model_copy(
        update={
            "action": task.action.model_copy(
                update={"signal": task.action.signal.model_copy(update={"frame": "joint"})}
            ),
            "initial_conditions": tuple(
                p.model_copy(update={"signal": "joint.position"}) for p in task.initial_conditions
            ),
            "success_conditions": tuple(
                p.model_copy(update={"signal": "joint.position"}) for p in task.success_conditions
            ),
            "observations": (
                SignalSpec(
                    name="joint.position",
                    shape=(1,),
                    dtype=SignalDType.FLOAT32,
                    unit="m",
                    frame="joint",
                ),
            ),
        }
    )
    config = {
        "controlled_joints": ["drawer/drawer.slide"],
        "initial_positions": [0.0],
        "effort_limits": [10.0],
    }
    if change == "unknown_joint":
        config["controlled_joints"] = ["drawer/missing"]
    elif change == "joint_limit":
        config["initial_positions"] = [1.0]
    elif change == "units":
        task = task.model_copy(
            update={
                "action": task.action.model_copy(
                    update={
                        "signal": task.action.signal.model_copy(update={"unit": "rad"}),
                    }
                )
            }
        )
    else:
        task = task.model_copy(
            update={
                "observations": (
                    *task.observations,
                    task.observations[0].model_copy(update={"name": "body.external_force"}),
                )
            }
        )

    def unexpected_start() -> None:
        pytest.fail("invalid contract must be rejected before probing or launching Kit")

    monkeypatch.setattr(isaac_sim, "isaac_installation", unexpected_start)
    with pytest.raises(DomainValidationError):
        IsaacSimBackend(px.artifacts).create_session(scene, {"drawer": asset}, task, config)
