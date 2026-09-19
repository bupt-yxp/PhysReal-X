from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from physrealx import PhysRealX
from physrealx.domain import (
    ActionSpec,
    ArtifactRef,
    AssetSpec,
    MethodCapability,
    Predicate,
    PredicateOperator,
    RunSpec,
    RunStatus,
    SceneEntityRole,
    SceneEntitySpec,
    SceneSpec,
    SignalDType,
    SignalSpec,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations.metric import fit_similarity
from physrealx.integrations.mujoco import MujocoConfig, compile_scene

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
trimesh = pytest.importorskip("trimesh")
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("coacd")
pytest.importorskip("pxr.Usd")


def array(px: PhysRealX, values: object) -> ArtifactRef:
    stream = io.BytesIO()
    np.save(stream, np.asarray(values), allow_pickle=False)
    return px.artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy")


def output(operation: object, name: str) -> ArtifactRef:
    return next(item.artifact for item in operation.outputs if item.name == name)


@pytest.fixture
def px(tmp_path: Path) -> PhysRealX:
    return PhysRealX.open(tmp_path / "workspace", discover_plugins=False)


def metric_inputs(px: PhysRealX) -> tuple[ArtifactRef, dict[str, object]]:
    camera = px.artifacts.put_json(
        {
            "schema": "physrealx.camera-rig-calibration/1",
            "scale_status": "unresolved",
            "streams": [
                {
                    "stream_name": "start",
                    "components": [
                        {
                            "component_id": 0,
                            "images": [
                                {
                                    "name": "frame-0",
                                    "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]],
                                }
                            ],
                            "cameras": [],
                        }
                    ],
                }
            ],
        }
    )
    points = np.random.default_rng(3).normal(size=(20, 3))
    target = 2 * points + np.array([0.1, 0.2, 0.3])
    evidence = {
        "schema": "physrealx.metric-alignment-evidence/1",
        "unit": "m",
        "camera_rig_artifact_id": camera.artifact_id,
        "component_id": "start:0",
        "source_frame": "colmap_world_unscaled",
        "target_frame": "object",
        "metric_reference": {
            "kind": "surveyed_length",
            "identifier": "fixture-ruler",
            "source": "fixture-certificate",
            "value_m": 1.0,
        },
        "provenance": {
            "capture_id": "synthetic-test-only",
            "measurement_method": "generated_ground_truth",
            "source_uri": "fixture://metric",
        },
        "train_correspondences": [
            {"id": f"train-{i}", "source_xyz": p.tolist(), "target_xyz_m": t.tolist()}
            for i, (p, t) in enumerate(zip(points[:15], target[:15]))
        ],
        "holdout_correspondences": [
            {"id": f"test-{i}", "source_xyz": p.tolist(), "target_xyz_m": t.tolist()}
            for i, (p, t) in enumerate(zip(points[15:], target[15:]))
        ],
    }
    return camera, evidence


def metric_camera(px: PhysRealX) -> ArtifactRef:
    camera, evidence = metric_inputs(px)
    result = px.assets.align_cameras_metric(
        camera, px.artifacts.put_json(evidence), method="ransac_umeyama"
    )
    return output(result, "camera_rig_calibration")


def test_metric_recovers_scale_rejects_outliers_and_transforms_camera_pose(px: PhysRealX) -> None:
    camera, evidence = metric_inputs(px)
    evidence["train_correspondences"][0]["target_xyz_m"] = [20, 20, 20]
    result = px.assets.align_cameras_metric(
        camera, px.artifacts.put_json(evidence), method="ransac_umeyama"
    )
    assert result.metrics["scale"] == pytest.approx(2)
    document = json.loads(px.artifacts.get_bytes(output(result, "camera_rig_calibration")))
    matrix = np.array(document["streams"][0]["components"][0]["images"][0]["cam_from_world"])
    np.testing.assert_allclose(matrix[:, 3], [-0.1, -0.2, 1.7], atol=1e-12)
    assert document["unit"] == "m"


def test_metric_rejects_holdout_leakage_and_does_not_publish_operation(px: PhysRealX) -> None:
    camera, evidence = metric_inputs(px)
    evidence["holdout_correspondences"][0]["source_xyz"] = evidence["train_correspondences"][0][
        "source_xyz"
    ]
    with pytest.raises(DomainValidationError, match="disjoint"):
        px.assets.align_cameras_metric(
            camera, px.artifacts.put_json(evidence), method="ransac_umeyama"
        )
    assert px.operations.list() == []
    with pytest.raises(DomainValidationError, match="degenerate"):
        fit_similarity(np.zeros((4, 3)), np.ones((4, 3)))


def test_depth_fills_small_holes_and_blocks_unsupported_regions(px: PhysRealX) -> None:
    camera = metric_camera(px)
    raw = np.ones((12, 12), dtype=np.float32)
    raw[6, 6] = 0
    request = {
        "schema": "physrealx.calibrated-rgbd-bundle/1",
        "camera_rig": camera.model_dump(mode="json"),
        "frames": [
            {
                "frame_id": "frame-0",
                "rgb": array(px, np.zeros((12, 12, 3), dtype=np.uint8)).model_dump(mode="json"),
                "depth": array(px, raw).model_dump(mode="json"),
                "depth_scale_m": 1.0,
                "reference_depth_m": array(px, np.ones((12, 12), dtype=np.float32)).model_dump(
                    mode="json"
                ),
                "capture_id": "synthetic-input",
                "reference_capture_id": "synthetic-holdout",
            }
        ],
    }
    result = px.assets.enhance_depth(px.artifacts.put_json(request), method="rgb_guided_fill")
    assert result.metrics["filled_pixels"] == 1
    with zipfile.ZipFile(
        io.BytesIO(px.artifacts.get_bytes(output(result, "enhanced_depth")))
    ) as archive:
        depth = np.load(io.BytesIO(archive.read("depth/frame-0.npy")), allow_pickle=False)
        np.testing.assert_array_equal(depth, np.ones((12, 12)))
    request["frames"][0]["depth"] = array(px, np.zeros((12, 12))).model_dump(mode="json")
    with pytest.raises(MethodExecutionError, match="local support"):
        px.assets.enhance_depth(px.artifacts.put_json(request), method="rgb_guided_fill")


def joint_articulation(px: PhysRealX, camera: ArtifactRef) -> ArtifactRef:
    points = np.random.default_rng(7).normal(size=(30, 3)) * 0.05
    request = {
        "schema": "physrealx.paired-interaction-point-cloud-bundle/1",
        "camera_rig": camera.model_dump(mode="json"),
        "joint_name": "slide",
        "parent": "base",
        "child": "drawer",
        "lower_limit": 0.0,
        "upper_limit": 0.5,
        "limit_unit": "m",
        "limit_evidence": px.artifacts.put_json({"fixture": "measured limits"}).model_dump(
            mode="json"
        ),
    }
    for key, values in {
        "source_points": points[:20],
        "target_points": points[:20] + np.array([0.2, 0, 0]),
        "holdout_source_points": points[20:],
        "holdout_target_points": points[20:] + np.array([0.2, 0, 0]),
    }.items():
        request[key] = array(px, values).model_dump(mode="json")
    result = px.assets.articulate(px.artifacts.put_json(request), method="rigid_pair_fit")
    return output(result, "articulation_model")


def identification_request(px: PhysRealX, *, degenerate: bool = False) -> ArtifactRef:
    trajectories = []
    for seed in [31, 55]:
        time = np.linspace(0, 8, 1000)
        phase = seed / 10
        q = 0.2 + 0.05 * np.sin(time * 2 + phase) + 0.03 * np.sin(time * 5)
        velocity = 0.1 * np.cos(time * 2 + phase) + 0.15 * np.cos(time * 5)
        acceleration = -0.2 * np.sin(time * 2 + phase) - 0.75 * np.sin(time * 5)
        if degenerate:
            q = 0.2 + 0.01 * time
            velocity[:] = 0.01
            acceleration[:] = 0
        force = 2 * acceleration + 0.3 * velocity + 0.2 * np.sign(velocity)
        trajectories.append(array(px, np.column_stack((time, q, velocity, acceleration, force))))
    return px.artifacts.put_json(
        {
            "schema": "physrealx.joint-system-identification-evidence/1",
            "joint_name": "slide",
            "joint_type": "slide",
            "train": trajectories[0].model_dump(mode="json"),
            "holdout": trajectories[1].model_dump(mode="json"),
            "train_capture_id": "synthetic-fit",
            "holdout_capture_id": "synthetic-select",
            "position_unit": "m",
            "force_unit": "N",
            "gravity_compensated": True,
            "prior_parameters": [3.0, 0.8, 0.5],
        }
    )


def test_identification_recovers_parameters_and_rejects_unobservable_input(px: PhysRealX) -> None:
    result = px.assets.calibrate_physics(
        identification_request(px), method="bounded_inverse_dynamics"
    )
    value = json.loads(px.artifacts.get_bytes(output(result, "identified_joint_physics")))
    assert [
        value["parameters"][name]["value"]
        for name in ["effective_inertia", "viscous_damping", "coulomb_friction"]
    ] == pytest.approx([2, 0.3, 0.2])
    with pytest.raises(MethodExecutionError, match="NOT_IDENTIFIABLE"):
        px.assets.calibrate_physics(
            identification_request(px, degenerate=True), method="bounded_inverse_dynamics"
        )


def assembled_asset(px: PhysRealX, *, textured: bool = False) -> tuple[AssetSpec, ArtifactRef]:
    camera = metric_camera(px)
    articulation = joint_articulation(px, camera)
    physics = output(
        px.assets.calibrate_physics(identification_request(px), method="bounded_inverse_dynamics"),
        "identified_joint_physics",
    )
    collision_reports = []
    bodies = []
    for link, position in [("base", [-0.2, 0, 0]), ("drawer", [0, 0, 0])]:
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        mesh.apply_translation(position)
        if textured:
            from PIL import Image

            pixels = np.zeros((4, 4, 3), np.uint8)
            pixels[::2] = [220, 120, 30]
            pixels[1::2] = [30, 80, 190]
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.vertices[:, :2] * 10,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray(pixels),
                    metallicFactor=0.15,
                    roughnessFactor=0.7,
                ),
            )
        visual = px.artifacts.put_bytes(
            mesh.export(file_type="glb"), media_type="model/gltf-binary"
        )
        collision = px.assets.build_collision(
            visual, method="coacd", config={"source_unit": "m", "link": link}
        )
        collision_reports.append(
            output(collision, "collision_build_report").model_dump(mode="json")
        )
        measured = px.artifacts.put_json({"fixture": "body measurements", "link": link})

        def parameter(
            value: float, unit: str, measured: ArtifactRef = measured
        ) -> dict[str, object]:
            return {
                "value": value,
                "unit": unit,
                "evidence_artifact_id": measured.artifact_id,
                "uncertainty": 0.001,
                "confidence": 0.95,
                "source": "measured",
            }

        bodies.append(
            {
                "link": link,
                "mass_kg": parameter(2, "kg"),
                "center_of_mass_m": [0, 0, 0],
                "inertia_kg_m2": [[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]],
                "static_friction": parameter(0.4, "1"),
                "dynamic_friction": parameter(0.3, "1"),
                "restitution": parameter(0, "1"),
                "evidence_artifact_id": measured.artifact_id,
                "uncertainty": 0.001,
                "confidence": 0.95,
                "source": "measured",
            }
        )
    body_physics = px.artifacts.put_json(
        {
            "schema": "physrealx.identified-body-physics/1",
            "status": "IDENTIFIED",
            "evidence_state": "identified",
            "bodies": bodies,
        }
    )
    request = px.artifacts.put_json(
        {
            "schema": "physrealx.asset-assembly-input/1",
            "resource_id": "fixture.asset",
            "version": "1.0.0",
            "display_name": "Synthetic integration fixture",
            "camera_rig": camera.model_dump(mode="json"),
            "articulation": articulation.model_dump(mode="json"),
            "collision_reports": collision_reports,
            "body_physics": body_physics.model_dump(mode="json"),
            "joint_physics": [physics.model_dump(mode="json")],
        }
    )
    assembly = output(
        px.assets.assemble(request, method="measured_asset_assembly"), "asset_assembly"
    )
    document = json.loads(px.artifacts.get_bytes(assembly))
    return AssetSpec.model_validate(document["asset_spec"]), assembly


def scene_task(px: PhysRealX, asset: AssetSpec) -> tuple[SceneSpec, TaskSpec, dict[str, object]]:
    asset_ref = px.assets.register(asset)
    scene = SceneSpec(
        resource_id="fixture.scene",
        version="1.0.0",
        display_name="Fixture scene",
        entities=(SceneEntitySpec(name="asset", asset=asset_ref, role=SceneEntityRole.OBJECT),),
    )
    scene_ref = px.scenes.register(scene)
    task = TaskSpec(
        resource_id="fixture.task",
        version="1.0.0",
        scene=scene_ref,
        instruction="Open slider",
        subject_entity="asset",
        observations=(
            SignalSpec(
                name="joint.position",
                shape=(1,),
                dtype=SignalDType.FLOAT32,
                unit="m",
                frame="joint",
            ),
            SignalSpec(
                name="joint.velocity",
                shape=(1,),
                dtype=SignalDType.FLOAT32,
                unit="m/s",
                frame="joint",
            ),
        ),
        action=ActionSpec(
            kind="joint_position",
            signal=SignalSpec(
                name="joint.command", shape=(1,), dtype=SignalDType.FLOAT32, unit="m", frame="joint"
            ),
            lower=(0,),
            upper=(0.5,),
        ),
        initial_conditions=(
            Predicate(signal="joint.position", operator=PredicateOperator.LE, value=0.01),
        ),
        success_conditions=(
            Predicate(signal="joint.position", operator=PredicateOperator.GE, value=0.25),
        ),
        max_steps=100,
    )
    px.tasks.register(task)
    config = {
        "controlled_joints": ["asset/slide"],
        "initial_positions": [0.0],
        "effort_limits": [20.0],
        "gravity_m_s2": [0.0, 0.0, 0.0],
    }
    return scene, task, config


def test_complete_assembly_usd_real_dynamics_dataset_and_training(px: PhysRealX) -> None:
    asset, assembly = assembled_asset(px)
    package = px.assets.package(assembly, method="openusd")
    usd = output(package, "usd_asset_bundle")
    assert px.artifacts.get_bytes(usd).startswith(b"PK")
    assert usd.media_type == "model/vnd.usdz+zip"
    scene, task, config = scene_task(px, asset)
    run = RunSpec(
        scene=scene.reference(),
        task=task.reference(),
        backend=px.methods.bind(MethodCapability.SIMULATION_RUN, "mujoco", config=config),
        policy=px.methods.bind(
            MethodCapability.POLICY_INFERENCE, "joint_target", config={"target": [0.3]}
        ),
        episode_count=2,
        base_seed=2,
        dataset_id="fixture.dataset",
        dataset_version="1.0.0",
        record_signals=("joint.position", "joint.velocity"),
    )
    record = px.runs.submit(run)
    assert record.status is RunStatus.SUCCEEDED, record.result
    episodes = px.runs.episodes(record)
    assert all(episode.success for episode in episodes)
    assert (
        episodes[0].steps[1].state["joint.position"].inline[0] < 0.3
    )  # finite acceleration, no teleporting.
    np.testing.assert_allclose(
        [s.state["joint.position"].inline for s in episodes[0].steps],
        [s.state["joint.position"].inline for s in episodes[1].steps],
    )
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    model = px.training.submit(
        TrainingSpec(
            resource_id="fixture.policy",
            version="1.0.0",
            dataset=record.result.dataset,
            trainer=px.methods.bind(
                MethodCapability.POLICY_TRAIN, "torch_bc", config={"epochs": 40, "device": "cpu"}
            ),
        )
    )
    assert px.artifacts.verify(px.training.get_model(model).weights)
    report = px.evaluations.submit(
        resource_id="fixture.eval",
        version="1.0.0",
        model=model,
        run=run,
        episode_count=2,
        base_seed=1000,
    )
    assert px.evaluations.get_report(report).success_rate >= 0.5


def test_physical_holdout_replays_and_rejects_wrong_trajectory(px: PhysRealX) -> None:
    asset, assembly = assembled_asset(px)
    package = output(px.assets.package(assembly, method="openusd"), "asset_package_manifest")
    scene, _, settings = scene_task(px, asset)
    config = MujocoConfig.model_validate(settings)
    compiled = compile_scene(scene, {"asset": asset}, px.artifacts, config)
    model = mujoco.MjModel.from_xml_string(compiled.xml, assets=compiled.files)
    data = mujoco.MjData(model)
    rows = []
    for index in range(60):
        force = 0.4 * np.sin(index / 6) + 0.25
        rows.append([index * 0.05, data.qpos[0], data.qvel[0], force])
        data.ctrl[0] = force
        mujoco.mj_step(model, data, nstep=25)
    request = {
        "schema": "physrealx.physical-validation-evidence/1",
        "package_manifest": package.model_dump(mode="json"),
        "trajectory": array(px, rows).model_dump(mode="json"),
        "capture_id": "synthetic-third-capture",
        "position_unit": "m",
        "force_unit": "N",
        "simulation": settings,
    }
    report = px.assets.validate_physics(
        package, px.artifacts.put_json(request), method="mujoco_holdout"
    )
    assert report.metrics["position_rmse"] < 1e-10
    bad = np.array(rows)
    bad[:, 1] += np.linspace(0, 0.2, len(bad))
    request["trajectory"] = array(px, bad).model_dump(mode="json")
    with pytest.raises(MethodExecutionError, match="holdout"):
        px.assets.validate_physics(package, px.artifacts.put_json(request), method="mujoco_holdout")


def test_revolute_recovery_and_rejection_of_helical_motion(px: PhysRealX) -> None:
    from scipy.spatial.transform import Rotation

    camera = metric_camera(px)
    points = np.random.default_rng(17).normal(size=(30, 3)) * 0.1
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    origin = np.array([0.15, 0.2, 0.1])
    rotation = Rotation.from_rotvec(axis * 0.5).as_matrix()
    target = (points - origin) @ rotation.T + origin
    request = {
        "schema": "physrealx.paired-interaction-point-cloud-bundle/1",
        "camera_rig": camera.model_dump(mode="json"),
        "joint_name": "hinge",
        "parent": "base",
        "child": "door",
        "lower_limit": 0.0,
        "upper_limit": 1.0,
        "limit_unit": "rad",
        "limit_evidence": px.artifacts.put_json({"fixture": "limits"}).model_dump(mode="json"),
        "source_points": array(px, points[:20]).model_dump(mode="json"),
        "target_points": array(px, target[:20]).model_dump(mode="json"),
        "holdout_source_points": array(px, points[20:]).model_dump(mode="json"),
        "holdout_target_points": array(px, target[20:]).model_dump(mode="json"),
    }
    result = px.assets.articulate(px.artifacts.put_json(request), method="rigid_pair_fit")
    joint = json.loads(px.artifacts.get_bytes(output(result, "articulation_model")))["joints"][0]
    np.testing.assert_allclose(joint["axis"], axis, atol=1e-7)
    assert joint["joint_type"] == "revolute"
    request["target_points"] = array(px, target[:20] + axis * 0.05).model_dump(mode="json")
    with pytest.raises(MethodExecutionError, match="helical"):
        px.assets.articulate(px.artifacts.put_json(request), method="rigid_pair_fit")


def test_assembly_and_usd_reject_tampered_physics(px: PhysRealX, tmp_path: Path) -> None:
    from pxr import Usd, UsdPhysics

    from physrealx.integrations.usd import validate_usd_stage

    asset, assembly = assembled_asset(px)
    document = json.loads(px.artifacts.get_bytes(assembly))
    document["asset_spec"]["physics"][0]["mass_kg"]["value"] += 1
    with pytest.raises(MethodExecutionError):
        px.assets.package(px.artifacts.put_json(document), method="openusd")
    package = px.assets.package(assembly, method="openusd")
    path = tmp_path / "tampered.usdz"
    px.artifacts.materialize(output(package, "usd_asset_bundle"), path)
    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdPhysics.MassAPI(stage.GetPrimAtPath("/Asset/link_0")).GetMassAttr().Set(20.0)
    with pytest.raises(MethodExecutionError, match="mass"):
        validate_usd_stage(stage, asset, px.artifacts)


def test_textures_survive_portable_packaging_and_simulator_compilation(
    px: PhysRealX, tmp_path: Path
):
    from pxr import Usd, UsdGeom, UsdShade

    from physrealx.integrations.usd import validate_usd_stage

    asset, assembly = assembled_asset(px, textured=True)
    package = px.assets.package(assembly, method="openusd")
    assert package.metrics["textures"] == 1
    path = tmp_path / "textured.usdz"
    px.artifacts.materialize(output(package, "usd_asset_bundle"), path)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist()[0] == "asset.usdc"
        assert len([name for name in archive.namelist() if name.endswith(".png")]) == 1
    # The export temporary directory is gone; dependencies must resolve inside USDZ.
    stage = Usd.Stage.Open(str(path))
    validate_usd_stage(stage, asset, px.artifacts)
    scene, _, config = scene_task(px, asset)
    compiled = compile_scene(
        scene, {"asset": asset}, px.artifacts, MujocoConfig.model_validate(config)
    )
    model = mujoco.MjModel.from_xml_string(compiled.xml, assets=compiled.files)
    assert model.ntex == 1 and model.nmat == 2
    assert np.any(model.tex_data > 0)
    visual = next(
        UsdGeom.Mesh(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    )
    stage.SetEditTarget(stage.GetSessionLayer())
    coordinates = UsdGeom.PrimvarsAPI(visual).GetPrimvar("st")
    values = np.asarray(coordinates.Get()).copy()
    values[0] += 0.25
    coordinates.Set(values.tolist())
    with pytest.raises(DomainValidationError, match="texture coordinates"):
        validate_usd_stage(stage, asset, px.artifacts)


def test_mujoco_lifecycle_units_and_command_bounds(px: PhysRealX) -> None:
    from physrealx.domain import DataValue
    from physrealx.errors import InvalidStateTransitionError

    asset, _ = assembled_asset(px)
    scene, task, config = scene_task(px, asset)
    backend = px.plugins.backend("mujoco")
    session = backend.create_session(scene, {"asset": asset}, task, config)
    command = DataValue(
        dtype=SignalDType.FLOAT32, shape=(1,), unit="m", frame="joint", inline=(0.3,)
    )
    with pytest.raises(InvalidStateTransitionError, match="reset"):
        session.step(command)
    session.reset(1)
    with pytest.raises(DomainValidationError, match="bounds"):
        session.step(command.model_copy(update={"inline": (0.8,)}))
    session.close()
    session.close()
    with pytest.raises(InvalidStateTransitionError, match="closed"):
        session.reset(1)
    invalid = task.model_copy(
        update={
            "action": task.action.model_copy(
                update={"signal": task.action.signal.model_copy(update={"unit": "rad"})}
            )
        }
    )
    with pytest.raises(DomainValidationError, match="units"):
        backend.create_session(scene, {"asset": asset}, invalid, config)


def test_mujoco_rgb_depth_from_the_same_state_in_headless_process(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    import textwrap

    # EGL selection belongs to the process launching MuJoCo, never to the core SDK.
    script = textwrap.dedent("""
        import io, json, sys
        import numpy as np
        from pathlib import Path
        from test_physical_pipeline import assembled_asset, scene_task
        from physrealx import PhysRealX
        from physrealx.domain import SignalSpec, SignalDType
        px = PhysRealX.open(Path(sys.argv[1]) / "render", discover_plugins=False)
        asset, _ = assembled_asset(px)
        scene, task, config = scene_task(px, asset)
        task = task.model_copy(update={"observations": (*task.observations,
            SignalSpec(name="camera.observer.rgb", shape=(32,32,3),
                dtype=SignalDType.UINT8, unit="1", frame="camera.observer"),
            SignalSpec(name="camera.observer.depth", shape=(32,32),
                dtype=SignalDType.FLOAT32, unit="m", frame="camera.observer"))})
        config["cameras"] = {"observer": {
            "pose": {"translation_m": [0,0,1]}, "width":32, "height":32}}
        session = px.plugins.backend("mujoco").create_session(scene, {"asset":asset}, task, config)
        try:
            frame = session.reset(0)
            rgb = frame.observations["camera.observer.rgb"]
            depth = frame.observations["camera.observer.depth"]
            assert rgb.timestamp_seconds == depth.timestamp_seconds == 0.0
            pixels = np.load(io.BytesIO(px.artifacts.get_bytes(rgb.artifact)), allow_pickle=False)
            metric = np.load(io.BytesIO(px.artifacts.get_bytes(depth.artifact)), allow_pickle=False)
            assert pixels.shape == (32,32,3) and pixels.max() > pixels.min()
            assert metric.shape == (32,32) and np.isfinite(metric).all() and metric.min() >= 0
            assert 0 < metric[metric > 0].min() < 1.0
            print("render_verified")
        finally:
            session.close()
    """)
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "egl"
    environment["PYTHONPATH"] = str(Path(__file__).parent)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "render_verified" in completed.stdout


def test_nonconvex_collision_is_really_decomposed(px: PhysRealX) -> None:
    # Closed L-shaped prism, triangulated without a boolean engine or external mesh tool.
    outline = np.array([[0, 0], [0.2, 0], [0.2, 0.1], [0.1, 0.1], [0.1, 0.2], [0, 0.2]])
    vertices = np.vstack(
        (np.column_stack((outline, np.zeros(6))), np.column_stack((outline, np.full(6, 0.1))))
    )
    top = [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]
    faces = [list(reversed(face)) for face in top] + [[v + 6 for v in face] for face in top]
    for i in range(6):
        j = (i + 1) % 6
        faces.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    assert mesh.is_volume and not mesh.is_convex
    reference = px.artifacts.put_bytes(mesh.export(file_type="glb"), media_type="model/gltf-binary")
    result = px.assets.build_collision(
        reference,
        method="coacd",
        config={"source_unit": "m", "link": "l_shape", "threshold_m": 0.001},
    )
    assert result.metrics["convex_parts"] >= 2
    assert result.metrics["relative_volume_error"] <= 0.15


def test_unavailable_backend_is_blocked_by_the_same_probe_as_catalog(
    px: PhysRealX, monkeypatch: pytest.MonkeyPatch
) -> None:
    from physrealx.contracts import MethodProbe
    from physrealx.errors import MethodUnavailableError

    backend = px.plugins.backend("mujoco")
    monkeypatch.setattr(backend, "probe", lambda: MethodProbe(False, "native library absent"))
    installation = px.methods.inspect(MethodCapability.SIMULATION_RUN, "mujoco")
    assert not installation.ready and "native library absent" in installation.detail
    with pytest.raises(MethodUnavailableError):
        px.plugins.backend("mujoco")


def test_joint_target_delta_respects_state_and_action_limits(px: PhysRealX) -> None:
    from physrealx.contracts import Frame
    from physrealx.domain import DataValue
    from physrealx.errors import InvalidStateTransitionError
    from physrealx.policies import JointTargetPolicy

    asset, _ = assembled_asset(px)
    _, task, _ = scene_task(px, asset)
    task = task.model_copy(
        update={
            "action": task.action.model_copy(
                update={"kind": "joint_delta", "lower": (-0.05,), "upper": (0.05,)}
            )
        }
    )
    policy = JointTargetPolicy({"target": [0.3]})
    state = DataValue(dtype=SignalDType.FLOAT32, shape=(1,), unit="m", frame="joint", inline=(0.1,))
    assert policy.act(Frame(observations={"joint.position": state}, state={}), task, 0).inline == (
        0.05,
    )
    policy.close()
    with pytest.raises(InvalidStateTransitionError):
        policy.act(Frame(observations={}, state={}), task, 0)


def test_seeded_multimodal_observations_labels_and_external_wrench(px: PhysRealX) -> None:
    from physrealx.domain import DataValue
    from physrealx.integrations.sensors import project_depth

    asset, _ = assembled_asset(px)
    scene, task, config = scene_task(px, asset)
    modalities = (
        ("rgb", (32, 32, 3), "uint8", "1"),
        ("depth", (32, 32), "float32", "m"),
        ("mask", (32, 32), "int32", "1"),
        ("pointcloud", (32, 32, 3), "float32", "m"),
        ("events", (32, 32, 2), "int32", "1"),
        ("intrinsics", (3, 3), "float64", "1"),
        ("position", (3,), "float64", "m"),
        ("orientation", (4,), "float64", "1"),
    )
    signals = tuple(
        SignalSpec(
            name=f"camera.observer.{name}",
            shape=shape,
            dtype=dtype,
            unit=unit,
            frame="world" if name in {"position", "orientation"} else "camera.observer",
        )
        for name, shape, dtype, unit in modalities
    ) + tuple(
        SignalSpec(name="body." + name, shape=(2, width), dtype="float64", unit=unit, frame="world")
        for name, width, unit in (
            ("position", 3, "m"),
            ("orientation", 4, "1"),
            ("external_force", 3, "N"),
            ("external_torque", 3, "N*m"),
            ("wrench_origin", 3, "m"),
        )
    )
    task = task.model_copy(update={"observations": task.observations + signals})
    config.update(
        cameras={"observer": {"pose": {"translation_m": [0, 0, 1]}, "width": 32, "height": 32}},
        randomization={
            "mass_scale": [0.8, 1.2],
            "friction_scale": [0.5, 1.5],
            "color_jitter": 0.2,
            "light_jitter": 0.2,
            "camera_position_jitter_m": 0.01,
            "occlusion_probability": 1.0,
        },
    )
    session = px.plugins.backend("mujoco").create_session(scene, {"asset": asset}, task, config)

    def pixels(frame, name):
        value = frame.observations["camera.observer." + name]
        return np.load(io.BytesIO(px.artifacts.get_bytes(value.artifact)), allow_pickle=False)

    try:
        first = session.reset(7)
        masses = session.model.body_mass.copy()
        second = session.reset(8)
        assert not np.array_equal(pixels(first, "rgb"), pixels(second, "rgb"))
        replay = session.reset(7)
        np.testing.assert_array_equal(session.model.body_mass, masses)
        for name, *_ in modalities:
            np.testing.assert_array_equal(pixels(first, name), pixels(replay, name))
        labels = first.metadata["segmentation_labels"]
        assert set(str(int(v)) for v in np.unique(pixels(first, "mask"))) <= set(labels)
        assert first.metadata["body_pose_order"] == ["asset/base", "asset/drawer"]
        assert "occluder/observer" in labels.values()
        np.testing.assert_allclose(
            pixels(first, "pointcloud"),
            project_depth(pixels(first, "depth"), pixels(first, "intrinsics")),
        )
        body_id = session.model.body("asset/drawer").id
        session.data.xfrc_applied[body_id, :3] = [0, 0, 2]
        result = session.step(
            DataValue(dtype="float32", shape=(1,), unit="m", frame="joint", inline=(0.1,))
        )
        force = np.array(result.frame.observations["body.external_force"].inline).reshape(2, 3)
        assert force[-1, 2] == pytest.approx(2)
        assert all(
            value.timestamp_seconds == pytest.approx(session.dt_seconds)
            for value in result.frame.observations.values()
        )
    finally:
        session.close()


def test_grounded_layout_task_and_trajectory_execute_real_dynamics(px: PhysRealX) -> None:
    from physrealx.integrations.scenes import asset_vertices

    asset, _ = assembled_asset(px)
    asset = asset.model_copy(
        update={
            "joints": tuple(
                joint.model_copy(update={"effort_limit": 40.0}) for joint in asset.joints
            )
        }
    )
    reference = px.assets.register(asset)
    request = {
        "resource_id": "generated.scene",
        "version": "1.0.0",
        "display_name": "Layout",
        "domain": "tabletop",
        "objects": [{"name": "cabinet", "asset": reference.model_dump(mode="json")}],
        "xy_bounds_m": [-1, -1, 1, 1],
        "seed": 13,
        "yaw_range_rad": [-0.2, 0.2],
    }
    layout = px.scenes.generate(request, method="bounded_layout")
    composed = px.scenes.compose(output(layout, "scene_draft"), method="collision_constraints")
    scene = SceneSpec.model_validate_json(px.artifacts.get_bytes(output(composed, "scene_spec")))
    scene_ref = px.scenes.register(scene)
    vertices = asset_vertices(asset, px.artifacts, link="drawer")
    assert len(vertices) > 0 and vertices.shape[1] == 3
    with pytest.raises(DomainValidationError, match="collision geometry"):
        asset_vertices(asset, px.artifacts, link="absent")
    compiled = px.tasks.compile(
        "打开抽屉",
        scene_ref,
        method="articulated_templates",
        config={
            "resource_id": "generated.task",
            "entity": "cabinet",
            "joint": asset.joints[0].name,
            "target_fraction": 0.7,
            "max_steps": 100,
        },
    )
    task = TaskSpec.model_validate_json(px.artifacts.get_bytes(output(compiled, "task_spec")))
    task_ref = px.tasks.register(task)
    assert px.tasks.validate(task_ref, method="joint_constraints").outputs
    backend = {
        "controlled_joints": task.metadata["controlled_joints"],
        "initial_positions": task.metadata["initial_positions"],
        "effort_limits": task.metadata["effort_limits"],
        "timestep_seconds": 0.002,
        "substeps": 25,
    }
    request = {
        "schema": "physrealx.task-instance/1",
        "task": task_ref.model_dump(mode="json"),
        "backend_config": backend,
        "target_positions": task.metadata["target_positions"],
    }
    source = px.artifacts.put_json(request)
    trajectory = px.trajectories.generate(
        source,
        method="mujoco_joint_trajectory",
        config={"duration_seconds": 2.0, "settle_seconds": 1.0, "max_tracking_error": 0.1},
    )
    assert trajectory.metrics["steps"] == 60
    plan = json.loads(px.artifacts.get_bytes(output(trajectory, "trajectory")))
    np.testing.assert_allclose(plan["commands"][-1], task.metadata["target_positions"], atol=1e-6)
    assert abs(plan["observed_positions"][-1][0] - plan["commands"][-1][0]) < 0.01
    with pytest.raises(MethodExecutionError, match="tracking_error"):
        px.trajectories.generate(
            source,
            method="mujoco_joint_trajectory",
            config={"duration_seconds": 2.0, "settle_seconds": 1.0, "max_tracking_error": 1e-8},
        )
    with pytest.raises(DomainValidationError, match="horizon"):
        px.trajectories.generate(
            source, method="mujoco_joint_trajectory", config={"duration_seconds": 100.0}
        )
    with pytest.raises(DomainValidationError, match="bounds"):
        px.trajectories.generate(
            px.artifacts.put_json({**request, "target_positions": [20]}),
            method="mujoco_joint_trajectory",
        )


def test_layout_infeasibility_collisions_and_ambiguous_instructions_are_rejected(px: PhysRealX):
    asset, _ = assembled_asset(px)
    reference = px.assets.register(asset)
    request = {
        "resource_id": "infeasible.scene",
        "version": "1.0.0",
        "display_name": "Layout",
        "domain": "shelf",
        "objects": [{"name": "cabinet", "asset": reference.model_dump(mode="json")}],
        "xy_bounds_m": [0, 0, 0.01, 0.01],
        "max_attempts": 2,
        "seed": 13,
    }
    with pytest.raises(MethodExecutionError, match="no feasible"):
        px.scenes.generate(request, method="bounded_layout")
    scene = SceneSpec(
        resource_id="colliding.scene",
        version="1.0.0",
        display_name="Collision",
        entities=tuple(
            SceneEntitySpec(name=name, asset=reference, role="object") for name in ("one", "two")
        ),
    )
    source = px.artifacts.put_json(
        {"schema": "physrealx.scene-draft/1", "scene_spec": scene.model_dump(mode="json")}
    )
    with pytest.raises(DomainValidationError, match="collision"):
        px.scenes.compose(source, method="collision_constraints")
    scene_ref = px.scenes.register(scene)
    for instruction in ("不要打开抽屉", "打开然后关闭", "pick up cabinet"):
        with pytest.raises(DomainValidationError, match="unambiguously"):
            px.tasks.compile(
                instruction,
                scene_ref,
                method="articulated_templates",
                config={"resource_id": "bad.task", "entity": "one", "joint": asset.joints[0].name},
            )


def test_run_can_record_post_success_dynamics_without_ignoring_termination(px: PhysRealX):
    asset, _ = assembled_asset(px)
    scene, task, config = scene_task(px, asset)
    backend = px.methods.bind(MethodCapability.SIMULATION_RUN, "mujoco", config=config)
    policy = px.methods.bind(
        MethodCapability.POLICY_INFERENCE, "joint_target", config={"target": [0.3]}
    )
    for stop_on_success in (True, False):
        record = px.runs.submit(
            RunSpec(
                scene=scene.reference(),
                task=task.reference(),
                backend=backend,
                policy=policy,
                episode_count=1,
                stop_on_success=stop_on_success,
                base_seed=13,
                record_signals=tuple(signal.name for signal in task.observations),
                dataset_id=f"duration.{int(stop_on_success)}",
                dataset_version="1.0.0",
            )
        )
        episode = px.runs.episodes(record)[0]
        assert episode.success and episode.steps[-1].is_terminal
        if stop_on_success:
            assert len(episode.steps) < task.max_steps + 1
        else:
            assert len(episode.steps) == task.max_steps + 1
    terminated = task.model_copy(
        update={
            "resource_id": "terminated.task",
            "termination_conditions": (
                Predicate(signal="joint.position", operator=PredicateOperator.GE, value=0.15),
            ),
        }
    )
    task_ref = px.tasks.register(terminated)
    record = px.runs.submit(
        RunSpec(
            scene=scene.reference(),
            task=task_ref,
            backend=backend,
            policy=policy,
            episode_count=1,
            stop_on_success=False,
            base_seed=13,
            record_signals=tuple(signal.name for signal in task.observations),
            dataset_id="terminated",
            dataset_version="1.0.0",
        )
    )
    episode = px.runs.episodes(record)[0]
    assert not episode.success and len(episode.steps) < task.max_steps + 1
    assert episode.failure_reason == "termination_condition"


def test_contact_task_goal_is_passive_object_not_robot_actuation(px: PhysRealX):
    asset, _ = assembled_asset(px)
    asset = asset.model_copy(
        update={"joints": tuple(j.model_copy(update={"effort_limit": 40.0}) for j in asset.joints)}
    )
    reference = px.assets.register(asset)
    scene = SceneSpec(
        resource_id="contact.scene",
        version="1.0.0",
        display_name="Contact",
        entities=(
            SceneEntitySpec(name="cabinet", asset=reference, role=SceneEntityRole.OBJECT),
            SceneEntitySpec(name="robot", asset=reference, role=SceneEntityRole.ROBOT),
        ),
        metadata={
            "robot_interaction": {
                "robot": "robot",
                "entity": "cabinet",
                "joint": "slide",
                "controlled_joints": ["slide"],
                "initial_positions": [0.0],
                "waypoints": [[0.15]],
                "target_positions": [0.3],
            }
        },
    )
    scene_ref = px.scenes.register(scene)
    config = {
        "resource_id": "contact.task",
        "entity": "cabinet",
        "joint": "slide",
        "initial_fraction": 0.7,
        "robot": "robot",
    }
    compiled = px.tasks.compile(
        "关闭抽屉", scene_ref, method="articulated_templates", config=config
    )
    task = TaskSpec.model_validate_json(px.artifacts.get_bytes(output(compiled, "task_spec")))
    assert task.metadata["controlled_joints"] == ["robot/slide"]
    assert task.metadata["waypoints"] == [[0.15]]
    assert task.success_conditions[0].signal == "joint.cabinet.slide.position"
    assert task.initial_conditions[0].signal == "joint.cabinet.slide.position"
    assert task.metadata["passive_positions"]["cabinet/slide"] == pytest.approx(0.35)
    config["robot"] = "cabinet"
    with pytest.raises(DomainValidationError, match="robot"):
        px.tasks.compile("关闭抽屉", scene_ref, method="articulated_templates", config=config)


def test_collision_welding_joins_material_boundaries(tmp_path):
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    box = trimesh.creation.box(extents=[0.2, 0.15, 0.1])
    scene = trimesh.Scene()
    for index, faces in enumerate(np.array_split(box.faces, 2)):
        mesh = trimesh.Trimesh(vertices=box.vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.full((len(mesh.vertices), 2), index))
        scene.add_geometry(mesh, geom_name=f"material_{index}")
    source = px.artifacts.put_bytes(scene.export(file_type="glb"), media_type="model/gltf-binary")
    operation = px.assets.build_collision(
        source,
        method="coacd",
        config={
            "source_unit": "m",
            "link": "body",
            "weld_tolerance_m": 0.000001,
            "preprocessing": "strict",
            "max_parts": 1,
        },
    )
    report = json.loads(px.artifacts.get_bytes(operation.output("collision_build_report")))
    assert report["preprocessing"]["source_closed_positive_volume"]
    assert operation.metrics["convex_parts"] == 1
    assert operation.metrics["relative_volume_error"] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("thickness,passes", [(0.002, True), (0.1, False)])
def test_open_collision_surface_uses_distance_not_undefined_volume(
    tmp_path, monkeypatch, thickness, passes
):
    import coacd

    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    closed = trimesh.creation.box(extents=[0.2, 0.2, thickness])
    top = np.all(closed.vertices[closed.faces, 2] > 0, axis=1)
    opened = trimesh.Trimesh(vertices=closed.vertices, faces=closed.faces[~top], process=False)
    assert not opened.is_watertight
    source = px.artifacts.put_bytes(opened.export(file_type="glb"), media_type="model/gltf-binary")
    monkeypatch.setattr(
        coacd, "run_coacd", lambda *args, **kwargs: [(closed.vertices, closed.faces)]
    )
    config = {"source_unit": "m", "link": "body", "preprocessing": "coacd", "threshold_m": 0.005}
    if not passes:
        with pytest.raises(MethodExecutionError, match="surface P95"):
            px.assets.build_collision(source, method="coacd", config=config)
        return
    operation = px.assets.build_collision(source, method="coacd", config=config)
    report = json.loads(px.artifacts.get_bytes(operation.output("collision_build_report")))
    assert "relative_volume_error" not in operation.metrics
    assert operation.metrics["bidirectional_surface_p95_m"] <= 0.005
    assert report["preprocessing"]["volume_comparison_scope"] == "not_applicable_open_surface"
    assert "volume_error" not in report["gates"]
    assert report["gates"]["surface_error"]


def test_proxy_surface_samples_exclude_internal_partition_faces():
    from physrealx.integrations.collision import _exterior_samples

    parts = [trimesh.creation.box(extents=[0.1, 0.2, 0.2]) for _ in range(2)]
    parts[0].apply_translation([-0.05, 0, 0])
    parts[1].apply_translation([0.05, 0, 0])
    samples = _exterior_samples(parts, trimesh.util.concatenate(parts), 7)
    outer = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    distance = trimesh.proximity.closest_point(outer, samples)[1]
    assert distance.max() < 1e-9


@pytest.mark.gpu
@pytest.mark.parametrize(
    "joint_kind,blocked", [("prismatic", False), ("revolute", False), ("prismatic", True)]
)
def test_isaac_sim_real_dynamics_reset_units_poses_and_contact(
    px: PhysRealX,
    joint_kind: str,
    blocked: bool,
) -> None:
    """Opt-in native-engine test; never substitutes mocks for PhysX dynamics."""
    import os

    from scipy.spatial.transform import Rotation

    from physrealx.domain import JointType, PoseSpec
    from physrealx.integrations.isaac_sim import IsaacSimBackend

    if os.environ.get("PHYSREALX_TEST_ISAAC_SIM") != "1":
        pytest.skip("set PHYSREALX_TEST_ISAAC_SIM=1 with Isaac Sim 4.5 and a usable GPU")
    asset, _ = assembled_asset(px, textured=True)
    unit = "m" if joint_kind == "prismatic" else "rad"
    if joint_kind == "revolute":
        asset = AssetSpec.model_validate(
            {
                **asset.model_dump(),
                "joints": [
                    asset.joints[0].model_copy(
                        update={
                            "joint_type": JointType.REVOLUTE,
                            "upper_limit": 1.2,
                        }
                    )
                ],
            }
        )
    scene, task, config = scene_task(px, asset)
    pose = PoseSpec(
        translation_m=(0.3, -0.2, 0.5),
        rotation_xyzw=Rotation.from_euler("z", 30, degrees=True).as_quat(),
    )
    entity = scene.entities[0].model_copy(update={"pose": pose})
    entities = [entity]
    assets = {entity.name: asset}
    if blocked:
        obstacle = AssetSpec.model_validate(
            {
                **asset.model_dump(),
                "resource_id": "fixture.obstacle",
                "root_link": "drawer",
                "links": ["drawer"],
                "joints": [],
                "contact_exclusions": [],
                "geometries": [g for g in asset.geometries if g.link == "drawer"],
                "physics": [p for p in asset.physics if p.link == "drawer"],
            }
        )
        position = np.array(pose.translation_m) + Rotation.from_quat(pose.rotation_xyzw).apply(
            [0.25, 0, 0]
        )
        entities.append(
            SceneEntitySpec(
                name="obstacle",
                asset=obstacle.reference(),
                role=SceneEntityRole.ENVIRONMENT,
                pose=pose.model_copy(update={"translation_m": tuple(position)}),
            )
        )
        assets["obstacle"] = obstacle
    scene = scene.model_copy(update={"entities": tuple(entities)})
    count = sum(len(a.links) for a in assets.values())
    observations = (
        *(
            s.model_copy(update={"unit": unit + ("/s" if s.name.endswith("velocity") else "")})
            for s in task.observations
        ),
        SignalSpec(
            name="body.position",
            shape=(count, 3),
            dtype=SignalDType.FLOAT32,
            unit="m",
            frame="world",
        ),
        SignalSpec(
            name="body.orientation",
            shape=(count, 4),
            dtype=SignalDType.FLOAT32,
            unit="1",
            frame="world",
        ),
        *(
            SignalSpec(
                name="body." + name,
                shape=(count, 3),
                dtype=SignalDType.FLOAT32,
                unit=unit,
                frame="world",
            )
            for name, unit in (
                ("external_force", "N"),
                ("external_torque", "N*m"),
                ("wrench_origin", "m"),
            )
        ),
        SignalSpec(
            name="contact.force", shape=(1,), dtype=SignalDType.FLOAT32, unit="N", frame="joint"
        ),
    )
    task = task.model_copy(
        update={
            "observations": observations,
            "action": task.action.model_copy(
                update={
                    "signal": task.action.signal.model_copy(update={"unit": unit}),
                }
            ),
        }
    )
    config.update(
        timeout_seconds=600, position_gain=50.0, velocity_gain=5.0, initial_positions=[0.05]
    )
    if joint_kind == "revolute":
        config.update(position_gain=20.0, velocity_gain=0.1, timestep_seconds=0.0005, substeps=100)
    session = IsaacSimBackend(px.artifacts).create_session(scene, assets, task, config)
    from physrealx.domain import DataValue

    action = DataValue(
        dtype=task.action.signal.dtype, shape=(1,), unit=unit, frame="joint", inline=(0.3,)
    )
    from physrealx.errors import InvalidStateTransitionError

    try:
        with pytest.raises(InvalidStateTransitionError, match="reset"):
            session.step(action)
        trajectories = []
        peaks = []
        for _ in range(2):
            frame = session.reset(23)
            assert frame.observations["joint.position"].inline[0] == pytest.approx(0.05, abs=1e-6)
            positions = np.array(frame.observations["body.position"].inline).reshape(count, 3)
            np.testing.assert_allclose(positions[0], pose.translation_m, atol=1e-5)
            samples, forces, opposing_forces = [], [], []
            for step_index in range(60):
                frame = session.step(action).frame
                samples.append(frame.observations["joint.position"].inline[0])
                forces.append(frame.observations["contact.force"].inline[0])
                loads = np.array(frame.observations["body.external_force"].inline).reshape(count, 3)
                np.testing.assert_allclose(loads.sum(axis=0), 0, atol=1e-6)
                moment = np.array(frame.observations["body.external_torque"].inline).reshape(
                    count, 3
                )
                origins = np.array(frame.observations["body.wrench_origin"].inline).reshape(
                    count, 3
                )
                np.testing.assert_allclose(
                    (moment + np.cross(origins, loads)).sum(axis=0), 0, atol=1e-6
                )
                opposing_forces.append(
                    float(
                        loads[1]
                        @ Rotation.from_quat(pose.rotation_xyzw).apply(asset.joints[0].axis)
                    )
                )
                assert frame.observations["body.position"].timestamp_seconds == pytest.approx(
                    (step_index + 1) * 0.05
                )

            # Compare recorded link poses with FK, independently of the engine joint readout.
            def transform(pose_spec: PoseSpec) -> object:
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat(pose_spec.rotation_xyzw).as_matrix()
                matrix[:3, 3] = pose_spec.translation_m
                return matrix

            motion = np.eye(4)
            axis = np.array(asset.joints[0].axis)
            if joint_kind == "prismatic":
                motion[:3, 3] = axis * samples[-1]
            else:
                motion[:3, :3] = Rotation.from_rotvec(axis * samples[-1]).as_matrix()
            expected = (
                transform(pose)
                @ transform(asset.root_pose)
                @ transform(asset.joints[0].parent_to_joint)
                @ motion
            )
            positions = np.array(frame.observations["body.position"].inline).reshape(count, 3)
            rotations = np.array(frame.observations["body.orientation"].inline).reshape(count, 4)
            moving_index = asset.links.index(asset.joints[0].child)
            np.testing.assert_allclose(positions[moving_index], expected[:3, 3], atol=1e-4)
            np.testing.assert_allclose(
                Rotation.from_quat(rotations[moving_index]).as_matrix(), expected[:3, :3], atol=1e-4
            )
            if blocked:
                assert min(opposing_forces) < -1.0
            trajectories.append(samples)
            peaks.append(max(forces))
        np.testing.assert_allclose(trajectories[0], trajectories[1], atol=1e-5)
        if blocked:
            assert trajectories[0][-1] < 0.2
            assert min(peaks) > 1.0
        else:
            assert trajectories[0][-1] == pytest.approx(0.3, abs=0.025)
        # Illegal commands must leave a usable session, and reset must restore passive state.
        with pytest.raises(DomainValidationError, match="bounds"):
            session.step(action.model_copy(update={"inline": (0.9,)}))
        assert session.reset(23).observations["joint.position"].inline[0] == pytest.approx(
            0.05, abs=1e-6
        )
    finally:
        session.close()
    with pytest.raises(InvalidStateTransitionError, match="closed"):
        session.reset(23)


@pytest.mark.gpu
def test_isaac_sim_publishes_replayable_episodes(px: PhysRealX) -> None:
    import os

    if os.environ.get("PHYSREALX_TEST_ISAAC_SIM") != "1":
        pytest.skip("set PHYSREALX_TEST_ISAAC_SIM=1 with Isaac Sim 4.5 and a usable GPU")
    asset, _ = assembled_asset(px)
    scene, task, config = scene_task(px, asset)
    config["timeout_seconds"] = 600
    run = px.simulations.run(
        scene.reference(),
        task.reference(),
        backend="isaac_sim",
        config=config,
        policy=px.methods.bind(
            MethodCapability.POLICY_INFERENCE, "joint_target", config={"target": [0.3]}
        ),
        dataset_id="isaac.integration.dataset",
        episode_count=2,
    )
    assert run.status is RunStatus.SUCCEEDED
    episodes = px.runs.episodes(run)
    assert len(episodes) == 2 and all(episode.success for episode in episodes)
    assert all(episode.backend.name == "isaac_sim" for episode in episodes)
    assert all(episode.steps[-1].action is None for episode in episodes)
    assert all(episode.steps[-1].state["joint.position"].inline[0] >= 0.25 for episode in episodes)
    assert episodes[0].metadata["sensor_metadata"]["physics_engine"] == "PhysX / Isaac Sim 4.5.0"
    assert px.runs.submit(run.spec) == run


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["mujoco", "isaac_sim"])
def test_native_multimodal_backends_randomization_and_wrenches(
    px: PhysRealX, backend: str
) -> None:
    """One task/config exercises native sensors and known loads in both real engines."""
    import os

    from physrealx.domain import DataValue, PoseSpec
    from physrealx.integrations.sensors import CameraSpec, project_depth, with_sensor_signals

    if backend == "isaac_sim" and os.environ.get("PHYSREALX_TEST_ISAAC_SIM") != "1":
        pytest.skip("requires Isaac Sim 4.5 and a GPU")
    asset, _ = assembled_asset(px, textured=True)
    scene, task, config = scene_task(px, asset)
    camera = CameraSpec(pose=PoseSpec(translation_m=(0, 0, 1)), width=64, height=48)
    task = with_sensor_signals(
        task, {"front": camera}, link_count=2, resource_id="native.camera.task", frequency_hz=20
    )
    config.update(
        cameras={"front": camera.model_dump(mode="json")},
        external_wrenches={"asset/drawer": {"force_n": [0, 0, 2], "torque_nm": [0, 0.1, 0]}},
        randomization={
            "mass_scale": [0.8, 1.2],
            "damping_scale": [0.8, 1.2],
            "friction_scale": [0.8, 1.2],
            "initial_position_jitter": 0.01,
            "color_jitter": 0.1,
            "light_jitter": 0.2,
            "material_shininess_jitter": 0.1,
            "material_specular_jitter": 0.02,
            "camera_position_jitter_m": 0.01,
            "occlusion_probability": 1.0,
        },
    )
    if backend == "isaac_sim":
        config["timeout_seconds"] = 600
    from physrealx.errors import MethodExecutionError

    try:
        session = px.plugins.backend(backend).create_session(scene, {"asset": asset}, task, config)
    except MethodExecutionError as error:
        pytest.fail(f"{error}: {error.details}")

    def pixels(frame, modality):
        value = frame.observations[f"camera.front.{modality}"]
        return np.load(io.BytesIO(px.artifacts.get_bytes(value.artifact)), allow_pickle=False)

    action = DataValue(dtype="float32", shape=(1,), unit="m", frame="joint", inline=(0.1,))
    try:
        first = session.reset(7)
        first_result = session.step(action).frame
        second = session.reset(8)
        replay = session.reset(7)
        replay_result = session.step(action).frame
        np.testing.assert_allclose(
            first.observations["joint.position"].inline,
            replay.observations["joint.position"].inline,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            first_result.observations["joint.position"].inline,
            replay_result.observations["joint.position"].inline,
            atol=1e-5,
        )
        np.testing.assert_allclose(pixels(first, "position"), pixels(replay, "position"), atol=1e-6)
        assert not np.array_equal(pixels(first, "position"), pixels(second, "position"))
        assert not np.array_equal(pixels(first, "rgb"), pixels(second, "rgb"))
        assert np.count_nonzero(pixels(first, "depth")) > 100
        assert np.ptp(pixels(first, "rgb")) > 50
        assert not pixels(first, "events").any()
        np.testing.assert_allclose(
            pixels(first, "pointcloud"),
            project_depth(pixels(first, "depth"), pixels(first, "intrinsics")),
        )
        labels = first.metadata["segmentation_labels"]
        visible = {labels[str(int(v))] for v in np.unique(pixels(first, "mask"))}
        assert "asset/drawer" in visible
        assert "occluder/front" in visible
        # Check raster calibration against authored geometry, not merely XYZ=unproject(depth).
        from scipy.spatial.transform import Rotation

        drawer_id = next(int(i) for i, name in labels.items() if name == "asset/drawer")
        optical = pixels(first, "pointcloud")[pixels(first, "mask") == drawer_id]
        world = Rotation.from_quat(pixels(first, "orientation")).apply(optical) + pixels(
            first, "position"
        )
        q = first.observations["joint.position"].inline[0]
        assert np.max(np.abs(world[:, 0] - q)) <= 0.06
        assert np.max(np.abs(world[:, 1])) <= 0.06
        np.testing.assert_allclose(world[:, 2], 0.05, atol=0.002)

        force = np.array(first_result.observations["body.external_force"].inline).reshape(2, 3)
        torque = np.array(first_result.observations["body.external_torque"].inline).reshape(2, 3)
        np.testing.assert_allclose(force[-1], [0, 0, 2], atol=1e-4)
        # MuJoCo reports torque about root-subtree COM; translate it to body COM for comparison.
        origins = np.array(first_result.observations["body.wrench_origin"].inline).reshape(2, 3)
        if backend == "mujoco":
            com = session.data.xipos[session.model.body("asset/drawer").id]
        else:
            com = origins[-1]
            assert first.metadata["randomization"] == replay.metadata["randomization"]
            assert first.metadata["randomization"] != second.metadata["randomization"]
            assert (
                first.metadata["randomization"]["runtime_body_masses_kg"]
                != second.metadata["randomization"]["runtime_body_masses_kg"]
            )
        np.testing.assert_allclose(
            torque[-1] + np.cross(origins[-1] - com, force[-1]), [0, 0.1, 0], atol=1e-4
        )
        assert all(
            v.timestamp_seconds == pytest.approx(session.dt_seconds)
            for v in first_result.observations.values()
        )
    finally:
        session.close()
