from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.digital_twin import (
    ObservationModality,
    ObservationSetManifest,
    ObservationStateRole,
    ObservationStream,
)
from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    MethodSpec,
    OperationSpec,
    VersionedRef,
)
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX
from physrealx.readiness import (
    EvidenceInput,
    LifecycleEvidence,
    LifecycleEvidenceKey,
    LifecycleReadinessReport,
    LifecycleStage,
    ReadinessBlockerCode,
    ReadinessStatus,
)

_CAPABILITIES: Mapping[LifecycleStage, MethodCapability] = {
    LifecycleStage.INGESTION: MethodCapability.OBSERVATION_IMPORT,
    LifecycleStage.CAMERA: MethodCapability.CAMERA_RECOVER,
    LifecycleStage.METRIC_ALIGNMENT: MethodCapability.CAMERA_ALIGN_METRIC,
    LifecycleStage.DEPTH: MethodCapability.DEPTH_ENHANCE,
    LifecycleStage.GEOMETRY: MethodCapability.GEOMETRY_RECONSTRUCT,
    LifecycleStage.ARTICULATION: MethodCapability.ARTICULATION_RECOVER,
    LifecycleStage.COLLISION: MethodCapability.COLLISION_BUILD,
    LifecycleStage.PHYSICS: MethodCapability.PHYSICS_CALIBRATE,
    LifecycleStage.ASSEMBLY: MethodCapability.ASSET_ASSEMBLE,
    LifecycleStage.USD: MethodCapability.ASSET_PACKAGE,
    LifecycleStage.PHYSICAL_VALIDATION: MethodCapability.ASSET_VALIDATE,
    LifecycleStage.DATASET: MethodCapability.DATASET_INGEST,
    LifecycleStage.POLICY: MethodCapability.POLICY_TRAIN,
    LifecycleStage.WORLD_MODEL: MethodCapability.WORLD_MODEL_TRAIN,
    LifecycleStage.BENCHMARK: MethodCapability.EVALUATION_RUN,
    LifecycleStage.ROS_DEPLOY: MethodCapability.DEPLOYMENT_STAGE,
    LifecycleStage.FAILURE: MethodCapability.FAILURE_ANALYZE,
    LifecycleStage.FEEDBACK: MethodCapability.FEEDBACK_BUILD_DATASET,
}


def _methods() -> dict[LifecycleStage | str, MethodSpec]:
    methods: dict[LifecycleStage | str, MethodSpec] = {}
    for stage, capability in _CAPABILITIES.items():
        methods[stage] = builtin_method(
            capability,
            f"gate-{stage.value.replace('_', '-')}",
            input_contract=f"{stage.value}.input",
            output_contract=f"{stage.value}.output",
        ).bind()
    return methods


class _ReadyAdapter:
    def __init__(self, method: MethodSpec, ready: bool = True) -> None:
        self.descriptor = method.descriptor
        self._ready = ready

    def probe(self) -> MethodProbe:
        return MethodProbe(self._ready, "adapter ready" if self._ready else "runtime unavailable")

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        del spec, artifacts
        return MethodExecutionOutput(outputs=(), metrics={})


def _register_adapters(
    platform: PhysRealX,
    methods: Mapping[LifecycleStage | str, MethodSpec],
) -> None:
    for method in methods.values():
        platform.plugins.register_adapter(_ReadyAdapter(method))


def _body_parameter(value: float, unit: str, evidence_artifact_id: str) -> dict[str, object]:
    return {"value": value, "unit": unit, "evidence_artifact_id": evidence_artifact_id}


def _joint_parameter(value: float, unit: str) -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "confidence_interval": {
            "level": 0.95,
            "lower": value * 0.9,
            "upper": value * 1.1,
        },
        "multi_start_spread": value * 0.001,
    }


_VALID_SEMANTIC_EVIDENCE: Mapping[LifecycleEvidenceKey, dict[str, object]] = {
    LifecycleEvidenceKey.CAMERA_RECONSTRUCTION: {
        "schema": "physrealx.camera-rig-calibration/1",
        "scale_status": "unresolved",
    },
    LifecycleEvidenceKey.CAMERA_SOLUTION: {
        "schema": "physrealx.camera-rig-calibration/1",
        "scale_status": "resolved",
    },
    LifecycleEvidenceKey.ENHANCED_DEPTH: {
        "schema": "physrealx.depth-enhancement-report/1",
        "status": "PASSED",
    },
    LifecycleEvidenceKey.ARTICULATION: {
        "schema": "physrealx.articulation-model/1",
        "status": "IDENTIFIED",
        "links": ["base", "drawer"],
        "joints": [{"name": "drawer_slide", "parent": "base", "child": "drawer"}],
    },
    LifecycleEvidenceKey.BODY_PHYSICS: {
        "schema": "physrealx.identified-body-physics/1",
        "status": "IDENTIFIED",
        "evidence_state": "identified",
        "bodies": [
            {
                "link": link,
                "mass_kg": _body_parameter(mass, "kg", artifact_id),
                "center_of_mass_m": [0.0, 0.0, 0.0],
                "inertia_kg_m2": [
                    [0.08, 0.0, 0.0],
                    [0.0, 0.09, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "static_friction": _body_parameter(0.35, "1", artifact_id),
                "dynamic_friction": _body_parameter(0.28, "1", artifact_id),
                "restitution": _body_parameter(0.02, "1", artifact_id),
                "evidence_artifact_id": artifact_id,
            }
            for link, mass, artifact_id in (
                ("base", 8.0, f"art_{'a' * 32}"),
                ("drawer", 2.0, f"art_{'b' * 32}"),
            )
        ],
    },
    LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS: {
        "schema": "physrealx.identified-joint-physics/1",
        "status": "IDENTIFIED",
        "evidence_state": "identified",
        "joint": {"name": "drawer_slide", "type": "slide"},
        "parameters": {
            "effective_inertia": _joint_parameter(2.3, "kg"),
            "viscous_damping": _joint_parameter(0.7, "N*s/m"),
            "coulomb_friction": _joint_parameter(0.24, "N"),
        },
        "evidence_artifact_id": f"art_{'c' * 32}",
    },
    LifecycleEvidenceKey.USD_ASSET: {
        "schema": "physrealx.asset-package-manifest/1",
        "roundtrip_status": "PASSED",
        "self_contained": True,
    },
    LifecycleEvidenceKey.VALIDATED_ASSET: {
        "schema": "physrealx.asset-validation-report/1",
        "status": "PASSED",
    },
    LifecycleEvidenceKey.EPISODES: {
        "schema": "physrealx.verified-episode-collection/1",
        "status": "FINALIZED",
    },
    LifecycleEvidenceKey.DATASET: {
        "schema": "physrealx.dataset-snapshot/1",
        "status": "VALIDATED",
    },
    LifecycleEvidenceKey.POLICY_MODEL: {
        "schema": "physrealx.policy-model-bundle/1",
        "status": "TRAINED",
    },
    LifecycleEvidenceKey.WORLD_MODEL: {
        "schema": "physrealx.world-model-bundle/1",
        "status": "TRAINED",
    },
    LifecycleEvidenceKey.BENCHMARK_SUITE: {
        "schema": "physrealx.benchmark-suite/1",
        "status": "FROZEN",
    },
    LifecycleEvidenceKey.BENCHMARK_REPORT: {
        "schema": "physrealx.benchmark-report/1",
        "status": "PASSED",
    },
    LifecycleEvidenceKey.DEPLOYMENT: {
        "schema": "physrealx.deployment-report/1",
        "status": "ACTIVE",
    },
    LifecycleEvidenceKey.TELEMETRY: {
        "schema": "physrealx.telemetry-manifest/1",
        "status": "COMPLETE",
    },
    LifecycleEvidenceKey.FAILURE_REPORT: {
        "schema": "physrealx.failure-report/1",
        "status": "ANALYZED",
    },
    LifecycleEvidenceKey.FEEDBACK_DATASET: {
        "schema": "physrealx.feedback-dataset/1",
        "status": "VALIDATED",
    },
}


def _metric_alignment_evidence(camera_rig_artifact_id: str) -> dict[str, object]:
    train = [
        {
            "id": f"train-{index}",
            "source_xyz": point,
            "target_xyz_m": point,
        }
        for index, point in enumerate(
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        )
    ]
    holdout = [
        {
            "id": f"holdout-{index}",
            "source_xyz": point,
            "target_xyz_m": point,
        }
        for index, point in enumerate(([0.2, 0.2, 0.0], [0.2, 0.0, 0.2], [0.0, 0.2, 0.2]))
    ]
    return {
        "schema": "physrealx.metric-alignment-evidence/1",
        "unit": "m",
        "camera_rig_artifact_id": camera_rig_artifact_id,
        "component_id": "0",
        "source_frame": "colmap_component_0",
        "target_frame": "capture_world",
        "metric_reference": {
            "kind": "surveyed_length",
            "identifier": "fixture-ruler",
            "source": "fixture-certificate",
            "value_m": 1.0,
        },
        "provenance": {
            "capture_id": "fixture-capture",
            "measurement_method": "surveyed_points",
            "source_uri": "file:///fixture/metric-evidence.json",
        },
        "train_correspondences": train,
        "holdout_correspondences": holdout,
    }


def _observation_resource(platform: PhysRealX) -> VersionedRef:
    frame = platform.artifacts.put_bytes(b"captured-rgb-frame", media_type="image/png")
    manifest = ObservationSetManifest(
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        importer="artmodeling",
        importer_version="1.0.0",
        source_uri="file:///capture/cabinet2",
        streams=(
            ObservationStream(
                name="cabinet2.rgb",
                modality=ObservationModality.RGB,
                state="closed",
                role=ObservationStateRole.RECONSTRUCTION,
                encoding="png",
                frame_count=1,
                artifacts=(frame,),
                coordinate_frame="camera",
            ),
        ),
    )
    artifact = platform.artifacts.put_json(manifest.model_dump(mode="json"))
    return platform.repository.register_resource(manifest.reference(artifact.artifact_id))


def _complete_roots(
    platform: PhysRealX,
    capture: Path,
) -> dict[LifecycleEvidenceKey | str, EvidenceInput]:
    roots: dict[LifecycleEvidenceKey | str, EvidenceInput] = {
        LifecycleEvidenceKey.CAPTURE: capture,
        LifecycleEvidenceKey.OBSERVATIONS: _observation_resource(platform),
    }
    body_measurements = {
        link: platform.artifacts.put_bytes(
            f"measured body physics: {link}".encode(),
            media_type="application/octet-stream",
        )
        for link in ("base", "drawer")
    }
    joint_measurement = platform.artifacts.put_bytes(
        b"measured drawer joint excitation and holdout",
        media_type="application/octet-stream",
    )
    semantic_documents = dict(_VALID_SEMANTIC_EVIDENCE)
    semantic_documents[LifecycleEvidenceKey.BODY_PHYSICS] = {
        "schema": "physrealx.identified-body-physics/1",
        "status": "IDENTIFIED",
        "evidence_state": "identified",
        "bodies": [
            {
                "link": link,
                "mass_kg": _body_parameter(mass, "kg", measurement.artifact_id),
                "center_of_mass_m": [0.0, 0.0, 0.0],
                "inertia_kg_m2": [
                    [0.08, 0.0, 0.0],
                    [0.0, 0.09, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "static_friction": _body_parameter(0.35, "1", measurement.artifact_id),
                "dynamic_friction": _body_parameter(0.28, "1", measurement.artifact_id),
                "restitution": _body_parameter(0.02, "1", measurement.artifact_id),
                "evidence_artifact_id": measurement.artifact_id,
            }
            for (link, measurement), mass in zip(body_measurements.items(), (8.0, 2.0))
        ],
    }
    semantic_documents[LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS] = {
        "schema": "physrealx.identified-joint-physics/1",
        "status": "IDENTIFIED",
        "evidence_state": "identified",
        "joint": {"name": "drawer_slide", "type": "slide"},
        "parameters": {
            "effective_inertia": _joint_parameter(2.3, "kg"),
            "viscous_damping": _joint_parameter(0.7, "N*s/m"),
            "coulomb_friction": _joint_parameter(0.24, "N"),
        },
        "evidence_artifact_id": joint_measurement.artifact_id,
    }
    roots.update(
        {key: platform.artifacts.put_json(document) for key, document in semantic_documents.items()}
    )
    camera_reconstruction = roots[LifecycleEvidenceKey.CAMERA_RECONSTRUCTION]
    assert isinstance(camera_reconstruction, ArtifactRef)
    roots[LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE] = platform.artifacts.put_json(
        _metric_alignment_evidence(camera_reconstruction.artifact_id)
    )
    camera_solution = roots[LifecycleEvidenceKey.CAMERA_SOLUTION]
    enhanced_depth = roots[LifecycleEvidenceKey.ENHANCED_DEPTH]
    articulation = roots[LifecycleEvidenceKey.ARTICULATION]
    body_physics = roots[LifecycleEvidenceKey.BODY_PHYSICS]
    joint_physics = roots[LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS]
    assert all(
        isinstance(item, ArtifactRef)
        for item in (camera_solution, enhanced_depth, articulation, body_physics, joint_physics)
    )
    assert isinstance(camera_solution, ArtifactRef)
    assert isinstance(enhanced_depth, ArtifactRef)
    assert isinstance(articulation, ArtifactRef)
    visual_meshes = {
        link: platform.artifacts.put_bytes(
            f"o {link}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".encode(),
            media_type="model/obj",
        )
        for link in ("base", "drawer")
    }
    collision_meshes = {
        link: platform.artifacts.put_bytes(
            f"glTF collision fixture: {link}".encode(),
            media_type="model/gltf-binary",
        )
        for link in ("base", "drawer")
    }
    visual_geometry = platform.artifacts.put_json(
        {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "status": "PASSED",
            "unit": "m",
            "source_camera_artifact_id": camera_solution.artifact_id,
            "source_depth_artifact_id": enhanced_depth.artifact_id,
            "method": {
                "name": "fixture_tsdf",
                "version": "1.0.0",
                "source_revision": "fixture-1.0.0",
            },
            "metrics": {"surface_vertices": 6.0, "link_count": 2.0},
            "gates": {"metric_scale": True, "nonempty_meshes": True},
            "meshes": [
                {"link": link, "artifact": artifact.model_dump(mode="json")}
                for link, artifact in visual_meshes.items()
            ],
        }
    )
    roots[LifecycleEvidenceKey.VISUAL_GEOMETRY] = visual_geometry
    collision_geometry = platform.artifacts.put_json(
        {
            "schema": "physrealx.collision-build-report/1",
            "status": "PASSED",
            "unit": "m",
            "link": "drawer",
            "method": {
                "name": "coacd",
                "version": "1.0.14",
                "source_revision": "v1.0.14",
            },
            "source_geometry": visual_meshes["drawer"].model_dump(mode="json"),
            "collision_geometry": collision_meshes["drawer"].model_dump(mode="json"),
            "metrics": {"convex_parts": 2.0},
            "gates": {
                "source_mesh_nonempty": True,
                "decomposition_nonempty": True,
                "output_mesh_nonempty": True,
                "finite_metrics": True,
                "metric_unit_declared": True,
            },
        }
    )
    roots[LifecycleEvidenceKey.COLLISION_GEOMETRY] = collision_geometry
    body_document = semantic_documents[LifecycleEvidenceKey.BODY_PHYSICS]
    joint_document = semantic_documents[LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS]
    assert isinstance(body_document["bodies"], list)
    assert isinstance(joint_document["joint"], dict)
    full_physics = platform.artifacts.put_json(
        {
            "schema": "physrealx.identified-physics/1",
            "status": "IDENTIFIED",
            "evidence_state": "identified",
            "bodies": body_document["bodies"],
            "joints": [
                {
                    **joint_document["joint"],
                    "parameters": joint_document["parameters"],
                    "evidence_artifact_id": joint_document["evidence_artifact_id"],
                }
            ],
        }
    )
    scale_evidence = platform.artifacts.put_bytes(
        b"surveyed-metric-scale",
        media_type="application/octet-stream",
    )
    articulation_evidence = platform.artifacts.put_bytes(
        b"articulation-observations",
        media_type="application/octet-stream",
    )
    physics_evidence = platform.artifacts.put_bytes(
        b"body-and-joint-measurements",
        media_type="application/octet-stream",
    )
    roots[LifecycleEvidenceKey.ASSET_ASSEMBLY] = platform.artifacts.put_json(
        {
            "schema": "physrealx.asset-assembly-bundle/1",
            "resource_id": "cabinet2.asset",
            "version": "1.0.0",
            "metric_scale": {
                "status": "resolved",
                "meters_per_unit": 1.0,
                "evidence_artifact_id": scale_evidence.artifact_id,
            },
            "up_axis": "Z",
            "root_link": "base",
            "visual_meshes": [
                {"link": link, "artifact": artifact.model_dump(mode="json"), "units": "meters"}
                for link, artifact in visual_meshes.items()
            ],
            "collision_meshes": [
                {"link": link, "artifact": artifact.model_dump(mode="json"), "units": "meters"}
                for link, artifact in collision_meshes.items()
            ],
            "articulation_model": articulation.model_dump(mode="json"),
            "identified_physics": full_physics.model_dump(mode="json"),
            "evidence": [
                scale_evidence.model_dump(mode="json"),
                camera_solution.model_dump(mode="json"),
                articulation_evidence.model_dump(mode="json"),
                physics_evidence.model_dump(mode="json"),
                body_physics.model_dump(mode="json"),
                joint_physics.model_dump(mode="json"),
                *[
                    measurement.model_dump(mode="json")
                    for measurement in body_measurements.values()
                ],
                joint_measurement.model_dump(mode="json"),
            ],
        }
    )
    auxiliary = platform.artifacts.put_bytes(
        b"immutable-source-evidence",
        media_type="application/octet-stream",
    )
    urdf = platform.artifacts.put_bytes(b"<robot name='fixture'/>", media_type="application/xml")
    controller = platform.artifacts.put_bytes(
        b"controller_manager: {}\n",
        media_type="application/yaml",
    )
    robot_target = platform.artifacts.put_json(
        {
            "schema": "physrealx.robot-interface-spec/1",
            "status": "DECLARED",
            "schema_version": 1,
            "robot_id": "cabinet-arm",
            "interface_version": "1.0.0",
            "joint_names": ["drawer_slide"],
            "base_frame": "robot/base_link",
            "tool_frame": "robot/tool0",
            "urdf": urdf.model_dump(mode="json"),
            "controller_configuration": controller.model_dump(mode="json"),
            "state_interface": "/joint_states",
            "command_interface": "/joint_commands",
            "requires_ros2": True,
            "requires_hardware": True,
        }
    )
    roots.update(
        {
            LifecycleEvidenceKey.PHYSICS_EVIDENCE: auxiliary,
            LifecycleEvidenceKey.VALIDATION_EVIDENCE: auxiliary,
            LifecycleEvidenceKey.ROBOT_TARGET: robot_target,
        }
    )
    return roots


def test_plan_is_complete_ordered_and_capability_bound(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()

    plan = platform.readiness.plan(methods)

    assert tuple(item.stage for item in plan.stages) == tuple(LifecycleStage)
    assert all(item.capability is _CAPABILITIES[item.stage] for item in plan.stages)
    assert plan.stages[0].requires == (LifecycleEvidenceKey.CAPTURE,)
    metric_stage = plan.stages[tuple(LifecycleStage).index(LifecycleStage.METRIC_ALIGNMENT)]
    assert metric_stage.requires == (
        LifecycleEvidenceKey.CAMERA_RECONSTRUCTION,
        LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE,
    )
    assert metric_stage.produces == (LifecycleEvidenceKey.CAMERA_SOLUTION,)
    geometry_stage = plan.stages[tuple(LifecycleStage).index(LifecycleStage.GEOMETRY)]
    assert geometry_stage.requires == (
        LifecycleEvidenceKey.CAMERA_SOLUTION,
        LifecycleEvidenceKey.ENHANCED_DEPTH,
    )
    assert geometry_stage.produces == (LifecycleEvidenceKey.VISUAL_GEOMETRY,)
    collision_stage = plan.stages[tuple(LifecycleStage).index(LifecycleStage.COLLISION)]
    assert collision_stage.requires == (
        LifecycleEvidenceKey.VISUAL_GEOMETRY,
        LifecycleEvidenceKey.ARTICULATION,
    )
    assembly_stage = plan.stages[tuple(LifecycleStage).index(LifecycleStage.ASSEMBLY)]
    assert assembly_stage.produces == (LifecycleEvidenceKey.ASSET_ASSEMBLY,)
    usd_stage = plan.stages[tuple(LifecycleStage).index(LifecycleStage.USD)]
    assert usd_stage.requires == (LifecycleEvidenceKey.ASSET_ASSEMBLY,)
    assert plan.stages[-1].produces == (LifecycleEvidenceKey.FEEDBACK_DATASET,)

    methods[LifecycleStage.CAMERA] = builtin_method(
        MethodCapability.DEPTH_ENHANCE,
        "wrong-camera",
        input_contract="observations",
        output_contract="camera",
    ).bind()
    with pytest.raises(ValidationError, match=r"camera\.recover"):
        platform.readiness.plan(methods)


def test_plan_rejects_missing_stage(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    del methods[LifecycleStage.FEEDBACK]

    with pytest.raises(ValueError, match="missing=feedback"):
        platform.readiness.plan(methods)


def test_preflight_does_not_infer_planned_upstream_outputs(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    capture = tmp_path / "cabinet2"
    capture.mkdir()
    (capture / "frame.png").write_bytes(b"captured-frame")

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={LifecycleEvidenceKey.CAPTURE: capture},
    )

    assert report.status is ReadinessStatus.BLOCKED
    assert report.stages[0].status is ReadinessStatus.READY
    assert report.stages[1].status is ReadinessStatus.BLOCKED
    assert report.stages[1].blockers[0].code is ReadinessBlockerCode.EVIDENCE_MISSING
    assert "observations" in report.stages[1].blockers[0].message


def test_preflight_requires_executable_ready_exact_methods(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    for stage, method in methods.items():
        if stage is LifecycleStage.CAMERA:
            platform.plugins.register_method(method.descriptor)
        elif stage is LifecycleStage.DEPTH:
            platform.plugins.register_adapter(_ReadyAdapter(method, ready=False))

    report = platform.readiness.preflight(platform.readiness.plan(methods), roots={})

    ingestion = report.stages[0]
    camera = report.stages[1]
    depth = report.stages[tuple(LifecycleStage).index(LifecycleStage.DEPTH)]
    assert ingestion.blockers[0].code is ReadinessBlockerCode.METHOD_NOT_INSTALLED
    assert camera.blockers[0].code is ReadinessBlockerCode.METHOD_NOT_EXECUTABLE
    assert depth.blockers[0].code is ReadinessBlockerCode.METHOD_NOT_READY


def test_all_stages_ready_only_with_verified_concrete_evidence(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    capture = tmp_path / "cabinet2"
    capture.mkdir()
    (capture / "frame.png").write_bytes(b"captured-frame")
    roots = _complete_roots(platform, capture)
    plan = platform.readiness.plan(methods)

    first = platform.readiness.preflight(plan, roots=roots)
    second = platform.readiness.preflight(plan, roots=roots)

    assert first == second
    assert first.status is ReadinessStatus.READY
    assert all(item.status is ReadinessStatus.READY for item in first.stages)
    assert all(not item.blockers for item in first.stages)
    assert LifecycleReadinessReport.model_validate_json(first.model_dump_json()) == first


def test_unresolved_metric_scale_blocks_depth_and_articulation(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    unresolved = platform.artifacts.put_json(
        {
            "schema": "physrealx.camera-rig-calibration/1",
            "scale_status": "unresolved",
        }
    )

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={
            LifecycleEvidenceKey.OBSERVATIONS: _observation_resource(platform),
            LifecycleEvidenceKey.CAMERA_SOLUTION: unresolved,
            LifecycleEvidenceKey.ENHANCED_DEPTH: platform.artifacts.put_json(
                _VALID_SEMANTIC_EVIDENCE[LifecycleEvidenceKey.ENHANCED_DEPTH]
            ),
        },
    )

    for stage in (LifecycleStage.DEPTH, LifecycleStage.ARTICULATION):
        result = report.stages[tuple(LifecycleStage).index(stage)]
        blocker = next(
            item
            for item in result.blockers
            if item.code is ReadinessBlockerCode.EVIDENCE_GATE_FAILED
        )
        assert "scale_status" in blocker.message
        assert LifecycleEvidenceKey.CAMERA_SOLUTION not in result.satisfied_evidence


def test_metric_alignment_stage_requires_evidence_bound_to_camera_artifact(
    tmp_path: Path,
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    camera = platform.artifacts.put_json(
        _VALID_SEMANTIC_EVIDENCE[LifecycleEvidenceKey.CAMERA_RECONSTRUCTION]
    )
    evidence = platform.artifacts.put_json(_metric_alignment_evidence(f"art_{'f' * 32}"))

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={
            LifecycleEvidenceKey.CAMERA_RECONSTRUCTION: camera,
            LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE: evidence,
        },
    )

    stage = report.stages[tuple(LifecycleStage).index(LifecycleStage.METRIC_ALIGNMENT)]
    assert stage.status is ReadinessStatus.BLOCKED
    assert any("does not match camera_reconstruction" in item.message for item in stage.blockers)
    assert LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE not in stage.satisfied_evidence


def test_failed_depth_report_cannot_satisfy_articulation_gate(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    failed_depth = platform.artifacts.put_json(
        {
            "schema": "physrealx.depth-enhancement-report/1",
            "status": "FAILED",
        }
    )

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={
            LifecycleEvidenceKey.CAMERA_SOLUTION: platform.artifacts.put_json(
                _VALID_SEMANTIC_EVIDENCE[LifecycleEvidenceKey.CAMERA_SOLUTION]
            ),
            LifecycleEvidenceKey.ENHANCED_DEPTH: failed_depth,
        },
    )

    articulation = report.stages[tuple(LifecycleStage).index(LifecycleStage.ARTICULATION)]
    assert any(
        item.code is ReadinessBlockerCode.EVIDENCE_GATE_FAILED and "status 'FAILED'" in item.message
        for item in articulation.blockers
    )
    assert LifecycleEvidenceKey.ENHANCED_DEPTH not in articulation.satisfied_evidence


def test_geometry_collision_and_assembly_gates_reject_unbound_or_incomplete_evidence(
    tmp_path: Path,
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    missing_mesh = ArtifactRef(
        artifact_id=f"art_{'d' * 32}",
        size_bytes=128,
        media_type="model/obj",
    )
    geometry = platform.artifacts.put_json(
        {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "status": "PASSED",
            "unit": "m",
            "source_camera_artifact_id": f"art_{'a' * 32}",
            "source_depth_artifact_id": f"art_{'b' * 32}",
            "method": {
                "name": "fixture_tsdf",
                "version": "1.0.0",
                "source_revision": "fixture-1.0.0",
            },
            "metrics": {"surface_vertices": 3.0},
            "gates": {"metric_scale": True, "nonempty_meshes": True},
            "meshes": [{"link": "base", "artifact": missing_mesh.model_dump(mode="json")}],
        }
    )
    incomplete_assembly = platform.artifacts.put_json(
        {
            "schema": "physrealx.asset-assembly-bundle/1",
            "resource_id": "cabinet2.asset",
            "version": "1.0.0",
            "metric_scale": {"status": "unresolved"},
        }
    )

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={
            LifecycleEvidenceKey.VISUAL_GEOMETRY: geometry,
            LifecycleEvidenceKey.ASSET_ASSEMBLY: incomplete_assembly,
        },
    )

    articulation = report.stages[tuple(LifecycleStage).index(LifecycleStage.ARTICULATION)]
    usd = report.stages[tuple(LifecycleStage).index(LifecycleStage.USD)]
    assert any("unavailable artifacts" in item.message for item in articulation.blockers)
    assert any("asset assembly" in item.message for item in usd.blockers)


@pytest.mark.parametrize(
    ("payload", "media_type", "message"),
    [
        (b"opaque-but-hash-valid", "application/octet-stream", "binary media type"),
        (
            b'{"schema":"vendor.camera/1","scale_status":"resolved"}',
            "application/json",
            "recognizable versioned",
        ),
    ],
)
def test_unknown_or_binary_artifact_cannot_satisfy_scientific_gate(
    tmp_path: Path,
    payload: bytes,
    media_type: str,
    message: str,
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    evidence = platform.artifacts.put_bytes(payload, media_type=media_type)

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={LifecycleEvidenceKey.CAMERA_SOLUTION: evidence},
    )

    depth = report.stages[tuple(LifecycleStage).index(LifecycleStage.DEPTH)]
    assert any(
        item.code is ReadinessBlockerCode.EVIDENCE_GATE_FAILED and message in item.message
        for item in depth.blockers
    )


def test_observations_require_registered_manifest_and_verified_closure(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    resource = _observation_resource(platform)
    manifest_artifact = platform.artifacts.put_bytes(
        platform.artifacts.get_bytes(resource.artifact_id or ""),
        media_type="application/json",
    )

    unregistered = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={LifecycleEvidenceKey.OBSERVATIONS: manifest_artifact},
    )
    camera = unregistered.stages[tuple(LifecycleStage).index(LifecycleStage.CAMERA)]
    assert camera.blockers[0].code is ReadinessBlockerCode.EVIDENCE_GATE_FAILED
    assert "registered immutable observation_set" in camera.blockers[0].message

    missing_artifact_id = f"art_{'f' * 32}"
    manifest = ObservationSetManifest.model_validate_json(
        platform.artifacts.get_bytes(resource.artifact_id or "")
    )
    invalid_stream = manifest.streams[0].model_copy(
        update={
            "artifacts": (
                manifest.streams[0]
                .artifacts[0]
                .model_copy(update={"artifact_id": missing_artifact_id}),
            )
        }
    )
    invalid_manifest = manifest.model_copy(
        update={"version": "1.0.1", "streams": (invalid_stream,)}
    )
    invalid_artifact = platform.artifacts.put_json(invalid_manifest.model_dump(mode="json"))
    invalid_resource = platform.repository.register_resource(
        invalid_manifest.reference(invalid_artifact.artifact_id)
    )

    missing_closure = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={LifecycleEvidenceKey.OBSERVATIONS: invalid_resource},
    )
    camera = missing_closure.stages[tuple(LifecycleStage).index(LifecycleStage.CAMERA)]
    assert camera.blockers[0].code is ReadinessBlockerCode.EVIDENCE_GATE_FAILED
    assert "unavailable referenced artifacts" in camera.blockers[0].message


def test_empty_path_and_unpublished_artifact_are_blocked(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = platform.artifacts.put_bytes(b"temporary", media_type="application/octet-stream")
    # A different object ID is well-formed but absent from the local object store.
    missing = missing.model_copy(update={"artifact_id": f"art_{'f' * 32}"})

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={
            LifecycleEvidenceKey.CAPTURE: empty,
            LifecycleEvidenceKey.OBSERVATIONS: missing,
        },
    )

    assert report.stages[0].blockers[0].code is ReadinessBlockerCode.EVIDENCE_UNAVAILABLE
    assert report.stages[1].blockers[0].code is ReadinessBlockerCode.EVIDENCE_UNAVAILABLE


def test_robot_target_rejects_an_ordinary_path(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    methods = _methods()
    _register_adapters(platform, methods)
    robot_directory = tmp_path / "robot"
    robot_directory.mkdir()
    (robot_directory / "robot.urdf").write_text("<robot name='unbound'/>", encoding="utf-8")

    report = platform.readiness.preflight(
        platform.readiness.plan(methods),
        roots={LifecycleEvidenceKey.ROBOT_TARGET: robot_directory},
    )

    deployment = report.stages[tuple(LifecycleStage).index(LifecycleStage.ROS_DEPLOY)]
    assert any(
        item.code is ReadinessBlockerCode.EVIDENCE_GATE_FAILED
        and "integrity-verified artifact or resource" in item.message
        for item in deployment.blockers
    )


def test_readiness_evidence_model_is_strict() -> None:
    with pytest.raises(ValidationError):
        LifecycleEvidence(key="capture", path="/tmp/capture")  # type: ignore[arg-type]
