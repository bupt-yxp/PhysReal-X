"""Independent trajectory replay against a packaged, fully assembled physical asset."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    MethodCapability,
    OperationSpec,
    SceneEntityRole,
    SceneEntitySpec,
    SceneSpec,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.evidence import ScientificEvidenceValidator
from physrealx.integrations._artifacts import (
    dependency_probe,
    input_artifact,
    publish,
    read_array,
    read_json,
)
from physrealx.integrations.mujoco import MujocoConfig, compile_scene
from physrealx.methods import builtin_method


class PhysicalValidationEvidence(DomainModel):
    schema_id: Literal["physrealx.physical-validation-evidence/1"] = Field(
        default="physrealx.physical-validation-evidence/1", alias="schema"
    )
    package_manifest: ArtifactRef
    # Columns: time, all q, all qdot, all generalized actuator efforts.
    trajectory: ArtifactRef
    capture_id: str = Field(min_length=1)
    position_unit: Literal["m", "rad"]
    force_unit: Literal["N", "N*m"]
    simulation: MujocoConfig


class PhysicalValidationConfig(DomainModel):
    max_position_rmse: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    max_velocity_rmse: float = Field(default=0.05, gt=0, allow_inf_nan=False)


class PhysicalValidationAdapter:
    descriptor = builtin_method(
        MethodCapability.ASSET_VALIDATE,
        "mujoco_holdout",
        input_contract="AssetPackageManifest/1+PhysicalValidationEvidence/1",
        output_contract="PhysicalAssetValidationReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "mujoco": ("mujoco", "3.3.7"),
                "numpy": ("numpy", "1.26.4"),
                "trimesh": ("trimesh", "4.5.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import mujoco
        import numpy as np

        config = PhysicalValidationConfig.model_validate(spec.method.config)
        package_ref = input_artifact(spec, "package", artifacts)
        evidence_ref = input_artifact(spec, "evidence", artifacts)
        evidence = PhysicalValidationEvidence.model_validate(
            read_json(artifacts, evidence_ref, "physrealx.physical-validation-evidence/1")
        )
        if evidence.package_manifest != package_ref:
            raise DomainValidationError("validation evidence belongs to another package")
        package = read_json(artifacts, package_ref, "physrealx.asset-package-manifest/1")
        if package.get("roundtrip_status") != "PASSED" or package.get("self_contained") is not True:
            raise DomainValidationError(
                "physical validation requires a self-contained validated package"
            )
        usd = ArtifactRef.model_validate(package["usd_asset"])
        if not artifacts.verify(usd):
            raise DomainValidationError("USD package artifact is unavailable")
        assembly_ref = ArtifactRef.model_validate(package["source_assembly"])
        asset = ScientificEvidenceValidator(artifacts).assembled_asset(
            read_json(artifacts, assembly_ref, "physrealx.asset-assembly-bundle/1")
        )
        if package.get("asset_spec") != asset.model_dump(mode="json"):
            raise DomainValidationError("package asset differs from assembled physics")
        from pxr import Usd

        from physrealx.integrations.usd import validate_usd_stage

        with tempfile.TemporaryDirectory(prefix="physrealx-physical-usd-") as directory:
            path = Path(directory) / "asset.usdz"
            artifacts.materialize(usd, path)
            validate_usd_stage(Usd.Stage.Open(str(path)), asset, artifacts)
        # Freeze a third capture: neither fitting nor model-selection trajectories may be reused.
        for ref in asset.evidence:
            if ref.media_type != "application/json":
                continue
            from physrealx.evidence import strict_json_object

            document = strict_json_object(artifacts.get_bytes(ref), "asset evidence")
            if document.get("schema") == "physrealx.identified-joint-physics/1":
                measured = strict_json_object(
                    artifacts.get_bytes(document["evidence_artifact_id"]),
                    "joint identification evidence",
                )
                if evidence.capture_id in {
                    measured.get("train_capture_id"),
                    measured.get("holdout_capture_id"),
                }:
                    raise DomainValidationError(
                        "physical validation capture overlaps identification"
                    )
                if evidence.trajectory.artifact_id in {
                    measured.get("train", {}).get("artifact_id"),
                    measured.get("holdout", {}).get("artifact_id"),
                }:
                    raise DomainValidationError(
                        "physical validation trajectory overlaps identification"
                    )
        scene = SceneSpec(
            resource_id="physical.validation",
            version="1.0.0",
            display_name="Physical validation",
            entities=(
                SceneEntitySpec(name="asset", asset=asset.reference(), role=SceneEntityRole.OBJECT),
            ),
        )
        compiled = compile_scene(scene, {"asset": asset}, artifacts, evidence.simulation)
        selected = [compiled.joints[name] for name in evidence.simulation.controlled_joints]
        units = {"m" if joint.joint_type.value == "prismatic" else "rad" for joint in selected}
        if units != {evidence.position_unit} or evidence.force_unit != (
            "N" if evidence.position_unit == "m" else "N*m"
        ):
            raise DomainValidationError(
                "holdout trajectory units disagree with the controlled joints"
            )
        model = mujoco.MjModel.from_xml_string(compiled.xml, assets=compiled.files)
        data = mujoco.MjData(model)
        qpos = [int(model.joint(name).qposadr[0]) for name in evidence.simulation.controlled_joints]
        dofs = [int(model.joint(name).dofadr[0]) for name in evidence.simulation.controlled_joints]
        count = len(qpos)
        trajectory = read_array(artifacts, evidence.trajectory).astype(float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 1 + 3 * count or len(trajectory) < 16:
            raise DomainValidationError(
                "physical holdout requires at least 16 aligned state/action samples"
            )
        interval = evidence.simulation.timestep_seconds * evidence.simulation.substeps
        if not np.allclose(np.diff(trajectory[:, 0]), interval, atol=1e-9, rtol=1e-6):
            raise DomainValidationError(
                "holdout timestamps must match the declared control interval"
            )
        position, velocity, force = (
            trajectory[:, 1 : 1 + count],
            trajectory[:, 1 + count : 1 + 2 * count],
            trajectory[:, 1 + 2 * count :],
        )
        if (np.abs(force) > evidence.simulation.effort_limits).any():
            raise DomainValidationError("holdout requires efforts beyond declared physical limits")
        data.qpos[qpos], data.qvel[dofs] = position[0], velocity[0]
        mujoco.mj_forward(model, data)
        errors = []
        for index in range(len(trajectory) - 1):
            data.ctrl[:] = force[index]
            mujoco.mj_step(model, data, nstep=evidence.simulation.substeps)
            if (
                any(w.number for w in data.warning)
                or not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                raise MethodExecutionError("physical holdout dynamics are unstable")
            errors.append(
                (data.qpos[qpos] - position[index + 1], data.qvel[dofs] - velocity[index + 1])
            )
        errors_array = np.array(errors)
        position_rmse = float(np.sqrt(np.mean(errors_array[:, 0] ** 2)))
        velocity_rmse = float(np.sqrt(np.mean(errors_array[:, 1] ** 2)))
        if position_rmse > config.max_position_rmse or velocity_rmse > config.max_velocity_rmse:
            raise MethodExecutionError(
                "physical asset failed independent position/velocity holdout"
            )
        metrics = {
            "position_rmse": position_rmse,
            "velocity_rmse": velocity_rmse,
            "samples": float(len(trajectory)),
        }
        report = {
            "schema": "physrealx.asset-validation-report/1",
            "status": "PASSED",
            "validation_scope": "independent_physical_trajectory",
            "physical_validation": True,
            "package_artifact_id": package_ref.artifact_id,
            "evidence_artifact_id": evidence_ref.artifact_id,
            "method": self.descriptor.model_dump(mode="json"),
            "metrics": metrics,
            "limitations": [
                "fixed asset root",
                "MuJoCo dynamic sliding friction and solver contact response",
            ],
        }
        return publish(spec, artifacts, {"physical_validation_report": report}, metrics=metrics)
