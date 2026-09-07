"""Recover a one-DOF rigid joint from independently paired metric point tracks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    JointSpec,
    JointType,
    MethodCapability,
    OperationSpec,
    PoseSpec,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import (
    dependency_probe,
    input_artifact,
    publish,
    read_array,
    read_json,
)
from physrealx.integrations.metric import fit_similarity
from physrealx.methods import builtin_method


class InteractionPair(DomainModel):
    schema_id: Literal["physrealx.paired-interaction-point-cloud-bundle/1"] = Field(
        default="physrealx.paired-interaction-point-cloud-bundle/1", alias="schema"
    )
    camera_rig: ArtifactRef
    joint_name: str
    parent: str
    child: str
    # Rows correspond to tracked material points; arrays are already in parent-link metres.
    source_points: ArtifactRef
    target_points: ArtifactRef
    holdout_source_points: ArtifactRef
    holdout_target_points: ArtifactRef
    lower_limit: float = Field(allow_inf_nan=False)
    upper_limit: float = Field(allow_inf_nan=False)
    limit_unit: Literal["m", "rad"]
    limit_evidence: ArtifactRef

    @model_validator(mode="after")
    def validate_evidence(self) -> InteractionPair:
        ids = [
            item.artifact_id
            for item in (
                self.source_points,
                self.target_points,
                self.holdout_source_points,
                self.holdout_target_points,
            )
        ]
        if len(set(ids)) != 4 or self.parent == self.child:
            raise ValueError(
                "joint recovery requires separate fit/holdout point artifacts and links"
            )
        if not self.lower_limit <= 0 < self.upper_limit:
            raise ValueError("reference pose must be within explicitly measured joint limits")
        return self


class ArticulationConfig(DomainModel):
    max_rmse_m: float = Field(default=0.003, gt=0, allow_inf_nan=False)
    min_motion_m: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    min_rotation_rad: float = Field(default=0.02, gt=0, le=0.5)


class ArticulationAdapter:
    descriptor = builtin_method(
        MethodCapability.ARTICULATION_RECOVER,
        "rigid_pair_fit",
        input_contract="PairedInteractionPointCloudBundle/1",
        output_contract="ArticulationModel/1+ArticulationRecoveryReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe({"numpy": ("numpy", "1.26.4"), "scipy": ("scipy", "1.13.1")})

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        from scipy.spatial.transform import Rotation

        config = ArticulationConfig.model_validate(spec.method.config)
        reference = input_artifact(spec, "pair", artifacts)
        pair = InteractionPair.model_validate(
            read_json(artifacts, reference, "physrealx.paired-interaction-point-cloud-bundle/1")
        )
        camera = read_json(artifacts, pair.camera_rig, "physrealx.camera-rig-calibration/1")
        if camera.get("scale_status") != "resolved" or not artifacts.verify(pair.limit_evidence):
            raise DomainValidationError(
                "joint recovery requires metric cameras and measured limit evidence"
            )
        source, target, test_source, test_target = [
            read_array(artifacts, item).astype(float)
            for item in (
                pair.source_points,
                pair.target_points,
                pair.holdout_source_points,
                pair.holdout_target_points,
            )
        ]
        fit_similarity(source, target)  # Validate rank and shape before rigid fitting.
        fit_similarity(test_source, test_target)
        if set(map(tuple, source)) & set(map(tuple, test_source)):
            raise DomainValidationError("fit and holdout material points overlap")
        u, _, vt = np.linalg.svd((target - target.mean(0)).T @ (source - source.mean(0)))
        signs = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
        rotation = u @ signs @ vt
        translation = target.mean(0) - rotation @ source.mean(0)
        rotation_vector = Rotation.from_matrix(rotation).as_rotvec()
        angle = float(np.linalg.norm(rotation_vector))
        if angle < config.min_rotation_rad:
            displacement = target.mean(0) - source.mean(0)
            motion = float(np.linalg.norm(displacement))
            if motion < config.min_motion_m:
                raise MethodExecutionError("joint motion is not observable")
            axis, origin = displacement / motion, np.zeros(3)
            rotation, translation = np.eye(3), displacement
            joint_type, unit = JointType.PRISMATIC, "m"
        else:
            axis = rotation_vector / angle
            if abs(float(axis @ translation)) > config.max_rmse_m:
                raise MethodExecutionError(
                    "helical motion cannot be represented as a revolute joint"
                )
            origin = np.linalg.lstsq(
                np.vstack((np.eye(3) - rotation, axis)), np.append(translation, 0.0), rcond=None
            )[0]
            translation = (np.eye(3) - rotation) @ origin
            motion, joint_type, unit = angle, JointType.REVOLUTE, "rad"
        if pair.limit_unit != unit or not pair.lower_limit <= motion <= pair.upper_limit:
            raise DomainValidationError(
                "identified motion disagrees with measured joint limits or units"
            )
        errors = [
            float(np.sqrt(np.mean(np.sum((a @ rotation.T + translation - b) ** 2, axis=1))))
            for a, b in ((source, target), (test_source, test_target))
        ]
        if max(errors) > config.max_rmse_m:
            raise MethodExecutionError("rigid joint model failed fit or independent holdout error")
        joint = JointSpec(
            name=pair.joint_name,
            parent=pair.parent,
            child=pair.child,
            joint_type=joint_type,
            axis=tuple(axis.tolist()),
            parent_to_joint=PoseSpec(translation_m=tuple(origin.tolist())),
            lower_limit=pair.lower_limit,
            upper_limit=pair.upper_limit,
        )
        model = {
            "schema": "physrealx.articulation-model/1",
            "status": "IDENTIFIED",
            "unit": "m",
            "root_link": pair.parent,
            "links": [pair.parent, pair.child],
            "joints": [joint.model_dump(mode="json")],
            "evidence_artifact_id": reference.artifact_id,
            "reference_geometry_frame": "parent_link",
        }
        metrics = {"fit_rmse_m": errors[0], "holdout_rmse_m": errors[1], "observed_motion": motion}
        report = {
            "schema": "physrealx.articulation-recovery-report/1",
            "status": "PASSED",
            "method": self.descriptor.model_dump(mode="json"),
            "metrics": metrics,
            "source_artifact_id": reference.artifact_id,
        }
        return publish(
            spec,
            artifacts,
            {"articulation_model": model, "articulation_recovery_report": report},
            metrics=metrics,
        )
