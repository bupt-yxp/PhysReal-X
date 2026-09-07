"""Robust, independently validated Sim(3) metrology for COLMAP cameras."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import dependency_probe, input_artifact, publish, read_json
from physrealx.methods import builtin_method


class MetricAlignmentConfig(DomainModel):
    inlier_threshold_m: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    max_holdout_rmse_m: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    min_inlier_fraction: float = Field(default=0.75, gt=0.5, le=1)
    trials: int = Field(default=256, ge=1, le=10000)
    seed: int = Field(default=0, ge=0)


def fit_similarity(source: Any, target: Any) -> tuple[float, Any, Any]:
    """Least-squares proper similarity; reject collinear or collapsed evidence."""
    import numpy as np

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise DomainValidationError("correspondences must be matching N by 3 arrays")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise DomainValidationError("at least three finite correspondences are required")
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    for cloud in (x, y):
        singular = np.linalg.svd(cloud, compute_uv=False)
        if singular[0] <= 1e-12 or singular[1] / singular[0] <= 1e-6:
            raise DomainValidationError("correspondences are geometrically degenerate")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    scale = float(singular @ signs / np.mean(np.sum(x * x, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise DomainValidationError("similarity scale must be positive and finite")
    return scale, rotation, target.mean(axis=0) - scale * rotation @ source.mean(axis=0)


class MetricAlignmentAdapter:
    descriptor = builtin_method(
        MethodCapability.CAMERA_ALIGN_METRIC,
        "ransac_umeyama",
        input_contract="CameraRigCalibration/1+MetricAlignmentEvidence/1",
        output_contract="CameraRigCalibration/1+MetricAlignmentReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe({"numpy": ("numpy", "1.26.4")})

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np

        from physrealx.evidence import ScientificEvidenceValidator

        config = MetricAlignmentConfig.model_validate(spec.method.config)
        camera_ref = input_artifact(spec, "camera_rig", artifacts)
        evidence_ref = input_artifact(spec, "evidence", artifacts)
        camera = read_json(artifacts, camera_ref, "physrealx.camera-rig-calibration/1")
        evidence = read_json(artifacts, evidence_ref, "physrealx.metric-alignment-evidence/1")
        error = ScientificEvidenceValidator.metric_alignment_evidence_error(evidence)
        if error:
            raise DomainValidationError(error)
        if evidence["camera_rig_artifact_id"] != camera_ref.artifact_id:
            raise DomainValidationError("metrology is bound to a different camera rig")
        if camera.get("scale_status") != "unresolved":
            raise DomainValidationError("metric alignment requires unresolved camera scale")
        source = np.array([item["source_xyz"] for item in evidence["train_correspondences"]])
        target = np.array([item["target_xyz_m"] for item in evidence["train_correspondences"]])
        generator = np.random.default_rng(config.seed)
        best: Any = None
        best_score = (-1, -float("inf"))
        for _ in range(config.trials):
            sample = generator.choice(len(source), 3, replace=False)
            try:
                scale, rotation, translation = fit_similarity(source[sample], target[sample])
            except DomainValidationError:
                continue
            residual = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
            mask = residual <= config.inlier_threshold_m
            score = (int(mask.sum()), -float(np.sum(residual[mask] ** 2)))
            if score > best_score:
                best, best_score = mask, score
        if best is None or best.sum() < max(4, config.min_inlier_fraction * len(source)):
            raise MethodExecutionError("metric alignment has insufficient nondegenerate inliers")
        scale, rotation, translation = fit_similarity(source[best], target[best])
        source_test = np.array([v["source_xyz"] for v in evidence["holdout_correspondences"]])
        target_test = np.array([v["target_xyz_m"] for v in evidence["holdout_correspondences"]])
        residual = np.linalg.norm(
            scale * source_test @ rotation.T + translation - target_test, axis=1
        )
        rmse = float(np.sqrt(np.mean(residual**2)))
        if rmse > config.max_holdout_rmse_m:
            raise MethodExecutionError("metric alignment failed independent holdout RMSE")
        # A reconstruction may contain disconnected components in unrelated coordinate frames.
        # Publish only the explicitly measured component; never scale other components by guess.
        selected = []
        for stream in camera.get("streams", []):
            for component in stream["components"]:
                key = f"{stream['stream_name']}:{component['component_id']}"
                if key != evidence["component_id"]:
                    continue
                for image in component["images"]:
                    pose = np.asarray(image["cam_from_world"], dtype=float)
                    if pose.shape != (3, 4) or not np.isfinite(pose).all():
                        raise DomainValidationError("camera pose must be a finite 3 by 4 matrix")
                    old_rotation = pose[:, :3]
                    if not np.allclose(old_rotation.T @ old_rotation, np.eye(3), atol=1e-6):
                        raise DomainValidationError("camera rotation must be orthonormal")
                    new_rotation = old_rotation @ rotation.T
                    new_translation = scale * pose[:, 3] - new_rotation @ translation
                    image["cam_from_world"] = np.column_stack(
                        (new_rotation, new_translation)
                    ).tolist()
                for track in component.get("fiducial_tracks", []):
                    if "xyz" in track:
                        track["xyz"] = (
                            scale * rotation @ np.array(track["xyz"]) + translation
                        ).tolist()
                selected.append({**stream, "components": [component]})
        if len(selected) != 1:
            raise DomainValidationError("metrology must select one existing stream:component_id")
        camera.update(
            streams=selected,
            scale_status="resolved",
            source_camera_artifact_id=camera_ref.artifact_id,
            metric_alignment_evidence_artifact_id=evidence_ref.artifact_id,
            coordinate_convention={
                "frame": evidence["target_frame"],
                "pose": "world_to_camera",
                "translation_unit": "m",
                "quaternion_order": "wxyz",
            },
        )
        # Cross-component relative transforms refer to the original coordinate systems.
        camera.pop("fiducial_alignments", None)
        camera.update(unit="m", coordinate_frame=evidence["target_frame"])
        metrics = {"scale": scale, "holdout_rmse_m": rmse, "inlier_fraction": float(best.mean())}
        report = {
            "schema": "physrealx.metric-alignment-report/1",
            "status": "PASSED",
            "source_camera_artifact_id": camera_ref.artifact_id,
            "evidence_artifact_id": evidence_ref.artifact_id,
            "component_id": evidence["component_id"],
            "scale_status": "resolved",
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "method": self.descriptor.model_dump(mode="json"),
            "metrics": metrics,
            "gates": {"nondegenerate": True, "independent_holdout": True},
        }
        return publish(
            spec,
            artifacts,
            {"camera_rig_calibration": camera, "metric_alignment_report": report},
            metrics=metrics,
        )
