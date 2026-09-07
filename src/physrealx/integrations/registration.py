"""Register a two-state observed slider to its supplied articulated USD reference.

The support plane and recovered motion initialize orientation. Robust surface
registration fits pose and the two joint positions on training views only.
This is model-conditioned registration, never independent physical metrology.
"""

from __future__ import annotations

import io
import zipfile
from importlib import import_module
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import ArtifactRef, DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import input_artifact, publish, read_json
from physrealx.integrations.capture_rgbd import capture_support_region
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.rgbd import MetricRGBDInput, load_rgbd
from physrealx.methods import builtin_method


class ReferenceRegistrationInput(DomainModel):
    schema_id: Literal["physrealx.reference-registration-input/1"] = Field(
        default="physrealx.reference-registration-input/1", alias="schema"
    )
    rgbd: MetricRGBDInput
    reconstruction_report: ArtifactRef


class RegistrationConfig(DomainModel):
    joint: str = Field(min_length=1)
    seed: int = Field(default=2026, ge=0)
    points_per_view: int = Field(default=1200, ge=300, le=5000)
    mesh_samples: int = Field(default=20000, ge=5000, le=100000)
    fit_points_per_state: int = Field(default=2500, ge=1000, le=10000)
    component_voxel_m: float = Field(default=0.012, gt=0, le=0.03, allow_inf_nan=False)
    max_holdout_median_m: float = Field(default=0.015, gt=0, le=0.03, allow_inf_nan=False)
    max_holdout_p90_m: float = Field(default=0.03, gt=0, le=0.05, allow_inf_nan=False)
    min_foreground_fraction: float = Field(default=0.75, gt=0.5, le=1)


def connected_foreground(points: Any, plane_z: float, voxel: float) -> Any:
    """Remove detached background components without using reference-mesh distances."""
    import numpy as np
    from scipy.ndimage import label

    points = points[points[:, 2] - plane_z > voxel]
    if len(points) < 100:
        raise MethodExecutionError("registration has insufficient above-plane object points")
    grid = np.floor((points - points.min(0)) / voxel)
    shape = grid.max(0) + 1
    if not np.isfinite(shape).all() or np.prod(shape) > 10_000_000:
        raise DomainValidationError("registration foreground exceeds its bounded spatial grid")
    cells = grid.astype(int)
    occupancy = np.zeros(tuple(shape.astype(int)), bool)
    occupancy[tuple(cells.T)] = True
    components, _ = label(occupancy, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    # Articulated surfaces can be disconnected when the guide/hinge is occluded.
    # Retain substantial components; tiny detached markers remain excluded.
    retained = sizes >= max(1, sizes.max() * 0.1)
    retained[0] = False
    return points[retained[components[tuple(cells.T)]]]


def fit_articulated_surfaces(
    points: list[Any],
    static: Any,
    moving: Any,
    initial_translation: Any,
    limits: tuple[float, float],
    *,
    seed: int,
    max_points: int,
) -> tuple[Any, Any, Any, float]:
    """Fit one rigid transform and observed positions; evaluate no held-out data here."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    generator = np.random.default_rng(seed)
    samples = [
        row[generator.choice(len(row), min(len(row), max_points), replace=False)] for row in points
    ]
    static_tree, moving_tree = cKDTree(static), cKDTree(moving)

    def residual(parameters: Any) -> Any:
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        errors = []
        for index, sample in enumerate(samples):
            transformed = sample @ rotation.T + parameters[3:6]
            displacement = np.array([parameters[6 + index], 0, 0])
            errors.append(
                np.minimum(
                    static_tree.query(transformed)[0],
                    moving_tree.query(transformed - displacement)[0],
                )
            )
        # The observed support plane is an orientation/location constraint, not a
        # camera-independent surface truth. Small model/depth mismatch is allowed.
        errors.append(
            2 * np.array([parameters[0], parameters[1], parameters[5] - initial_translation[2]])
        )
        return np.concatenate(errors)

    best = None
    for yaw in (-0.4, 0.0, 0.4):
        initial = np.r_[
            0.0,
            0.0,
            yaw,
            initial_translation,
            limits[0] + 0.15 * (limits[1] - limits[0]),
            limits[0] + 0.85 * (limits[1] - limits[0]),
        ]
        lower = np.r_[
            -0.08, -0.08, -0.8, initial_translation - [0.08, 0.08, 0.02], limits[0], limits[0]
        ]
        upper = np.r_[
            0.08,
            0.08,
            0.8,
            initial_translation + np.array([0.08, 0.08, 0.02]),
            limits[1],
            limits[1],
        ]
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.006,
            max_nfev=100,
            diff_step=1e-4,
        )
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
    if best is None or not np.isfinite(best.x).all():
        raise MethodExecutionError("reference registration did not find a finite pose")
    if best.x[7] - best.x[6] < 0.01:
        raise MethodExecutionError("reference registration lacks a consistent observed slider span")
    return (Rotation.from_rotvec(best.x[:3]).as_matrix(), best.x[3:6], best.x[6:], float(best.cost))


class ReferenceRegistrationAdapter:
    descriptor = builtin_method(
        MethodCapability.CAMERA_ALIGN_METRIC,
        "articulated_surface_icp",
        input_contract="CameraRigCalibration/1+MetricAlignmentEvidence/1",
        output_contract="CameraRigCalibration/1+MetricAlignmentReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        from physrealx.integrations.rgbd_fusion import RGBDFusionAdapter

        return RGBDFusionAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import json

        import numpy as np
        from scipy.spatial import cKDTree

        trimesh = import_module("trimesh")
        config = RegistrationConfig.model_validate(spec.method.config)
        evidence = input_artifact(spec, "evidence", artifacts)
        request = ReferenceRegistrationInput.model_validate(
            read_json(artifacts, evidence, "physrealx.reference-registration-input/1")
        )
        camera_ref = input_artifact(spec, "camera_rig", artifacts)
        if request.rgbd.camera_rig != camera_ref:
            raise DomainValidationError("reference registration is bound to another camera")
        camera, _, views = load_rgbd(request.rgbd, artifacts)
        reconstruction = read_json(
            artifacts, request.reconstruction_report, "physrealx.geometry-reconstruction-report/1"
        )
        if (
            reconstruction.get("source_camera_artifact_id") != camera_ref.artifact_id
            or reconstruction.get("source_depth_artifact_id")
            != request.rgbd.enhanced_depth.artifact_id
            or reconstruction.get("evaluation_scope") != "held_out_views"
        ):
            raise DomainValidationError(
                "registration requires the matching held-out reconstruction"
            )
        train_ids = set(reconstruction["training_frame_ids"])
        test_ids = set(reconstruction["validation_frame_ids"])
        if train_ids & test_ids or train_ids | test_ids != {view.frame_id for view in views}:
            raise DomainValidationError("registration view split leaks or omits capture frames")
        visual = ArtifactRef.model_validate(reconstruction["bundle"])
        if visual.size_bytes > 1024**3:
            raise DomainValidationError("registration visual bundle exceeds its size bound")
        with zipfile.ZipFile(io.BytesIO(artifacts.get_bytes(visual))) as archive:
            if archive.getinfo("model/joint_info.json").file_size > 1024**2:
                raise DomainValidationError("registration joint manifest exceeds its bound")
            recovered = json.loads(archive.read("model/joint_info.json"))
        observed = [row for row in recovered if row["joint"] == "slider"]
        if len(recovered) != 2 or len(observed) != 1:
            raise DomainValidationError("registration supports one observed sliding motion group")
        reference = read_json(artifacts, request.rgbd.reference_model, "physrealx.usd-reference/1")
        joint = next((j for j in reference["joints"] if j["name"] == config.joint), None)
        if (
            joint is None
            or joint["joint_type"] != "slide"
            or not np.allclose(joint["axis"], [1, 0, 0])
        ):
            raise DomainValidationError("surface registration requires an authored +X slider")
        support = capture_support_region(camera)
        normal = support["normal"]
        direction = np.asarray(observed[0]["jointData"]["axis"]["direction"], dtype=float)
        if (
            direction.shape != (3,)
            or not np.isfinite(direction).all()
            or not np.isclose(np.linalg.norm(direction), 1, atol=1e-5)
            or abs(direction @ normal) > 0.35
        ):
            raise DomainValidationError("recovered slider is inconsistent with the support plane")
        horizontal = direction - normal * (direction @ normal)
        horizontal /= np.linalg.norm(horizontal)
        basis = np.array([horizontal, np.cross(normal, horizontal), normal])
        generator = np.random.default_rng(config.seed)
        streams_by_id = {view.frame_id: view.stream for view in views}
        streams = tuple(
            dict.fromkeys(streams_by_id[frame] for frame in reconstruction["training_frame_ids"])
        )
        if len(streams) != 2:
            raise DomainValidationError("registration requires two observed states")
        for stream in streams:
            if sum(streams_by_id[key] == stream for key in train_ids) < 2 or not any(
                streams_by_id[key] == stream for key in test_ids
            ):
                raise DomainValidationError(
                    "each registration state needs train and held-out views"
                )
        groups: dict[str, dict[str, Any]] = {name: {"train": [], "test": []} for name in streams}
        for view in views:
            xyz = (view.rays * view.depth[..., None] - view.pose[:, 3]) @ view.pose[:, :3]
            points = xyz[view.mask & (view.depth > 0)]
            indices = generator.choice(
                len(points), min(len(points), config.points_per_view), replace=False
            )
            split = "train" if view.frame_id in train_ids else "test"
            groups[view.stream][split].append(points[indices] @ basis.T)
        fractions = {}
        plane_z = float(support["origin"] @ normal)
        for stream, partitions in groups.items():
            for split, rows in partitions.items():
                cloud = np.concatenate(rows)
                foreground = connected_foreground(cloud, plane_z, config.component_voxel_m)
                fraction = len(foreground) / len(cloud)
                if fraction < config.min_foreground_fraction:
                    raise MethodExecutionError("registration foreground coverage is insufficient")
                fractions[f"{stream}/{split}"] = fraction
                partitions[split] = foreground
        meshes = {}
        for link in reference["links"]:
            mesh = trimesh.util.concatenate(
                load_meshes(artifacts, ArtifactRef.model_validate(link["geometry"]))
            )
            meshes[link["name"]] = trimesh.sample.sample_surface(
                mesh, config.mesh_samples, seed=config.seed
            )[0]
        body = meshes[reference["root_link"]]
        z_min, z_max = body[:, 2].min(), body[:, 2].max()
        top = body[body[:, 2] > z_min + 0.68 * (z_max - z_min)]
        fit = np.concatenate([part["train"] for part in groups.values()])
        upper = fit[fit[:, 2] - plane_z > 0.7 * (z_max - z_min)]
        if len(upper) < 100:
            raise MethodExecutionError("registration lacks the upper static reference surface")
        translation = np.median(top, axis=0) - np.median(upper, axis=0)
        translation[2] = z_min - plane_z
        tree = cKDTree(top)
        for _ in range(25):
            distances, ids = tree.query(upper + translation)
            keep = distances < np.quantile(distances, 0.8)
            delta = np.median(top[ids[keep]] - upper[keep] - translation, axis=0)
            translation += delta
            if np.linalg.norm(delta) < 1e-6:
                break
        moving = meshes[joint["child"]]
        static = np.concatenate(
            [points for name, points in meshes.items() if name != joint["child"]]
        )
        rotation, translation, positions, cost = fit_articulated_surfaces(
            [groups[name]["train"] for name in streams],
            static,
            moving,
            translation,
            (joint["lower_limit"], joint["upper_limit"]),
            seed=config.seed,
            max_points=config.fit_points_per_state,
        )
        evaluations = {}
        for index, stream in enumerate(streams):
            target = cKDTree(np.concatenate([static, moving + np.array([positions[index], 0, 0])]))
            points = groups[stream]["test"] @ rotation.T + translation
            distances = target.query(points)[0]
            median, p90 = np.quantile(distances, [0.5, 0.9])
            if median > config.max_holdout_median_m or p90 > config.max_holdout_p90_m:
                raise MethodExecutionError("reference registration failed held-out surface gates")
            evaluations[stream] = {
                "median_m": float(median),
                "p90_m": float(p90),
                "points": len(points),
                "joint_position": float(positions[index]),
            }
        rotation = rotation @ basis
        source_frame = camera["coordinate_frame"]
        for stream in camera["streams"]:
            for component in stream["components"]:
                for image in component["images"]:
                    pose = np.asarray(image["cam_from_world"], dtype=float)
                    rotated = pose[:, :3] @ rotation.T
                    image["cam_from_world"] = np.column_stack(
                        [rotated, pose[:, 3] - rotated @ translation]
                    ).tolist()
                for track in component["fiducial_tracks"]:
                    track["xyz"] = (np.asarray(track["xyz"]) @ rotation.T + translation).tolist()
        camera.update(
            coordinate_frame=reference["coordinate_frame"],
            source_camera_artifact_id=camera_ref.artifact_id,
            metric_alignment_evidence_artifact_id=evidence.artifact_id,
        )
        camera["coordinate_convention"]["frame"] = reference["coordinate_frame"]
        metrics = {
            "max_holdout_median_m": max(row["median_m"] for row in evaluations.values()),
            "max_holdout_p90_m": max(row["p90_m"] for row in evaluations.values()),
            "fit_cost": cost,
        }
        report = {
            "schema": "physrealx.metric-alignment-report/1",
            "status": "PASSED",
            "method": self.descriptor.model_dump(mode="json"),
            "evidence_artifact_id": evidence.artifact_id,
            "evidence_scope": "model_conditioned_full_pose_registration",
            "source_frame": source_frame,
            "target_frame": reference["coordinate_frame"],
            "source_camera": camera_ref.model_dump(mode="json"),
            "source_reference": request.rgbd.reference_model.model_dump(mode="json"),
            "source_visual": visual.model_dump(mode="json"),
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
            "observed_joint": config.joint,
            "state_order": list(streams),
            "evaluations": evaluations,
            "foreground_fractions": fractions,
            "train_frame_ids": sorted(train_ids),
            "holdout_frame_ids": sorted(test_ids),
            "metrics": metrics,
            "motion_groups": {
                str(observed[0]["id"]): [joint["child"]],
                "0": [name for name in meshes if name != joint["child"]],
            },
            "unobserved_joint_motion": [j["name"] for j in reference["joints"] if j != joint],
            "gates": {"held_out_views": True, "independent_metric_truth": False},
        }
        return publish(
            spec,
            artifacts,
            {"camera_rig_calibration": camera, "metric_alignment_report": report},
            metrics=metrics,
        )


def load_registration(store: ArtifactStore, reference: ArtifactRef) -> dict[str, Any]:
    """Validate the portable appearance binding before assembly or rendering."""
    import numpy as np

    report = read_json(store, reference, "physrealx.metric-alignment-report/1")
    rotation, translation = (
        np.asarray(report.get("rotation")),
        np.asarray(report.get("translation_m")),
    )
    if (
        report.get("status") != "PASSED"
        or report.get("evidence_scope") != "model_conditioned_full_pose_registration"
        or report.get("target_frame") != "reference_z_up"
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise DomainValidationError(
            "registered appearance requires a finite proper rigid transform"
        )
    for key in ("source_camera", "source_reference", "source_visual"):
        if not store.verify(ArtifactRef.model_validate(report[key])):
            raise DomainValidationError("registered appearance source is unavailable or modified")
    model = read_json(
        store, ArtifactRef.model_validate(report["source_reference"]), "physrealx.usd-reference/1"
    )
    covered = [link for values in report["motion_groups"].values() for link in values]
    if len(set(covered)) != len(covered) or set(covered) != {
        link["name"] for link in model["links"]
    }:
        raise DomainValidationError("appearance motion groups must cover each reference link once")
    if set(report["train_frame_ids"]) & set(report["holdout_frame_ids"]):
        raise DomainValidationError("appearance registration has overlapping evaluation views")
    if len(report["state_order"]) != 2 or set(report["state_order"]) != set(report["evaluations"]):
        raise DomainValidationError("appearance registration needs two ordered observed states")
    return report
