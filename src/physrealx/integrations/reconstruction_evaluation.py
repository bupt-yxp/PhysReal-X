"""DTArt articulated reconstruction metrics with explicit reference provenance.

One fixed capture-to-reference pose is used for every part and state. Missing or
merged parts remain visible in the report; evaluation never repairs reconstruction.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.registration import load_registration
from physrealx.integrations.usd_reference import reference_link_transforms
from physrealx.reconstruction import ReconstructionPrediction, joint_transform

_SURFACE_SAMPLES = 10000


class ReferenceEvaluationPose(DomainModel):
    """Externally supplied pose, correspondence and ordered capture-state annotations."""

    rotation: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    translation_m: tuple[float, float, float]
    source_frame: str = Field(min_length=1)
    target_frame: str = Field(min_length=1)
    part_mapping: dict[str, tuple[str, ...]]
    start_positions: dict[str, float]
    end_positions: dict[str, float]
    evidence_scope: Literal["capture_annotations", "model_fitted"]


class ReconstructionEvaluationInput(DomainModel):
    schema_id: Literal["physrealx.reconstruction-evaluation-input/1"] = Field(
        default="physrealx.reconstruction-evaluation-input/1", alias="schema"
    )
    reconstruction_report: ArtifactRef
    reference_model: ArtifactRef
    registration: ArtifactRef | None = None
    pose: ReferenceEvaluationPose | None = None
    state_pair: tuple[str, str] | None = None
    # Select joint/child-part metrics only; whole/static geometry always remains complete.
    joint_names: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def one_pose_source(self) -> Self:
        if (self.registration is None) == (self.pose is None):
            raise ValueError("provide exactly one registration report or explicit reference pose")
        if self.joint_names is not None and len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("evaluation joint_names must be unique")
        return self


class ReconstructionEvaluationConfig(DomainModel):
    seed: int = Field(default=2026, ge=0)


def chamfer_distance(predicted: Any, reference: Any) -> dict[str, float]:
    """DTArt's symmetric squared L2 CD times 1000 (10^-3 m² for metre inputs)."""
    import numpy as np
    from scipy.spatial import cKDTree

    for points in (predicted, reference):
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise DomainValidationError("CD requires nonempty finite Nx3 point clouds")
    forward = cKDTree(reference).query(predicted)[0]
    backward = cKDTree(predicted).query(reference)[0]
    return {
        "cd": float(np.mean(forward**2) + np.mean(backward**2)) * 1000,
    }


def _axis(value: Any) -> Any:
    import numpy as np

    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-12:
        raise DomainValidationError("joint axis must be a finite nonzero 3-vector")
    return vector / np.linalg.norm(vector)


def joint_errors(
    kind: str,
    predicted_axis: Any,
    predicted_origin: Any,
    predicted_span: float,
    reference_axis: Any,
    reference_origin: Any,
    reference_span: float,
) -> dict[str, Any]:
    """Axis sign is immaterial to the line, but remains coupled to signed motion."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    a, b = _axis(predicted_axis), _axis(reference_axis)
    p, q = np.asarray(predicted_origin, float), np.asarray(reference_origin, float)
    if (
        p.shape != (3,)
        or q.shape != (3,)
        or not np.isfinite([p, q]).all()
        or not np.isfinite([predicted_span, reference_span]).all()
    ):
        raise DomainValidationError("joint origins and observed spans must be finite")
    result: dict[str, Any] = {
        "axis_ang_deg": float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))),
    }
    if kind == "slide":
        result.update(
            axis_pos=None,
            part_motion=float(np.linalg.norm(a * predicted_span - b * reference_span)),
            part_motion_unit="m",
        )
    elif kind == "hinge":
        cross = np.cross(a, b)
        distance = (
            np.linalg.norm(np.cross(q - p, a))
            if np.linalg.norm(cross) < 1e-6
            else abs((q - p) @ cross) / np.linalg.norm(cross)
        )
        motion = Rotation.from_rotvec(a * predicted_span).inv() * Rotation.from_rotvec(
            b * reference_span
        )
        result.update(
            axis_pos=float(distance) * 10,
            part_motion=float(np.degrees(motion.magnitude())),
            part_motion_unit="deg",
        )
    else:
        raise DomainValidationError("joint metrics support slide and hinge only")
    return result


def _prediction(
    store: ArtifactStore, report: dict[str, Any], state_pair: tuple[str, str] | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[str, str], bool]:
    """Consume the public reconstruction contract without inspecting method metadata."""
    import numpy as np

    trimesh = import_module("trimesh")
    try:
        prediction = ReconstructionPrediction.model_validate(report.get("prediction"))
    except ValueError as error:
        raise DomainValidationError(f"invalid normalized reconstruction: {error}") from error
    pair = state_pair or prediction.motion_pair
    if pair is None or pair[0] == pair[1] or not set(pair) <= prediction.states.keys():
        raise DomainValidationError("select two distinct published states with state_pair")
    start, end = prediction.states[pair[0]], prediction.states[pair[1]]
    if set(start) != prediction.parts.keys() or (
        set(end) != prediction.parts.keys() and pair != prediction.motion_pair
    ):
        raise DomainValidationError(
            "selected states need complete meshes or declared forward motion"
        )
    # Reversing/selecting a different pair requires freshly normalized motion, not
    # reusing displacement annotations for another pair.
    if prediction.motion_pair is not None and pair != prediction.motion_pair:
        raise DomainValidationError("selected states differ from predicted joint motion_pair")

    def meshes(refs: dict[str, ArtifactRef]) -> dict[str, Any]:
        return {
            name: trimesh.util.concatenate(load_meshes(store, ref, process=False))
            for name, ref in refs.items()
        }

    joints = {
        name: dict(
            kind=part.kind,
            parent=part.parent,
            axis=_axis(part.axis),
            origin=np.asarray(part.origin),
            span=part.span,
        )
        for name, part in prediction.parts.items()
    }
    return meshes(start), joints, meshes(end), pair, prediction.segmented


def _pose(
    store: ArtifactStore,
    request: ReconstructionEvaluationInput,
    reconstruction: dict[str, Any],
    reference: dict[str, Any],
    state_pair: tuple[str, str],
) -> tuple[ReferenceEvaluationPose, list[str]]:
    assumptions: list[str] = []
    if request.pose is not None:
        return request.pose, assumptions
    assert request.registration is not None
    registration = load_registration(store, request.registration)
    if (
        ArtifactRef.model_validate(registration["source_reference"]) != request.reference_model
        or registration["source_visual"] != reconstruction["bundle"]
        or registration["source_camera"]["artifact_id"]
        != reconstruction["source_camera_artifact_id"]
    ):
        raise DomainValidationError(
            "evaluation registration belongs to different reconstruction/reference"
        )
    if tuple(registration["state_order"]) != state_pair:
        raise DomainValidationError("registration state order differs from evaluation state_pair")
    positions = []
    for state in registration["state_order"]:
        values = {joint["name"]: joint["lower_limit"] for joint in reference["joints"]}
        values[registration["observed_joint"]] = registration["evaluations"][state][
            "joint_position"
        ]
        positions.append(values)
    assumptions = [
        f"{name}: unobserved reference joint assumed at lower limit in both states"
        for name in registration["unobserved_joint_motion"]
    ]
    return ReferenceEvaluationPose(
        rotation=registration["rotation"],
        translation_m=registration["translation_m"],
        source_frame=registration["source_frame"],
        target_frame=registration["target_frame"],
        part_mapping=registration["motion_groups"],
        start_positions=positions[0],
        end_positions=positions[1],
        evidence_scope="model_fitted",
    ), assumptions


def _sample(meshes: list[Any], count: int, generator: Any) -> Any:
    """Area-weighted union sampling; no equal-per-link bias in the whole-object CD."""
    import numpy as np

    triangles = np.concatenate([mesh.triangles for mesh in meshes])
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise DomainValidationError("surface sampling requires positive finite triangle area")
    selected = triangles[generator.choice(len(triangles), count, p=area / area.sum())]
    uv = generator.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return (
        selected[:, 0]
        + uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    )


def filter_prediction(mesh: Any) -> tuple[Any, dict[str, int]]:
    """DTArt edge-connected triangles, retaining components >= 10% of the largest.

    Preserve vertex indices: welding coincident vertices would change the topology.
    The sparse face/edge incidence graph also handles nonmanifold shared edges.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(mesh.faces)
    if not len(faces):
        raise DomainValidationError("DTArt filtering requires nonempty triangle geometry")
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, edge_ids = np.unique(edges, axis=0, return_inverse=True)
    face_ids = np.repeat(np.arange(len(faces)), 3)
    edge_ids = edge_ids + len(faces)
    graph = coo_matrix(
        (np.ones(len(face_ids)), (face_ids, edge_ids)),
        shape=(int(edge_ids.max()) + 1, int(edge_ids.max()) + 1),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels[: len(faces)])
    retained = counts[labels[: len(faces)]] >= counts.max() * 0.1
    filtered = mesh.copy()
    filtered.update_faces(retained)
    filtered.remove_unreferenced_vertices()
    return filtered, {
        "input_triangles": len(faces),
        "retained_triangles": int(retained.sum()),
        "input_components": int(np.count_nonzero(counts)),
        "retained_components": int(np.count_nonzero(counts >= counts.max() * 0.1)),
    }


def evaluate_reconstruction(
    store: ArtifactStore,
    request: ReconstructionEvaluationInput,
    config: ReconstructionEvaluationConfig,
) -> dict[str, Any]:
    import numpy as np

    trimesh = import_module("trimesh")
    reconstruction = read_json(
        store, request.reconstruction_report, "physrealx.geometry-reconstruction-report/1"
    )
    reference = read_json(store, request.reference_model, "physrealx.usd-reference/1")
    if reconstruction.get("unit") != "m" or reference.get("unit") != "m":
        raise DomainValidationError("reconstruction evaluation requires metre geometry")
    predicted, joints, end_meshes, state_pair, segmented = _prediction(
        store, reconstruction, request.state_pair
    )
    pose, assumptions = _pose(store, request, reconstruction, reference, state_pair)
    rotation, translation = np.asarray(pose.rotation), np.asarray(pose.translation_m)
    if (
        pose.source_frame != reconstruction["coordinate_frame"]
        or pose.target_frame != reference["coordinate_frame"]
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)
    ):
        raise DomainValidationError(
            "evaluation requires a proper rigid pose between declared frames"
        )
    ref_meshes = {
        row["name"]: trimesh.util.concatenate(
            load_meshes(store, ArtifactRef.model_validate(row["geometry"]), process=False)
        )
        for row in reference["links"]
    }
    if not segmented and pose.part_mapping:
        raise DomainValidationError("unsegmented reconstruction cannot declare part correspondence")
    mapping = pose.part_mapping
    covered = [link for links in mapping.values() for link in links]
    if (
        len(ref_meshes) != len(reference["links"])
        or not set(mapping) <= set(predicted)
        or any(not links for links in mapping.values())
        or len(covered) != len(set(covered))
        or not set(covered) <= set(ref_meshes)
    ):
        raise DomainValidationError(
            "part correspondence has duplicate or unknown groups/reference links"
        )
    reference_transforms = [
        reference_link_transforms(reference, positions)
        for positions in (pose.start_positions, pose.end_positions)
    ]
    if any(set(transforms) != set(ref_meshes) for transforms in reference_transforms):
        raise DomainValidationError("reference kinematics must cover all reference links")
    reference_joints = reference["joints"]
    if request.joint_names is not None:
        unknown = set(request.joint_names) - {joint["name"] for joint in reference_joints}
        if unknown:
            raise DomainValidationError(f"unknown evaluation joints: {sorted(unknown)}")
        reference_joints = [
            joint for joint in reference_joints if joint["name"] in request.joint_names
        ]
    part_names = (
        set(ref_meshes)
        if request.joint_names is None
        else {joint["child"] for joint in reference_joints}
    )
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    end_transforms: dict[str, Any] = {}
    remaining = dict(joints)
    while remaining:
        ready = [
            name
            for name, joint in remaining.items()
            if joint["parent"] is None or joint["parent"] in end_transforms
        ]
        if not ready:
            raise DomainValidationError("predicted joint graph is disconnected or cyclic")
        for name in ready:
            joint = remaining.pop(name)
            joint["axis"] = rotation @ joint["axis"]
            joint["origin"] = rotation @ joint["origin"] + translation
            parent = end_transforms.get(joint["parent"], np.eye(4))
            end_transforms[name] = parent @ joint_transform(
                joint["kind"], joint["axis"], joint["origin"], joint["span"]
            )
    geometry: dict[str, Any] = {}
    for index, state in enumerate(("start", "end")):
        pred_state, ref_state = {}, {}
        for name, mesh in predicted.items():
            native_end = index and name in end_meshes
            pred_state[name] = (end_meshes[name] if native_end else mesh).copy()
            pred_state[name].apply_transform(
                (end_transforms[name] if index and not native_end else np.eye(4)) @ transform
            )
        for name, mesh in ref_meshes.items():
            ref_state[name] = mesh.copy()
            ref_state[name].apply_transform(reference_transforms[index][name])

        def score(
            groups: list[str],
            links: list[str],
            pred_state: dict[str, Any] = pred_state,
            ref_state: dict[str, Any] = ref_state,
        ) -> dict[str, Any]:
            # Reuse the seed, not generator state: identical surfaces/state pairs
            # must give identical scores, including a static group and its union.
            generator = np.random.default_rng(config.seed)
            # Filter the union separately for whole/static CD, as DTArt does;
            # filtering each part first changes the largest-component threshold.
            prediction, preprocessing = filter_prediction(
                trimesh.util.concatenate([pred_state[name] for name in groups])
            )
            return {
                **chamfer_distance(
                    _sample([prediction], _SURFACE_SAMPLES, generator),
                    _sample([ref_state[name] for name in links], _SURFACE_SAMPLES, generator),
                ),
                "preprocessing": preprocessing,
            }

        group_scores = {
            name: {"reference_links": list(links), **score([name], list(links))}
            for name, links in sorted(mapping.items())
            if len(links) == 1 and links[0] in part_names
        }
        per_link = {}
        for name in sorted(part_names):
            group = next((group for group, links in mapping.items() if name in links), None)
            status = (
                "unsegmented"
                if not segmented
                else "missing"
                if group is None
                else "merged"
                if len(mapping[group]) > 1
                else "measured"
            )
            per_link[name] = {
                "status": status,
                "predicted_group": group,
                "cd": group_scores[group]["cd"]
                if group is not None and status == "measured"
                else None,
                "preprocessing": group_scores[group]["preprocessing"]
                if group is not None and status == "measured"
                else None,
            }
        static_groups = []
        for name in joints:
            ancestor: str | None = name
            while ancestor is not None and joints[ancestor]["kind"] == "fixed":
                ancestor = joints[ancestor]["parent"]
            if ancestor is None and segmented:
                static_groups.append(name)
        static_links = [link for name in static_groups for link in mapping.get(name, ())]
        geometry[state] = {
            "whole": score(list(predicted), list(ref_meshes)),
            "static": {"reference_links": static_links, **score(static_groups, static_links)}
            if static_links
            else {"cd": None, "status": "unmapped"},
            "parts": per_link,
        }
    joint_metrics = {}
    for reference_joint in reference_joints:
        name, child = reference_joint["name"], reference_joint["child"]
        group = next((group for group, links in mapping.items() if child in links), None)
        row: dict[str, Any] = {
            "reference_child": child,
            "predicted_group": group,
            "joint_type": reference_joint["joint_type"],
            "axis_ang_deg": None,
            "axis_pos": None,
            "part_motion": None,
            "part_motion_unit": "deg" if reference_joint["joint_type"] == "hinge" else "m",
        }
        if not segmented:
            row["status"] = "unsegmented"
        elif group is None:
            row["status"] = "missing"
        elif len(mapping[group]) != 1 or joints[group]["kind"] == "fixed":
            row["status"] = "unobserved_or_merged"
        elif joints[group]["kind"] != reference_joint["joint_type"]:
            row["status"] = "joint_type_mismatch"
        elif reference_joint["parent"] not in mapping.get(joints[group]["parent"], ()):
            row["status"] = "parent_mismatch"
        elif abs(pose.end_positions[name] - pose.start_positions[name]) < 1e-10:
            row["status"] = "reference_motion_unobserved"
        elif abs(joints[group]["span"]) < 1e-10:
            row["status"] = "predicted_motion_degenerate"
        else:
            predicted_joint = joints[group]
            parent = reference_transforms[0][reference_joint["parent"]]
            reference_axis = parent[:3, :3] @ _axis(reference_joint["axis"])
            reference_origin = (
                parent[:3, :3] @ np.asarray(reference_joint["origin_m"]) + parent[:3, 3]
            )
            row.update(
                status="measured",
                **joint_errors(
                    reference_joint["joint_type"],
                    predicted_joint["axis"],
                    predicted_joint["origin"],
                    predicted_joint["span"],
                    reference_axis,
                    reference_origin,
                    pose.end_positions[name] - pose.start_positions[name],
                ),
            )
        joint_metrics[name] = row
    measured_parts = sum(row["status"] == "measured" for row in geometry["start"]["parts"].values())
    measured_joints = sum(row["status"] == "measured" for row in joint_metrics.values())
    return {
        "schema": "physrealx.reconstruction-evaluation-report/2",
        "status": "COMPLETED",
        "quality_verdict": "not_thresholded",
        "source": request.model_dump(mode="json", by_alias=True),
        "reference_pose": pose.model_dump(mode="json"),
        "state_pair": list(state_pair),
        "independent_physical_validation": False,
        "metric_evidence_scope": reconstruction.get("metric_evidence_scope", "unspecified"),
        "state_evidence_scope": pose.evidence_scope,
        "assumptions": assumptions,
        "protocol": {
            "name": "dtart",
            "source": "https://github.com/NVlabs/DigitalTwinArt/tree/master/eval",
            "cd": "mean squared nearest distance P->Q + mean squared nearest distance Q->P",
            "cd_multiplier": 1000,
            "cd_unit": "10^-3 m^2",
            "axis_pos_multiplier": 10,
            "axis_pos_unit": "10^-1 m",
            "prediction_filter": "edge-connected triangle count >= 0.1 * largest component",
            "reference_filter": "none",
            "surface_samples_per_comparison_side": _SURFACE_SAMPLES,
            "seed": config.seed,
            "alignment": "one fixed rigid transform; no per-part ICP",
            "geometry_scope": "filtered predicted surfaces versus full supplied reference surfaces",
            "start_geometry": "published reconstructed start meshes",
            "end_geometry": {
                name: "published state mesh"
                if name in end_meshes
                else "start mesh transformed by recovered motion"
                for name in predicted
            },
            "static_scope": "recovered fixed groups; may include unexcited reference movable parts",
            "sampling": "seeded area-weighted surface samples; finite-sampling error remains",
            "part_motion": "SO(3) rotation error in degrees or displacement vector error in metres",
        },
        "coverage": {
            "reference_parts": len(ref_meshes),
            "individually_evaluated_parts": measured_parts,
            "reference_joints": len(reference["joints"]),
            "evaluated_joints": measured_joints,
            "unmapped_predicted_groups": sorted(set(predicted) - set(mapping)),
            "missing_reference_parts": sorted(set(ref_meshes) - set(covered)),
        },
        "geometry": geometry,
        "joints": joint_metrics,
    }
