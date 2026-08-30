"""Build a self-contained, physically identified OpenUSD articulation asset.

The adapter deliberately accepts only metric, identified evidence.  It does not
fill missing mass, inertia, friction, damping, joint limits, or scale with
defaults: those values change simulation behaviour and must come from an
immutable evidence artifact in the input resource closure.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from physrealx.external_workers import WorkerProducedFile, WorkerRequest, WorkerResponse, run_worker
from physrealx_workers.common import (
    dependency_path,
    is_probe,
    report_probe,
    resource_json,
)

_ASSEMBLY_SCHEMA = "physrealx.asset-assembly-bundle/1"
_ARTICULATION_SCHEMA = "physrealx.articulation-model/1"
_PHYSICS_SCHEMA = "physrealx.identified-physics/1"
_PACKAGE_SCHEMA = "physrealx.asset-package-manifest/1"
_REPORT_SCHEMA = "physrealx.usd-roundtrip-validation-report/1"
_IDENTIFIED_STATES = {"IDENTIFIED", "VALIDATED"}
_IDENTIFIED_EVIDENCE = {"identified", "validated"}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_MESH_MEDIA = {
    "model/obj": "obj",
    "model/ply": "ply",
    "model/stl": "stl",
    "model/gltf-binary": "glb",
}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a valid USD identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _artifact(value: Any, label: str) -> dict[str, Any]:
    item = _object(value, label)
    _keys(
        item,
        required={"digest", "size_bytes", "media_type"},
        label=label,
    )
    _digest(item["digest"], f"{label}.digest")
    if isinstance(item["size_bytes"], bool) or not isinstance(item["size_bytes"], int):
        raise ValueError(f"{label}.size_bytes must be a non-negative integer")
    if item["size_bytes"] < 0:
        raise ValueError(f"{label}.size_bytes must be a non-negative integer")
    if not isinstance(item["media_type"], str) or not item["media_type"]:
        raise ValueError(f"{label}.media_type must be a non-empty string")
    return item


def _vector(value: Any, label: str, size: int) -> tuple[float, ...]:
    items = _array(value, label)
    if len(items) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(items))


def _pose(value: Any, label: str) -> dict[str, tuple[float, ...]]:
    pose = _object(value, label)
    _keys(pose, required={"translation_m", "rotation_xyzw"}, label=label)
    translation = _vector(pose["translation_m"], f"{label}.translation_m", 3)
    rotation = _vector(pose["rotation_xyzw"], f"{label}.rotation_xyzw", 4)
    norm = math.sqrt(sum(component * component for component in rotation))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{label}.rotation_xyzw must be a unit quaternion")
    return {"translation_m": translation, "rotation_xyzw": rotation}


def _metric_scale(value: Any, label: str) -> dict[str, Any]:
    scale = _object(value, label)
    _keys(
        scale,
        required={"status", "meters_per_unit", "evidence_digest"},
        label=label,
    )
    if scale["status"] != "resolved":
        raise ValueError("metric scale is unresolved; OpenUSD packaging is forbidden")
    meters_per_unit = _number(
        scale["meters_per_unit"],
        f"{label}.meters_per_unit",
        minimum=1e-12,
    )
    return {
        "status": "resolved",
        "meters_per_unit": meters_per_unit,
        "evidence_digest": _digest(scale["evidence_digest"], f"{label}.evidence_digest"),
    }


def _parameter(
    value: Any,
    label: str,
    *,
    unit: str,
    evidence_digests: set[str],
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    parameter = _object(value, label)
    _keys(
        parameter,
        required={"value", "unit", "source", "confidence_interval", "evidence_digest"},
        label=label,
    )
    if parameter["unit"] != unit:
        raise ValueError(f"{label}.unit must be {unit!r}")
    if parameter["source"] not in {"measured", "identified"}:
        raise ValueError(f"{label}.source must be measured or identified")
    result = _number(parameter["value"], f"{label}.value", minimum=minimum)
    if maximum is not None and result > maximum:
        raise ValueError(f"{label}.value must not exceed {maximum}")
    interval = _vector(parameter["confidence_interval"], f"{label}.confidence_interval", 2)
    if interval[0] > result or result > interval[1]:
        raise ValueError(f"{label}.confidence_interval must contain the identified value")
    evidence_digest = _digest(parameter["evidence_digest"], f"{label}.evidence_digest")
    if evidence_digest not in evidence_digests:
        raise ValueError(f"{label} refers to evidence outside the assembly closure")
    return {
        "value": result,
        "unit": unit,
        "source": parameter["source"],
        "confidence_interval": interval,
        "evidence_digest": evidence_digest,
    }


def _validate_assembly(document: dict[str, Any]) -> dict[str, Any]:
    _keys(
        document,
        required={
            "schema",
            "resource_id",
            "version",
            "metric_scale",
            "up_axis",
            "root_link",
            "visual_meshes",
            "collision_meshes",
            "articulation_model",
            "identified_physics",
            "evidence",
        },
        optional={"display_name"},
        label="AssetAssemblyBundle",
    )
    if document["schema"] != _ASSEMBLY_SCHEMA:
        raise ValueError(f"AssetAssemblyBundle.schema must be {_ASSEMBLY_SCHEMA!r}")
    resource_id = document["resource_id"]
    version = document["version"]
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("AssetAssemblyBundle.resource_id must be a non-empty string")
    if not isinstance(version, str) or not version:
        raise ValueError("AssetAssemblyBundle.version must be a non-empty string")
    up_axis = document["up_axis"]
    if up_axis not in {"Y", "Z"}:
        raise ValueError("AssetAssemblyBundle.up_axis must be 'Y' or 'Z'")
    root_link = _token(document["root_link"], "AssetAssemblyBundle.root_link")
    scale = _metric_scale(document["metric_scale"], "AssetAssemblyBundle.metric_scale")
    evidence = [
        _artifact(item, f"AssetAssemblyBundle.evidence[{index}]")
        for index, item in enumerate(_array(document["evidence"], "AssetAssemblyBundle.evidence"))
    ]
    if not evidence:
        raise ValueError("AssetAssemblyBundle.evidence must not be empty")
    evidence_digests = {str(item["digest"]) for item in evidence}
    if len(evidence_digests) != len(evidence):
        raise ValueError("AssetAssemblyBundle.evidence digests must be unique")
    if scale["evidence_digest"] not in evidence_digests:
        raise ValueError("metric scale evidence is outside the assembly closure")

    mesh_names: set[str] = set()
    mesh_digests: set[str] = set()

    def meshes(field: str, role: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        values = _array(document[field], f"AssetAssemblyBundle.{field}")
        if not values:
            raise ValueError(f"AssetAssemblyBundle.{field} must not be empty")
        for index, raw in enumerate(values):
            label = f"AssetAssemblyBundle.{field}[{index}]"
            item = _object(raw, label)
            _keys(
                item,
                required={"name", "link", "artifact", "units", "local_pose"},
                label=label,
            )
            name = _token(item["name"], f"{label}.name")
            link = _token(item["link"], f"{label}.link")
            if name in mesh_names:
                raise ValueError(f"duplicate geometry name: {name}")
            mesh_names.add(name)
            artifact = _artifact(item["artifact"], f"{label}.artifact")
            media_type = str(artifact["media_type"])
            if media_type not in _SUPPORTED_MESH_MEDIA:
                raise ValueError(f"{label}.artifact has unsupported mesh media type {media_type!r}")
            digest = str(artifact["digest"])
            if digest in mesh_digests:
                raise ValueError("each geometry must use a distinct immutable artifact")
            mesh_digests.add(digest)
            if item["units"] != "meters":
                raise ValueError(f"{label}.units must be 'meters'")
            result.append(
                {
                    "name": name,
                    "link": link,
                    "role": role,
                    "artifact": artifact,
                    "local_pose": _pose(item["local_pose"], f"{label}.local_pose"),
                }
            )
        return result

    visual_meshes = meshes("visual_meshes", "visual")
    collision_meshes = meshes("collision_meshes", "collision")
    articulation = _artifact(document["articulation_model"], "articulation_model")
    physics = _artifact(document["identified_physics"], "identified_physics")
    if articulation["media_type"] != "application/json":
        raise ValueError("articulation_model must be an application/json artifact")
    if physics["media_type"] != "application/json":
        raise ValueError("identified_physics must be an application/json artifact")
    all_digests = (
        mesh_digests
        | evidence_digests
        | {
            str(articulation["digest"]),
            str(physics["digest"]),
        }
    )
    expected_count = len(mesh_digests) + len(evidence_digests) + 2
    if len(all_digests) != expected_count:
        raise ValueError("all AssetAssemblyBundle artifact digests must be unique")
    return {
        "resource_id": resource_id,
        "version": version,
        "display_name": document.get("display_name", resource_id),
        "up_axis": up_axis,
        "root_link": root_link,
        "metric_scale": scale,
        "visual_meshes": visual_meshes,
        "collision_meshes": collision_meshes,
        "articulation_model": articulation,
        "identified_physics": physics,
        "evidence": evidence,
        "evidence_digests": evidence_digests,
        "artifact_digests": all_digests,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, member in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = member
        return value

    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not unambiguous UTF-8 JSON") from error
    return _object(document, label)


def _validate_articulation(
    document: dict[str, Any],
    assembly: dict[str, Any],
) -> dict[str, Any]:
    _keys(
        document,
        required={
            "schema",
            "status",
            "evidence_state",
            "metric_scale",
            "links",
            "joints",
            "evidence_digest",
        },
        label="ArticulationModel",
    )
    if document["schema"] != _ARTICULATION_SCHEMA:
        raise ValueError(f"ArticulationModel.schema must be {_ARTICULATION_SCHEMA!r}")
    if document["status"] not in _IDENTIFIED_STATES:
        raise ValueError("ArticulationModel must be IDENTIFIED or VALIDATED")
    if document["evidence_state"] not in _IDENTIFIED_EVIDENCE:
        raise ValueError("ArticulationModel cannot use prior-only evidence")
    scale = _metric_scale(document["metric_scale"], "ArticulationModel.metric_scale")
    if not math.isclose(
        scale["meters_per_unit"],
        assembly["metric_scale"]["meters_per_unit"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("ArticulationModel and AssetAssemblyBundle scale disagree")
    evidence_digest = _digest(document["evidence_digest"], "ArticulationModel.evidence_digest")
    if evidence_digest not in assembly["evidence_digests"]:
        raise ValueError("ArticulationModel evidence is outside the assembly closure")
    links = [
        _token(value, f"ArticulationModel.links[{index}]")
        for index, value in enumerate(_array(document["links"], "ArticulationModel.links"))
    ]
    if not links or len(links) != len(set(links)):
        raise ValueError("ArticulationModel.links must be non-empty and unique")
    if assembly["root_link"] not in links:
        raise ValueError("assembly root_link is missing from ArticulationModel.links")
    joints: list[dict[str, Any]] = []
    joint_names: set[str] = set()
    children: set[str] = set()
    parents: dict[str, str] = {}
    for index, raw in enumerate(_array(document["joints"], "ArticulationModel.joints")):
        label = f"ArticulationModel.joints[{index}]"
        item = _object(raw, label)
        _keys(
            item,
            required={
                "name",
                "parent",
                "child",
                "type",
                "axis",
                "lower_limit",
                "upper_limit",
                "limit_unit",
                "parent_frame",
                "child_frame",
            },
            label=label,
        )
        name = _token(item["name"], f"{label}.name")
        parent = _token(item["parent"], f"{label}.parent")
        child = _token(item["child"], f"{label}.child")
        if name in joint_names:
            raise ValueError(f"duplicate articulation joint name: {name}")
        if parent not in links or child not in links or parent == child:
            raise ValueError(f"{label} must connect two distinct declared links")
        if child in children:
            raise ValueError(f"link {child!r} has more than one parent joint")
        joint_type = item["type"]
        if joint_type not in {"fixed", "revolute", "prismatic"}:
            raise ValueError(f"{label}.type is unsupported")
        axis = item["axis"]
        if axis not in {"X", "Y", "Z"}:
            raise ValueError(f"{label}.axis must be X, Y, or Z")
        lower = _number(item["lower_limit"], f"{label}.lower_limit")
        upper = _number(item["upper_limit"], f"{label}.upper_limit")
        expected_unit = {"fixed": "none", "revolute": "rad", "prismatic": "m"}[joint_type]
        if item["limit_unit"] != expected_unit:
            raise ValueError(f"{label}.limit_unit must be {expected_unit!r}")
        if lower > upper:
            raise ValueError(f"{label} lower_limit exceeds upper_limit")
        if joint_type == "fixed" and (lower != 0.0 or upper != 0.0):
            raise ValueError(f"{label} fixed limits must both be zero")
        joint_names.add(name)
        children.add(child)
        parents[child] = parent
        joints.append(
            {
                "name": name,
                "parent": parent,
                "child": child,
                "type": joint_type,
                "axis": axis,
                "lower_limit": lower,
                "upper_limit": upper,
                "limit_unit": expected_unit,
                "parent_frame": _pose(item["parent_frame"], f"{label}.parent_frame"),
                "child_frame": _pose(item["child_frame"], f"{label}.child_frame"),
            }
        )
    root = assembly["root_link"]
    if root in children or children != set(links) - {root}:
        raise ValueError("ArticulationModel joints must form one rooted link tree")
    for link in links:
        seen: set[str] = set()
        current = link
        while current in parents:
            if current in seen:
                raise ValueError("ArticulationModel joint graph contains a cycle")
            seen.add(current)
            current = parents[current]
        if current != root:
            raise ValueError("ArticulationModel contains a link disconnected from the root")
    geometry_links = {
        item["link"] for item in assembly["visual_meshes"] + assembly["collision_meshes"]
    }
    if not geometry_links.issubset(set(links)):
        raise ValueError("an assembly geometry refers to an undeclared articulation link")
    visual_links = {item["link"] for item in assembly["visual_meshes"]}
    collision_links = {item["link"] for item in assembly["collision_meshes"]}
    if visual_links != set(links) or collision_links != set(links):
        raise ValueError("every articulation link requires visual and collision geometry")
    return {"links": links, "joints": joints, "evidence_digest": evidence_digest}


def _validate_physics(
    document: dict[str, Any],
    assembly: dict[str, Any],
    articulation: dict[str, Any],
) -> dict[str, Any]:
    _keys(
        document,
        required={"schema", "status", "evidence_state", "metric_scale", "bodies", "joints"},
        label="IdentifiedPhysics",
    )
    if document["schema"] != _PHYSICS_SCHEMA:
        raise ValueError(f"IdentifiedPhysics.schema must be {_PHYSICS_SCHEMA!r}")
    if document["status"] not in _IDENTIFIED_STATES:
        raise ValueError("IdentifiedPhysics must be IDENTIFIED or VALIDATED")
    if document["evidence_state"] not in _IDENTIFIED_EVIDENCE:
        raise ValueError("IdentifiedPhysics cannot use prior or default parameters")
    scale = _metric_scale(document["metric_scale"], "IdentifiedPhysics.metric_scale")
    if not math.isclose(
        scale["meters_per_unit"],
        assembly["metric_scale"]["meters_per_unit"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("IdentifiedPhysics and AssetAssemblyBundle scale disagree")
    if scale["evidence_digest"] not in assembly["evidence_digests"]:
        raise ValueError("IdentifiedPhysics scale evidence is outside the assembly closure")
    evidence_digests = assembly["evidence_digests"]
    bodies: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_array(document["bodies"], "IdentifiedPhysics.bodies")):
        label = f"IdentifiedPhysics.bodies[{index}]"
        item = _object(raw, label)
        _keys(
            item,
            required={
                "link",
                "mass_kg",
                "center_of_mass_m",
                "inertia_kg_m2",
                "static_friction",
                "dynamic_friction",
                "restitution",
                "evidence_digest",
            },
            label=label,
        )
        link = _token(item["link"], f"{label}.link")
        if link in bodies:
            raise ValueError(f"duplicate IdentifiedPhysics body: {link}")
        evidence_digest = _digest(item["evidence_digest"], f"{label}.evidence_digest")
        if evidence_digest not in evidence_digests:
            raise ValueError(f"{label} evidence is outside the assembly closure")
        inertia_rows = _array(item["inertia_kg_m2"], f"{label}.inertia_kg_m2")
        if len(inertia_rows) != 3:
            raise ValueError(f"{label}.inertia_kg_m2 must be a 3x3 matrix")
        inertia = tuple(
            _vector(row, f"{label}.inertia_kg_m2[{row_index}]", 3)
            for row_index, row in enumerate(inertia_rows)
        )
        if any(
            abs(inertia[row][column] - inertia[column][row]) > 1e-9
            for row in range(3)
            for column in range(3)
        ):
            raise ValueError(f"{label}.inertia_kg_m2 must be symmetric")
        first_minor = inertia[0][0]
        second_minor = inertia[0][0] * inertia[1][1] - inertia[0][1] * inertia[1][0]
        determinant = (
            inertia[0][0] * (inertia[1][1] * inertia[2][2] - inertia[1][2] * inertia[2][1])
            - inertia[0][1] * (inertia[1][0] * inertia[2][2] - inertia[1][2] * inertia[2][0])
            + inertia[0][2] * (inertia[1][0] * inertia[2][1] - inertia[1][1] * inertia[2][0])
        )
        if first_minor <= 0 or second_minor <= 0 or determinant <= 0:
            raise ValueError(f"{label}.inertia_kg_m2 must be positive definite")
        bodies[link] = {
            "link": link,
            "mass_kg": _parameter(
                item["mass_kg"],
                f"{label}.mass_kg",
                unit="kg",
                minimum=1e-12,
                evidence_digests=evidence_digests,
            ),
            "center_of_mass_m": _vector(item["center_of_mass_m"], f"{label}.center_of_mass_m", 3),
            "inertia_kg_m2": inertia,
            "static_friction": _parameter(
                item["static_friction"],
                f"{label}.static_friction",
                unit="1",
                minimum=0.0,
                evidence_digests=evidence_digests,
            ),
            "dynamic_friction": _parameter(
                item["dynamic_friction"],
                f"{label}.dynamic_friction",
                unit="1",
                minimum=0.0,
                evidence_digests=evidence_digests,
            ),
            "restitution": _parameter(
                item["restitution"],
                f"{label}.restitution",
                unit="1",
                minimum=0.0,
                maximum=1.0,
                evidence_digests=evidence_digests,
            ),
            "evidence_digest": evidence_digest,
        }
    if set(bodies) != set(articulation["links"]):
        raise ValueError("IdentifiedPhysics must contain one body for every articulation link")

    articulation_joints = {item["name"]: item for item in articulation["joints"]}
    joints: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_array(document["joints"], "IdentifiedPhysics.joints")):
        label = f"IdentifiedPhysics.joints[{index}]"
        item = _object(raw, label)
        _keys(
            item,
            required={"name", "damping", "friction", "stiffness", "evidence_digest"},
            label=label,
        )
        name = _token(item["name"], f"{label}.name")
        if name in joints or name not in articulation_joints:
            raise ValueError(f"{label}.name is duplicate or absent from ArticulationModel")
        evidence_digest = _digest(item["evidence_digest"], f"{label}.evidence_digest")
        if evidence_digest not in evidence_digests:
            raise ValueError(f"{label} evidence is outside the assembly closure")
        joint_type = articulation_joints[name]["type"]
        units = {
            "fixed": ("1", "1", "1"),
            "revolute": ("N*m*s/rad", "N*m", "N*m/rad"),
            "prismatic": ("N*s/m", "N", "N/m"),
        }[joint_type]
        joints[name] = {
            "name": name,
            "damping": _parameter(
                item["damping"],
                f"{label}.damping",
                unit=units[0],
                minimum=0.0,
                evidence_digests=evidence_digests,
            ),
            "friction": _parameter(
                item["friction"],
                f"{label}.friction",
                unit=units[1],
                minimum=0.0,
                evidence_digests=evidence_digests,
            ),
            "stiffness": _parameter(
                item["stiffness"],
                f"{label}.stiffness",
                unit=units[2],
                minimum=0.0,
                evidence_digests=evidence_digests,
            ),
            "evidence_digest": evidence_digest,
        }
    if set(joints) != set(articulation_joints):
        raise ValueError("IdentifiedPhysics must contain one entry for every articulation joint")
    return {"bodies": bodies, "joints": joints, "metric_scale": scale}


def _load_inputs(request: WorkerRequest) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assembly = _validate_assembly(resource_json(request, "asset"))
    actual_digests = {item.digest for item in request.resource_artifacts if item.owner == "asset"}
    if actual_digests != assembly["artifact_digests"]:
        raise ValueError("AssetAssemblyBundle resource closure is incomplete or contains extras")
    articulation_path = dependency_path(
        request, "asset", str(assembly["articulation_model"]["digest"])
    )
    physics_path = dependency_path(request, "asset", str(assembly["identified_physics"]["digest"]))
    articulation = _validate_articulation(
        _read_json(articulation_path, "ArticulationModel"), assembly
    )
    physics = _validate_physics(
        _read_json(physics_path, "IdentifiedPhysics"), assembly, articulation
    )
    return assembly, articulation, physics


def _quaternion(rotation_xyzw: tuple[float, ...], gf: Any) -> Any:
    x, y, z, w = rotation_xyzw
    return gf.Quatf(float(w), gf.Vec3f(float(x), float(y), float(z)))


def _rotation_quaternion(matrix: Any, gf: Any) -> Any:
    """Return an xyzw quaternion for a proper 3x3 rotation matrix."""

    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = float(matrix[2, 1] - matrix[1, 2]) / scale
        y = float(matrix[0, 2] - matrix[2, 0]) / scale
        z = float(matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = [float(matrix[index, index]) for index in range(3)]
        index = max(range(3), key=diagonal.__getitem__)
        if index == 0:
            scale = math.sqrt(1.0 + diagonal[0] - diagonal[1] - diagonal[2]) * 2.0
            w = float(matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = float(matrix[0, 1] + matrix[1, 0]) / scale
            z = float(matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + diagonal[1] - diagonal[0] - diagonal[2]) * 2.0
            w = float(matrix[0, 2] - matrix[2, 0]) / scale
            x = float(matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = float(matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + diagonal[2] - diagonal[0] - diagonal[1]) * 2.0
            w = float(matrix[1, 0] - matrix[0, 1]) / scale
            x = float(matrix[0, 2] + matrix[2, 0]) / scale
            y = float(matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return gf.Quatf(float(w), gf.Vec3f(float(x), float(y), float(z)))


def _add_pose(
    xformable: Any, pose: dict[str, tuple[float, ...]], meters_per_unit: float, gf: Any
) -> None:
    translation = pose["translation_m"]
    xformable.AddTranslateOp().Set(
        gf.Vec3d(*(float(component) / meters_per_unit for component in translation))
    )
    xformable.AddOrientOp().Set(_quaternion(pose["rotation_xyzw"], gf))


def _mesh_from_file(path: Path, media_type: str, trimesh: Any, np: Any) -> tuple[Any, Any]:
    loaded = trimesh.load(
        path,
        file_type=_SUPPORTED_MESH_MEDIA[media_type],
        force="mesh",
        process=False,
    )
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"mesh artifact did not decode to one triangle mesh: {path.name}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"mesh artifact has invalid vertices: {path.name}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise ValueError(f"mesh artifact must contain triangles: {path.name}")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"mesh artifact contains non-finite or invalid topology: {path.name}")
    return vertices, faces


def _author_stage(
    destination: Path,
    assembly: dict[str, Any],
    articulation: dict[str, Any],
    physics: dict[str, Any],
    request: WorkerRequest,
) -> dict[str, Any]:
    import numpy as np
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    major = int(np.__version__.split(".", 1)[0])
    if major != 1:
        raise RuntimeError("OpenUSD worker requires NumPy 1.x")
    meters_per_unit = float(assembly["metric_scale"]["meters_per_unit"])
    stage = Usd.Stage.CreateNew(str(destination))
    if stage is None:
        raise RuntimeError("OpenUSD failed to create the output stage")
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.z if assembly["up_axis"] == "Z" else UsdGeom.Tokens.y,
    )
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("physrealx:resourceId", assembly["resource_id"])
    root.GetPrim().SetCustomDataByKey("physrealx:version", assembly["version"])
    root.GetPrim().SetCustomDataByKey("physrealx:evidenceState", "identified")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    link_prims: dict[str, Any] = {}
    materials: dict[str, Any] = {}
    for link in articulation["links"]:
        link_prim = UsdGeom.Xform.Define(stage, f"/Asset/Links/{link}").GetPrim()
        link_prims[link] = link_prim
        rigid = UsdPhysics.RigidBodyAPI.Apply(link_prim)
        rigid.CreateRigidBodyEnabledAttr().Set(link != assembly["root_link"])
        rigid.CreateKinematicEnabledAttr().Set(link == assembly["root_link"])
        body = physics["bodies"][link]
        inertia = np.asarray(body["inertia_kg_m2"], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1.0
        if not np.all(eigenvalues > 0):
            raise ValueError(f"body {link!r} has non-positive principal inertia")
        mass_api = UsdPhysics.MassAPI.Apply(link_prim)
        mass_api.CreateMassAttr().Set(float(body["mass_kg"]["value"]))
        mass_api.CreateCenterOfMassAttr().Set(
            Gf.Vec3f(*(float(value) / meters_per_unit for value in body["center_of_mass_m"]))
        )
        mass_api.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(*(float(value) / meters_per_unit**2 for value in eigenvalues))
        )
        mass_api.CreatePrincipalAxesAttr().Set(_rotation_quaternion(eigenvectors, Gf))
        link_prim.CreateAttribute("physrealx:physicsEvidence", Sdf.ValueTypeNames.String).Set(
            body["evidence_digest"]
        )

        material = UsdShade.Material.Define(stage, f"/Asset/PhysicsMaterials/{link}")
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(float(body["static_friction"]["value"]))
        material_api.CreateDynamicFrictionAttr().Set(float(body["dynamic_friction"]["value"]))
        material_api.CreateRestitutionAttr().Set(float(body["restitution"]["value"]))
        material.GetPrim().CreateAttribute(
            "physrealx:physicsEvidence", Sdf.ValueTypeNames.String
        ).Set(body["evidence_digest"])
        materials[link] = material

    mesh_counts: dict[str, dict[str, int]] = {}
    for item in assembly["visual_meshes"] + assembly["collision_meshes"]:
        artifact = item["artifact"]
        source = dependency_path(request, "asset", str(artifact["digest"]))
        vertices, faces = _mesh_from_file(source, str(artifact["media_type"]), trimesh, np)
        mesh = UsdGeom.Mesh.Define(
            stage,
            f"/Asset/Links/{item['link']}/{item['role'].title()}/{item['name']}",
        )
        mesh.CreatePointsAttr().Set(
            [
                Gf.Vec3f(*(float(component) / meters_per_unit for component in vertex))
                for vertex in vertices
            ]
        )
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr().Set([int(value) for value in faces.reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        _add_pose(UsdGeom.Xformable(mesh.GetPrim()), item["local_pose"], meters_per_unit, Gf)
        mesh.GetPrim().CreateAttribute("physrealx:sourceDigest", Sdf.ValueTypeNames.String).Set(
            artifact["digest"]
        )
        mesh.GetPrim().CreateAttribute("physrealx:role", Sdf.ValueTypeNames.Token).Set(item["role"])
        if item["role"] == "collision":
            collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            collision.CreateCollisionEnabledAttr().Set(True)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                materials[item["link"]],
                materialPurpose="physics",
            )
        mesh_counts[item["name"]] = {"vertices": len(vertices), "triangles": len(faces)}

    for item in articulation["joints"]:
        path = f"/Asset/Joints/{item['name']}"
        if item["type"] == "fixed":
            joint = UsdPhysics.FixedJoint.Define(stage, path)
        elif item["type"] == "revolute":
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            joint.CreateAxisAttr().Set(item["axis"])
            joint.CreateLowerLimitAttr().Set(math.degrees(float(item["lower_limit"])))
            joint.CreateUpperLimitAttr().Set(math.degrees(float(item["upper_limit"])))
        else:
            joint = UsdPhysics.PrismaticJoint.Define(stage, path)
            joint.CreateAxisAttr().Set(item["axis"])
            joint.CreateLowerLimitAttr().Set(float(item["lower_limit"]) / meters_per_unit)
            joint.CreateUpperLimitAttr().Set(float(item["upper_limit"]) / meters_per_unit)
        joint.CreateBody0Rel().SetTargets([link_prims[item["parent"]].GetPath()])
        joint.CreateBody1Rel().SetTargets([link_prims[item["child"]].GetPath()])
        parent_frame = item["parent_frame"]
        child_frame = item["child_frame"]
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*(float(value) / meters_per_unit for value in parent_frame["translation_m"]))
        )
        joint.CreateLocalRot0Attr().Set(_quaternion(parent_frame["rotation_xyzw"], Gf))
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(*(float(value) / meters_per_unit for value in child_frame["translation_m"]))
        )
        joint.CreateLocalRot1Attr().Set(_quaternion(child_frame["rotation_xyzw"], Gf))
        identified = physics["joints"][item["name"]]
        joint.GetPrim().CreateAttribute("physrealx:jointFriction", Sdf.ValueTypeNames.Double).Set(
            float(identified["friction"]["value"])
        )
        joint.GetPrim().CreateAttribute("physrealx:physicsEvidence", Sdf.ValueTypeNames.String).Set(
            identified["evidence_digest"]
        )
        if item["type"] != "fixed":
            drive_name = "angular" if item["type"] == "revolute" else "linear"
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_name)
            drive.CreateTypeAttr().Set("force")
            drive.CreateDampingAttr().Set(float(identified["damping"]["value"]))
            drive.CreateStiffnessAttr().Set(float(identified["stiffness"]["value"]))

    root.GetPrim().CreateAttribute("physrealx:closureComplete", Sdf.ValueTypeNames.Bool).Set(True)
    stage.GetRootLayer().Save()
    return mesh_counts


def _close_enough(actual: Any, expected: float, tolerance: float = 1e-5) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)


def _validate_roundtrip(
    path: Path,
    assembly: dict[str, Any],
    articulation: dict[str, Any],
    physics: dict[str, Any],
    mesh_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError("the authored USD stage cannot be reopened")
    checks: dict[str, bool] = {}
    meters_per_unit = float(assembly["metric_scale"]["meters_per_unit"])
    checks["meters_per_unit"] = _close_enough(UsdGeom.GetStageMetersPerUnit(stage), meters_per_unit)
    checks["up_axis"] = str(UsdGeom.GetStageUpAxis(stage)) == assembly["up_axis"]
    root_layer = stage.GetRootLayer()
    checks["no_sublayers"] = not root_layer.subLayerPaths
    checks["no_external_references"] = not root_layer.GetExternalReferences()
    asset_paths: list[str] = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() in {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray}:
                value = attribute.Get()
                if value:
                    asset_paths.append(str(value))
    checks["no_asset_paths"] = not asset_paths
    root = stage.GetPrimAtPath("/Asset")
    checks["default_prim"] = bool(root) and stage.GetDefaultPrim() == root
    checks["articulation_root_api"] = bool(root and root.HasAPI(UsdPhysics.ArticulationRootAPI))
    checks["closure_marker"] = bool(root and root.GetAttribute("physrealx:closureComplete").Get())

    mesh_total = 0
    collision_total = 0
    for item in assembly["visual_meshes"] + assembly["collision_meshes"]:
        prim = stage.GetPrimAtPath(
            f"/Asset/Links/{item['link']}/{item['role'].title()}/{item['name']}"
        )
        valid = bool(prim) and prim.IsA(UsdGeom.Mesh)
        checks[f"mesh:{item['name']}"] = valid
        if not valid:
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = mesh_counts[item["name"]]
        checks[f"mesh_topology:{item['name']}"] = (
            len(mesh.GetPointsAttr().Get()) == counts["vertices"]
            and len(mesh.GetFaceVertexCountsAttr().Get()) == counts["triangles"]
            and prim.GetAttribute("physrealx:sourceDigest").Get() == item["artifact"]["digest"]
        )
        if item["role"] == "collision":
            collision_total += 1
            checks[f"collision_api:{item['name']}"] = prim.HasAPI(UsdPhysics.CollisionAPI)
        mesh_total += 1

    for link in articulation["links"]:
        prim = stage.GetPrimAtPath(f"/Asset/Links/{link}")
        body = physics["bodies"][link]
        mass_api = UsdPhysics.MassAPI(prim)
        checks[f"body_api:{link}"] = bool(prim) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        checks[f"mass:{link}"] = bool(mass_api) and _close_enough(
            mass_api.GetMassAttr().Get(), body["mass_kg"]["value"]
        )
        authored_com = mass_api.GetCenterOfMassAttr().Get()
        checks[f"center_of_mass:{link}"] = bool(authored_com) and all(
            _close_enough(authored_com[index], body["center_of_mass_m"][index] / meters_per_unit)
            for index in range(3)
        )
        expected_inertia = (
            np.linalg.eigvalsh(np.asarray(body["inertia_kg_m2"], dtype=np.float64))
            / meters_per_unit**2
        )
        authored_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        checks[f"inertia:{link}"] = bool(authored_inertia) and all(
            _close_enough(authored_inertia[index], expected_inertia[index]) for index in range(3)
        )
        checks[f"physics_evidence:{link}"] = (
            prim.GetAttribute("physrealx:physicsEvidence").Get() == body["evidence_digest"]
        )
        material = UsdShade.Material.Get(stage, f"/Asset/PhysicsMaterials/{link}")
        material_api = UsdPhysics.MaterialAPI(material.GetPrim())
        checks[f"material:{link}"] = (
            bool(material)
            and _close_enough(
                material_api.GetStaticFrictionAttr().Get(), body["static_friction"]["value"]
            )
            and _close_enough(
                material_api.GetDynamicFrictionAttr().Get(), body["dynamic_friction"]["value"]
            )
            and _close_enough(material_api.GetRestitutionAttr().Get(), body["restitution"]["value"])
        )

    for item in assembly["collision_meshes"]:
        prim = stage.GetPrimAtPath(f"/Asset/Links/{item['link']}/Collision/{item['name']}")
        bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        checks[f"material_binding:{item['name']}"] = (
            bool(bound_material)
            and str(bound_material.GetPath()) == f"/Asset/PhysicsMaterials/{item['link']}"
        )

    for item in articulation["joints"]:
        prim = stage.GetPrimAtPath(f"/Asset/Joints/{item['name']}")
        expected_type = {
            "fixed": UsdPhysics.FixedJoint,
            "revolute": UsdPhysics.RevoluteJoint,
            "prismatic": UsdPhysics.PrismaticJoint,
        }[item["type"]]
        checks[f"joint_type:{item['name']}"] = bool(prim) and prim.IsA(expected_type)
        joint = UsdPhysics.Joint(prim)
        checks[f"joint_bodies:{item['name']}"] = joint.GetBody0Rel().GetTargets() == [
            Sdf.Path(f"/Asset/Links/{item['parent']}")
        ] and joint.GetBody1Rel().GetTargets() == [Sdf.Path(f"/Asset/Links/{item['child']}")]
        parent_position = joint.GetLocalPos0Attr().Get()
        child_position = joint.GetLocalPos1Attr().Get()
        checks[f"joint_frames:{item['name']}"] = all(
            _close_enough(
                parent_position[index],
                item["parent_frame"]["translation_m"][index] / meters_per_unit,
            )
            and _close_enough(
                child_position[index],
                item["child_frame"]["translation_m"][index] / meters_per_unit,
            )
            for index in range(3)
        )
        if item["type"] == "revolute":
            typed = UsdPhysics.RevoluteJoint(prim)
            checks[f"joint_axis:{item['name']}"] = str(typed.GetAxisAttr().Get()) == item["axis"]
            checks[f"joint_limits:{item['name']}"] = _close_enough(
                typed.GetLowerLimitAttr().Get(), math.degrees(item["lower_limit"])
            ) and _close_enough(typed.GetUpperLimitAttr().Get(), math.degrees(item["upper_limit"]))
        elif item["type"] == "prismatic":
            typed = UsdPhysics.PrismaticJoint(prim)
            checks[f"joint_axis:{item['name']}"] = str(typed.GetAxisAttr().Get()) == item["axis"]
            checks[f"joint_limits:{item['name']}"] = _close_enough(
                typed.GetLowerLimitAttr().Get(), item["lower_limit"] / meters_per_unit
            ) and _close_enough(
                typed.GetUpperLimitAttr().Get(), item["upper_limit"] / meters_per_unit
            )
        checks[f"joint_evidence:{item['name']}"] = (
            prim.GetAttribute("physrealx:physicsEvidence").Get()
            == physics["joints"][item["name"]]["evidence_digest"]
        )
        identified = physics["joints"][item["name"]]
        checks[f"joint_friction:{item['name']}"] = _close_enough(
            prim.GetAttribute("physrealx:jointFriction").Get(),
            identified["friction"]["value"],
        )
        if item["type"] != "fixed":
            drive_name = "angular" if item["type"] == "revolute" else "linear"
            drive = UsdPhysics.DriveAPI(prim, drive_name)
            checks[f"joint_drive:{item['name']}"] = (
                bool(drive)
                and _close_enough(drive.GetDampingAttr().Get(), identified["damping"]["value"])
                and _close_enough(drive.GetStiffnessAttr().Get(), identified["stiffness"]["value"])
            )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"OpenUSD round-trip validation failed: {', '.join(failed)}")
    return {
        "schema": _REPORT_SCHEMA,
        "status": "PASSED",
        "checks": checks,
        "failed_checks": [],
        "summary": {
            "links": len(articulation["links"]),
            "joints": len(articulation["joints"]),
            "meshes": mesh_total,
            "collision_meshes": collision_total,
            "external_dependencies": 0,
        },
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    assembly, articulation, physics = _load_inputs(request)
    usd_path = output_directory / "asset.usdc"
    mesh_counts = _author_stage(usd_path, assembly, articulation, physics, request)
    usd_digest = _sha256(usd_path)
    report = _validate_roundtrip(usd_path, assembly, articulation, physics, mesh_counts)
    report["usd_digest"] = usd_digest
    report["source_assembly_digest"] = next(
        item.digest for item in request.inputs if item.name == "asset"
    )
    manifest = {
        "schema": _PACKAGE_SCHEMA,
        "resource_id": assembly["resource_id"],
        "version": assembly["version"],
        "format": "usdc",
        "usd": {
            "digest": usd_digest,
            "size_bytes": usd_path.stat().st_size,
            "media_type": "model/vnd.usd+binary",
        },
        "meters_per_unit": assembly["metric_scale"]["meters_per_unit"],
        "up_axis": assembly["up_axis"],
        "root_prim": "/Asset",
        "self_contained": True,
        "external_dependencies": [],
        "source_artifacts": sorted(assembly["artifact_digests"]),
        "evidence_state": "identified",
        "roundtrip_status": "PASSED",
    }
    manifest_path = output_directory / "asset_package_manifest.json"
    report_path = output_directory / "usd_roundtrip_validation_report.json"
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    produced = {
        "usd_asset_bundle": WorkerProducedFile(
            name="usd_asset_bundle",
            contract="UsdAssetBundle/1",
            path=usd_path.name,
            media_type="model/vnd.usd+binary",
        ),
        "asset_package_manifest": WorkerProducedFile(
            name="asset_package_manifest",
            contract="AssetPackageManifest/1",
            path=manifest_path.name,
            media_type="application/json",
        ),
        "roundtrip_validation_report": WorkerProducedFile(
            name="roundtrip_validation_report",
            contract="UsdRoundTripValidationReport/1",
            path=report_path.name,
            media_type="application/json",
        ),
    }
    try:
        outputs = tuple(produced[item.name] for item in request.operation.outputs)
    except KeyError as error:
        raise ValueError(f"unsupported OpenUSD output: {error.args[0]}") from error
    return WorkerResponse(
        outputs=outputs,
        metrics={
            "links": float(len(articulation["links"])),
            "joints": float(len(articulation["joints"])),
            "meshes": float(len(mesh_counts)),
            "roundtrip_checks": float(len(report["checks"])),
            "external_dependencies": 0.0,
        },
    )


def main() -> int:
    if is_probe():
        import numpy
        import trimesh
        from pxr import Usd, UsdGeom, UsdPhysics

        if int(numpy.__version__.split(".", 1)[0]) != 1:
            raise RuntimeError("OpenUSD worker requires NumPy 1.x")
        try:
            usd_core = importlib.metadata.version("usd-core")
        except importlib.metadata.PackageNotFoundError:
            usd_core = ".".join(str(value) for value in Usd.GetVersion())
        with tempfile.TemporaryDirectory(prefix="physrealx-openusd-probe-") as directory:
            fixture = Path(directory) / "articulation.usda"
            stage = Usd.Stage.CreateNew(str(fixture))
            root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
            UsdPhysics.ArticulationRootAPI.Apply(root)
            stage.SetDefaultPrim(root)
            stage.GetRootLayer().Save()
            reopened = Usd.Stage.Open(str(fixture))
            fixture_ready = bool(
                reopened
                and reopened.GetDefaultPrim()
                and reopened.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
            )
        return report_probe(
            capability="asset.package",
            name="openusd",
            version="1.0.0",
            source_revision="adapter-1.0.0",
            dependencies={
                "numpy": numpy.__version__,
                "openusd": usd_core,
                "trimesh": trimesh.__version__,
            },
            self_test_checks={
                "numpy_1x": True,
                "openusd_import": True,
                "trimesh_import": True,
                "articulation_create_reopen": fixture_ready,
            },
            ready=fixture_ready,
            reason=None if fixture_ready else "OpenUSD articulation create/reopen self-test failed",
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
