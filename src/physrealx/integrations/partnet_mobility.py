"""Validated PartNet-Mobility URDF import into the canonical asset contract."""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    AssetEvidenceState,
    AssetSpec,
    BodyPhysicsSpec,
    DomainModel,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    JointSpec,
    JointType,
    ParameterSource,
    PoseSpec,
    ScalarParameter,
)
from physrealx.errors import DomainValidationError

PartNetSceneType = Literal["home", "tabletop", "shelf", "industrial"]


@dataclass(frozen=True)
class PartNetSceneAsset:
    """One deterministic asset selection from a local PartNet-Mobility catalog."""

    name: str
    category: str
    model_id: str
    target_extent_m: float


class PartNetSceneProfileEntry(DomainModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    category: str = Field(min_length=1)
    model_ids: tuple[str, ...] = Field(min_length=1)
    target_extent_m: float = Field(gt=0, allow_inf_nan=False)


class PartNetSceneCatalog(DomainModel):
    schema_id: Literal["physrealx.partnet-scene-catalog/1"] = Field(
        default="physrealx.partnet-scene-catalog/1", alias="schema"
    )
    profiles: dict[PartNetSceneType, tuple[PartNetSceneProfileEntry, ...]]


def select_partnet_scene_assets(
    catalog: str | Path,
    *,
    scene_type: PartNetSceneType,
    seed: int,
) -> tuple[PartNetSceneAsset, ...]:
    """Select a reproducible scene profile from a validated local catalog."""
    if type(seed) is not int or seed < 0:
        raise DomainValidationError("PartNet-Mobility selection seed must be non-negative")
    root = Path(catalog).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DomainValidationError("PartNet-Mobility catalog must be a directory")
    try:
        profile_catalog = PartNetSceneCatalog.model_validate_json(
            (root / "catalog.json").read_text()
        )
    except (OSError, ValueError) as error:
        raise DomainValidationError(
            "PartNet-Mobility catalog requires a valid catalog.json scene profile"
        ) from error
    profile = profile_catalog.profiles.get(scene_type)
    if not profile:
        raise DomainValidationError(
            f"PartNet-Mobility catalog does not define scene type {scene_type!r}"
        )
    rng = random.Random(seed)
    selected = []
    for item in profile:
        model_id = item.model_ids[rng.randrange(len(item.model_ids))]
        if not model_id.isdigit():
            raise DomainValidationError("PartNet-Mobility profile model IDs must contain digits")
        model = root / model_id
        try:
            category = json.loads((model / "meta.json").read_text())["model_cat"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise DomainValidationError(
                f"PartNet-Mobility profile references invalid model {model_id!r}"
            ) from error
        if (
            category != item.category
            or not (model / "mobility.urdf").is_file()
            or not (model / "bounding_box.json").is_file()
        ):
            raise DomainValidationError(
                f"PartNet-Mobility profile model {model_id!r} does not satisfy {item.category!r}"
            )
        selected.append(
            PartNetSceneAsset(
                name=item.name,
                category=item.category,
                model_id=model_id,
                target_extent_m=item.target_extent_m,
            )
        )
    return tuple(selected)


def _numbers(value: str | None, *, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    try:
        result = tuple(float(item) for item in value.split())
    except ValueError as error:
        raise DomainValidationError("PartNet-Mobility URDF contains non-numeric values") from error
    if len(result) != size or any(not math.isfinite(item) for item in result):
        raise DomainValidationError("PartNet-Mobility URDF vector has invalid dimensions")
    return result


def _origin(node: ET.Element | None, scale: float) -> PoseSpec:
    from scipy.spatial.transform import Rotation

    if node is None:
        return PoseSpec()
    xyz = _numbers(node.get("xyz"), size=3, default=(0.0, 0.0, 0.0))
    rpy = _numbers(node.get("rpy"), size=3, default=(0.0, 0.0, 0.0))
    return PoseSpec(
        translation_m=tuple(scale * value for value in xyz),
        rotation_xyzw=tuple(float(value) for value in Rotation.from_euler("xyz", rpy).as_quat()),
    )


def _prior(value: float) -> ScalarParameter:
    return ScalarParameter(
        value=value,
        uncertainty=abs(value),
        source=ParameterSource.DEFAULT,
        confidence=0.0,
    )


def _safe_model(root: Path, model_id: str) -> Path:
    if not model_id.isdigit():
        raise DomainValidationError("PartNet-Mobility model_id must contain digits only")
    catalog = root.expanduser().resolve(strict=True)
    model = (catalog / model_id).resolve(strict=True)
    if model.parent != catalog or not model.is_dir():
        raise DomainValidationError("PartNet-Mobility model is outside the supplied catalog")
    return model


def _mesh_path(model: Path, filename: str) -> Path:
    candidate = (model / filename).resolve(strict=True)
    if model not in candidate.parents or candidate.suffix.lower() != ".obj":
        raise DomainValidationError("PartNet-Mobility mesh must be an OBJ inside its model")
    return candidate


def _model_extent(model: Path) -> float:
    try:
        bounds = json.loads((model / "bounding_box.json").read_text())
        lower = tuple(float(value) for value in bounds["min"])
        upper = tuple(float(value) for value in bounds["max"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DomainValidationError(
            "PartNet-Mobility model has invalid bounding_box.json"
        ) from error
    extents = tuple(high - low for low, high in zip(lower, upper))
    extent = max(extents, default=0.0)
    if len(lower) != 3 or len(upper) != 3 or not math.isfinite(extent) or extent <= 0:
        raise DomainValidationError("PartNet-Mobility model has invalid bounds")
    return extent


def _link_physics(link: str, vertices: list[Any]) -> BodyPhysicsSpec:
    import numpy as np

    if vertices:
        points = np.concatenate(vertices)
        lower, upper = points.min(axis=0), points.max(axis=0)
        dimensions = np.maximum(upper - lower, 1e-4)
        center = (lower + upper) / 2
        mass = max(0.01, min(float(np.prod(dimensions)) * 250.0, 50.0))
    else:
        dimensions = np.full(3, 1e-3)
        center = np.zeros(3)
        mass = 0.01
    moments = np.maximum(mass * (np.sum(dimensions**2) - dimensions**2) / 12, 1e-8)
    return BodyPhysicsSpec(
        link=link,
        mass_kg=_prior(mass),
        center_of_mass_m=tuple(float(value) for value in center),
        inertia_kg_m2=(
            (float(moments[0]), 0.0, 0.0),
            (0.0, float(moments[1]), 0.0),
            (0.0, 0.0, float(moments[2])),
        ),
        static_friction=_prior(0.6),
        dynamic_friction=_prior(0.5),
        restitution=_prior(0.05),
    )


def import_partnet_mobility_asset(
    artifacts: ArtifactStore,
    catalog: str | Path,
    *,
    model_id: str,
    resource_id: str,
    version: str,
    display_name: str | None,
    target_extent_m: float,
) -> AssetSpec:
    """Import one catalog model, retaining its URDF links and movable joints."""
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    if not math.isfinite(target_extent_m) or target_extent_m <= 0:
        raise DomainValidationError("target_extent_m must be finite and positive")
    model = _safe_model(Path(catalog), model_id)
    try:
        meta = json.loads((model / "meta.json").read_text())
        category = str(meta["model_cat"])
        robot = ET.parse(model / "mobility.urdf").getroot()
    except (OSError, KeyError, ET.ParseError, json.JSONDecodeError) as error:
        raise DomainValidationError("PartNet-Mobility model metadata or URDF is invalid") from error
    if robot.tag != "robot" or not category:
        raise DomainValidationError("PartNet-Mobility package requires robot and model_cat")
    scale = target_extent_m / _model_extent(model)
    link_nodes = robot.findall("link")
    links = tuple(node.get("name", "") for node in link_nodes)
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise DomainValidationError("PartNet-Mobility URDF links must be unique and named")

    geometries: list[GeometrySpec] = []
    link_vertices: dict[str, list[Any]] = {name: [] for name in links}
    for link_node in link_nodes:
        link = str(link_node.get("name"))
        for role, tag in ((GeometryRole.VISUAL, "visual"), (GeometryRole.COLLISION, "collision")):
            parts = []
            for geometry_node in link_node.findall(tag):
                mesh_node = geometry_node.find("geometry/mesh")
                filename = None if mesh_node is None else mesh_node.get("filename")
                if not filename:
                    raise DomainValidationError("PartNet-Mobility geometry requires an OBJ mesh")
                assert mesh_node is not None
                path = _mesh_path(model, filename)
                mesh_scale = _numbers(mesh_node.get("scale"), size=3, default=(1.0, 1.0, 1.0))
                if any(value <= 0 for value in mesh_scale):
                    raise DomainValidationError("PartNet-Mobility mesh scale must be positive")
                local_pose = _origin(geometry_node.find("origin"), scale)
                loaded: Any = trimesh.load(path, force="scene", process=False)
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(local_pose.rotation_xyzw).as_matrix()
                transform[:3, 3] = local_pose.translation_m
                for source_part in loaded.dump():
                    part = source_part.copy()
                    part.apply_scale(np.asarray(mesh_scale) * scale)
                    part.apply_transform(transform)
                    if len(part.vertices) >= 4 and len(part.faces) >= 4:
                        parts.append(part)
            if not parts:
                continue
            merged = trimesh.util.concatenate(parts)
            merged.remove_unreferenced_vertices()
            if role is GeometryRole.COLLISION:
                extents = np.maximum(merged.extents, 1e-4)
                center = merged.bounds.mean(axis=0)
                merged = trimesh.creation.box(extents=extents)
                merged.apply_translation(center)
            reference = artifacts.put_bytes(
                merged.export(file_type="glb"), media_type="model/gltf-binary"
            )
            geometries.append(
                GeometrySpec(
                    name=f"{link}_{role.value}",
                    link=link,
                    role=role,
                    representation=GeometryRepresentation.MESH,
                    uri=f"artifact://{reference.artifact_id}",
                )
            )
            if role is GeometryRole.VISUAL:
                link_vertices[link].append(np.asarray(merged.vertices, dtype=float))

    physics = tuple(_link_physics(link, link_vertices[link]) for link in links)
    physics_by_link = {item.link: item for item in physics}
    joints: list[JointSpec] = []
    for node in robot.findall("joint"):
        name = node.get("name", "")
        parent_node, child_node = node.find("parent"), node.find("child")
        parent = "" if parent_node is None else parent_node.get("link", "")
        child = "" if child_node is None else child_node.get("link", "")
        kind = node.get("type")
        if not name or parent not in links or child not in links:
            raise DomainValidationError("PartNet-Mobility joint has invalid link references")
        if kind == "fixed":
            joint_type, lower, upper = JointType.FIXED, 0.0, 0.0
        elif kind in ("revolute", "continuous"):
            limit = node.find("limit")
            joint_type = JointType.REVOLUTE
            if kind == "continuous":
                lower, upper = -math.pi, math.pi
            else:
                if limit is None:
                    raise DomainValidationError("bounded revolute joint requires limits")
                lower = float(limit.get("lower", "nan"))
                upper = float(limit.get("upper", "nan"))
        elif kind == "prismatic":
            limit = node.find("limit")
            if limit is None:
                raise DomainValidationError("prismatic joint requires limits")
            joint_type = JointType.PRISMATIC
            lower = float(limit.get("lower", "nan")) * scale
            upper = float(limit.get("upper", "nan")) * scale
        else:
            raise DomainValidationError(f"unsupported PartNet-Mobility joint type: {kind!r}")
        axis_node = node.find("axis")
        axis = np.asarray(
            _numbers(
                None if axis_node is None else axis_node.get("xyz"),
                size=3,
                default=(0.0, 0.0, 1.0),
            )
        )
        norm = float(np.linalg.norm(axis))
        if (
            norm <= 0
            or not math.isfinite(norm)
            or not all(math.isfinite(value) for value in (lower, upper))
        ):
            raise DomainValidationError("PartNet-Mobility joint parameters are invalid")
        axis /= norm
        damping = 0.0
        friction = 0.0
        if joint_type is not JointType.FIXED:
            body = physics_by_link[child]
            mass = body.mass_kg.value
            center = np.asarray(body.center_of_mass_m)
            if joint_type is JointType.REVOLUTE:
                lever_arm = float(np.linalg.norm(center - axis * np.dot(center, axis)))
                holding_load = mass * 9.80665 * lever_arm
            else:
                holding_load = mass * 9.80665 * abs(float(axis[2]))
            friction = max(0.01, 1.05 * holding_load)
            damping = max(0.05, 0.05 * friction)
        joints.append(
            JointSpec(
                name=name,
                parent=parent,
                child=child,
                joint_type=joint_type,
                parent_to_joint=_origin(node.find("origin"), scale),
                axis=tuple(float(value) for value in axis),
                lower_limit=lower,
                upper_limit=upper,
                damping=damping,
                friction=friction,
                effort_limit=20.0 if joint_type is not JointType.FIXED else None,
                velocity_limit=2.0 if joint_type is not JointType.FIXED else None,
            )
        )
    children = {joint.child for joint in joints}
    roots = set(links) - children
    if len(roots) != 1:
        raise DomainValidationError("PartNet-Mobility URDF must contain one rooted link tree")
    if not geometries:
        raise DomainValidationError("PartNet-Mobility model contains no mesh geometry")
    return AssetSpec(
        resource_id=resource_id,
        version=version,
        display_name=display_name or f"{category} {model_id}",
        root_link=roots.pop(),
        links=links,
        geometries=tuple(geometries),
        joints=tuple(joints),
        physics=physics,
        contact_exclusions=tuple(combinations(links, 2)),
        evidence_state=AssetEvidenceState.PRIOR,
        metadata={
            "source": "PartNet-Mobility",
            "model_id": model_id,
            "category": category,
            "target_extent_m": target_extent_m,
            "source_extent": _model_extent(model),
            "uniform_scale": scale,
            "orientation": "urdf_joint_frames",
            "collision_proxy": "per_link_axis_aligned_box",
            "self_collision": "disabled_for_conservative_link_proxies",
            "joint_dynamics": "geometry_derived_holding_prior",
        },
    )
