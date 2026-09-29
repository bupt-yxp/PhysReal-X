"""Validated PartNet-Mobility URDF import into the canonical asset contract."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PositiveFloat

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
PARTNET_IMPORT_REVISION = "visual-submeshes-v2"


@dataclass(frozen=True)
class PartNetSceneAsset:
    """One ranked asset selection from the complete local PartNet-Mobility library."""

    name: str
    category: str
    model_id: str
    target_extent_m: float
    score: float
    score_components: dict[str, float]


class SceneSmithAssetRequirement(DomainModel):
    """Task-grounded retrieval requirement consumed by the SceneSmith bridge."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    query: str = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=1)
    target_extent_m: float = Field(gt=0, allow_inf_nan=False)
    target_dimensions_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = None
    role: Literal["support", "context", "target"] = "context"
    required_joint_types: tuple[Literal["revolute", "prismatic"], ...] = ()


class SceneSmithSceneCatalog(DomainModel):
    """Scene semantics without pinning any PartNet model identifier."""

    schema_id: Literal["physrealx.scenesmith-partnet-catalog/1"] = Field(
        default="physrealx.scenesmith-partnet-catalog/1", alias="schema"
    )
    profiles: dict[PartNetSceneType, tuple[SceneSmithAssetRequirement, ...]]


@dataclass(frozen=True)
class PartNetSceneCandidate:
    """One diversified SceneSmith retrieval candidate."""

    candidate_id: str
    assets: tuple[PartNetSceneAsset, ...]
    score: float


@dataclass(frozen=True)
class PartNetRetrievalResult:
    """Auditable full-library retrieval result before assets are registered."""

    method: Literal["scenesmith"]
    instruction: str
    scene_type: PartNetSceneType
    candidates: tuple[PartNetSceneCandidate, ...]
    evaluated_models: int
    valid_models: int
    rejected_models: int


@dataclass(frozen=True)
class _PartNetModelRecord:
    model_id: str
    category: str
    extent: float
    dimensions: tuple[float, float, float]
    joint_types: frozenset[str]
    movable_joint_count: int
    visual_mesh_count: int
    collision_mesh_count: int


def _partnet_record(model: Path) -> _PartNetModelRecord | None:
    """Read only evidence required for retrieval and reject incomplete packages."""

    try:
        model_id = model.name
        if not model_id.isdigit():
            return None
        category = str(json.loads((model / "meta.json").read_text())["model_cat"])
        bounds = json.loads((model / "bounding_box.json").read_text())
        lower = tuple(float(value) for value in bounds["min"])
        upper = tuple(float(value) for value in bounds["max"])
        robot = ET.parse(model / "mobility.urdf").getroot()
    except (OSError, KeyError, TypeError, ValueError, ET.ParseError, json.JSONDecodeError):
        return None
    extents = tuple(high - low for low, high in zip(lower, upper))
    if (
        len(lower) != 3
        or len(upper) != 3
        or any(not math.isfinite(value) or value <= 0 for value in extents)
        or robot.tag != "robot"
        or not category
    ):
        return None

    mesh_counts: dict[str, int] = {"visual": 0, "collision": 0}
    try:
        for tag in mesh_counts:
            for node in robot.findall(f"link/{tag}/geometry/mesh"):
                filename = node.get("filename")
                if not filename:
                    return None
                mesh = (model / filename).resolve(strict=True)
                if model.resolve() not in mesh.parents or mesh.suffix.lower() != ".obj":
                    return None
                if mesh.stat().st_size <= 0:
                    return None
                mesh_counts[tag] += 1
    except OSError:
        return None
    if mesh_counts["visual"] == 0 or mesh_counts["collision"] == 0:
        return None

    joint_types = frozenset(
        str(node.get("type"))
        for node in robot.findall("joint")
        if node.get("type") in {"revolute", "continuous", "prismatic"}
    )
    if "continuous" in joint_types:
        joint_types = frozenset(
            "revolute" if item == "continuous" else item for item in joint_types
        )
    return _PartNetModelRecord(
        model_id=model_id,
        category=category,
        extent=max(extents),
        dimensions=extents,
        joint_types=joint_types,
        movable_joint_count=sum(
            node.get("type") in {"revolute", "continuous", "prismatic"}
            for node in robot.findall("joint")
        ),
        visual_mesh_count=mesh_counts["visual"],
        collision_mesh_count=mesh_counts["collision"],
    )


@lru_cache(maxsize=4)
def _scan_partnet_library(
    root: Path, model_ids: tuple[str, ...]
) -> tuple[_PartNetModelRecord, ...]:
    """Validate one immutable library snapshot once per process.

    The model-id tuple invalidates the cache when packages are added or removed.
    Canonical import revalidates every selected package before registration, so a
    source file changed in place cannot silently enter the asset registry.
    """

    return tuple(
        record
        for model_id in model_ids
        if (record := _partnet_record(root / model_id)) is not None
    )


def _instruction_relevance(instruction: str, requirement: SceneSmithAssetRequirement) -> float:
    normalized = instruction.casefold().replace(" ", "")
    terms = {requirement.query.casefold().replace(" ", "")}
    category_terms = {
        "Laptop": ("laptop", "笔记本", "电脑"),
        "Bottle": ("bottle", "瓶", "瓶盖"),
        "StorageFurniture": ("cabinet", "storage", "柜", "储物柜", "柜门"),
        "Safe": ("safe", "保险柜"),
        "Table": ("table", "desk", "桌", "工作台"),
        "Chair": ("chair", "椅"),
        "FoldingChair": ("chair", "椅"),
        "Lamp": ("lamp", "灯"),
        "Cart": ("cart", "推车"),
        "Box": ("box", "箱", "盒"),
    }
    for category in requirement.categories:
        terms.update(category_terms.get(category, (category.casefold(),)))
    return 1.0 if any(term and term in normalized for term in terms) else 0.5


def _rank_record(
    record: _PartNetModelRecord,
    requirement: SceneSmithAssetRequirement,
    instruction: str,
    *,
    seed: int,
) -> PartNetSceneAsset | None:
    if record.category not in requirement.categories:
        return None
    required = set(requirement.required_joint_types)
    if required and not required.intersection(record.joint_types):
        return None
    semantic = _instruction_relevance(instruction, requirement)
    functionality = (
        1.0
        if not required
        else min(1.0, record.movable_joint_count / max(1, len(required)))
    )
    geometry = min(1.0, min(record.visual_mesh_count, record.collision_mesh_count) / 2.0)
    scale = 1.0 / (1.0 + abs(math.log(requirement.target_extent_m / record.extent)))
    shape = 1.0
    if requirement.target_dimensions_m is not None:
        # PartNet-Mobility is Z-up but has no stable front axis, so compare the
        # horizontal dimensions independent of yaw and preserve vertical scale.
        source = (*sorted(record.dimensions[:2], reverse=True), record.dimensions[2])
        target = (
            *sorted(requirement.target_dimensions_m[:2], reverse=True),
            requirement.target_dimensions_m[2],
        )
        source_ratio = tuple(value / max(source) for value in source)
        target_ratio = tuple(float(value) / max(target) for value in target)
        shape_error = sum(
            abs(math.log(wanted / observed))
            for wanted, observed in zip(target_ratio, source_ratio)
        ) / 3
        shape = 1.0 / (1.0 + shape_error)
    # Stable tie breaking produces reproducible diversity without depending on directory order.
    digest = hashlib.sha256(f"{seed}:{requirement.name}:{record.model_id}".encode()).digest()
    tie_break = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    components = {
        "semantic": semantic,
        "function": functionality,
        "geometry": geometry,
        "scale": scale,
        "shape": shape,
    }
    score = (
        0.25 * semantic
        + 0.25 * functionality
        + 0.10 * geometry
        + 0.10 * scale
        + 0.30 * shape
    )
    score += tie_break * 1e-6
    return PartNetSceneAsset(
        name=requirement.name,
        category=record.category,
        model_id=record.model_id,
        target_extent_m=requirement.target_extent_m,
        score=score,
        score_components=components,
    )


def retrieve_partnet_scene_assets(
    catalog: str | Path,
    *,
    scene_type: PartNetSceneType,
    instruction: str,
    seed: int,
    candidate_count: int = 3,
) -> PartNetRetrievalResult:
    """Search the full PM library and return up to the requested candidate count.

    This is the dependency-light PhysReal-X bridge for SceneSmith's articulated
    retrieval stages: semantic filtering, function gating, size re-ranking and
    multi-candidate output. It consumes the complete local library rather than a
    hand-picked model-id list.
    """
    if type(seed) is not int or seed < 0:
        raise DomainValidationError("PartNet-Mobility selection seed must be non-negative")
    if not instruction.strip():
        raise DomainValidationError("SceneSmith retrieval requires a natural-language task")
    if type(candidate_count) is not int or not 1 <= candidate_count <= 16:
        raise DomainValidationError("SceneSmith candidate_count must be between 1 and 16")
    root = Path(catalog).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DomainValidationError("PartNet-Mobility catalog must be a directory")
    try:
        profile_catalog = SceneSmithSceneCatalog.model_validate_json(
            (root / "catalog.json").read_text()
        )
    except (OSError, ValueError) as error:
        raise DomainValidationError(
            "PartNet-Mobility catalog requires SceneSmith semantic profiles"
        ) from error
    profile = profile_catalog.profiles.get(scene_type)
    if not profile:
        raise DomainValidationError(
            f"PartNet-Mobility catalog does not define scene type {scene_type!r}"
        )
    model_directories = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    model_ids = tuple(path.name for path in model_directories)
    records = _scan_partnet_library(root, model_ids)
    ranked: dict[str, list[PartNetSceneAsset]] = {}
    for requirement in profile:
        matches = [
            result
            for record in records
            if (
                result := _rank_record(record, requirement, instruction, seed=seed)
            )
            is not None
        ]
        matches.sort(key=lambda item: (-item.score, int(item.model_id)))
        if not matches:
            raise DomainValidationError(
                f"SceneSmith found no functional PM asset for role {requirement.name!r}"
            )
        ranked[requirement.name] = matches

    candidates: list[PartNetSceneCandidate] = []
    used_signatures: set[tuple[str, ...]] = set()
    for candidate_index in range(candidate_count * 4):
        chosen: list[PartNetSceneAsset] = []
        used_models: set[str] = set()
        for requirement_index, requirement in enumerate(profile):
            pool = ranked[requirement.name]
            start = (candidate_index + requirement_index * 3) % min(len(pool), 16)
            asset = next(
                (
                    pool[(start + offset) % min(len(pool), 16)]
                    for offset in range(min(len(pool), 16))
                    if pool[(start + offset) % min(len(pool), 16)].model_id not in used_models
                ),
                pool[start],
            )
            chosen.append(asset)
            used_models.add(asset.model_id)
        signature = tuple(item.model_id for item in chosen)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        payload = f"{scene_type}:{seed}:" + ":".join(signature)
        candidates.append(
            PartNetSceneCandidate(
                candidate_id=f"pm-{hashlib.sha256(payload.encode()).hexdigest()[:12]}",
                assets=tuple(chosen),
                score=sum(item.score for item in chosen) / len(chosen),
            )
        )
        if len(candidates) == candidate_count:
            break
    if not candidates:
        raise DomainValidationError("SceneSmith could not construct a distinct PM candidate")
    return PartNetRetrievalResult(
        method="scenesmith",
        instruction=instruction.strip(),
        scene_type=scene_type,
        candidates=tuple(candidates),
        evaluated_models=len(model_directories),
        valid_models=len(records),
        rejected_models=len(model_directories) - len(records),
    )


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
    source_faces = {role.value: 0 for role in GeometryRole}
    imported_faces = {role.value: 0 for role in GeometryRole}
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
                    source_faces[role.value] += len(part.faces)
                    if len(part.vertices) < 3 or len(part.faces) < 1:
                        continue
                    if not np.isfinite(part.vertices).all() or not np.isfinite(
                        part.face_normals
                    ).all():
                        raise DomainValidationError(
                            "PartNet-Mobility mesh contains non-finite geometry"
                        )
                    parts.append(part)
                    imported_faces[role.value] += len(part.faces)
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
            "import_revision": PARTNET_IMPORT_REVISION,
            "model_id": model_id,
            "category": category,
            "target_extent_m": target_extent_m,
            "source_extent": _model_extent(model),
            "uniform_scale": scale,
            "source_faces": source_faces,
            "imported_faces": imported_faces,
            "orientation": "urdf_joint_frames",
            "collision_proxy": "per_link_axis_aligned_box",
            "self_collision": "disabled_for_conservative_link_proxies",
            "joint_dynamics": "geometry_derived_holding_prior",
        },
    )
