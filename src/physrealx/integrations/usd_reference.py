"""Import a supplied reverse-engineered USD with its local dependency closure.

This boundary preserves authored geometry/kinematics as model evidence. It does
not turn an annotation or a USD default into a measured physical parameter.
"""

from __future__ import annotations

import io
import math
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.digital_twin import ObservationSetManifest
from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError
from physrealx.integrations.visuals import usd_visual_mesh


def import_usd_reference(
    observations: ObservationSetManifest,
    artifacts: ArtifactStore,
    *,
    usd_path: str,
) -> ArtifactRef:
    """Extract canonical metre/Z-up mesh and joint evidence from imported USD."""
    from importlib import import_module

    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    trimesh = import_module("trimesh")

    path = PurePosixPath(usd_path)
    if path.is_absolute() or any(p in (".", "..") for p in path.parts):
        raise DomainValidationError("USD entry must be a normalized package-relative path")
    attachments = {item.path: item.artifact for item in observations.attachments}
    if usd_path not in attachments:
        raise DomainValidationError(
            "USD entry is absent; ingest the package with candidates included"
        )
    if not all(artifacts.verify(ref) for ref in attachments.values()):
        raise DomainValidationError("reference source contains unavailable attachments")
    with tempfile.TemporaryDirectory(prefix="physrealx-usd-reference-") as directory:
        root = Path(directory)
        for name, ref in attachments.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            artifacts.materialize(ref, target)
        stage = Usd.Stage.Open(str(root / usd_path))
        if not stage or not stage.HasAuthoredMetadata("metersPerUnit"):
            raise DomainValidationError("reference USD must explicitly author metresPerUnit")
        scale = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(scale) or scale <= 0:
            raise DomainValidationError("USD length scale must be finite and positive")
        up = str(UsdGeom.GetStageUpAxis(stage))
        if up not in ("Y", "Z"):
            raise DomainValidationError("reference USD requires Y or Z up")
        basis = np.eye(3) if up == "Z" else np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # Inspect asset paths too: unresolved references must not quietly disappear.
        for layer in stage.GetUsedLayers():
            if not layer.anonymous and not Path(layer.realPath).resolve().is_relative_to(root):
                raise DomainValidationError("USD layer dependency escapes the supplied package")
        shader_dependencies = []
        for prim in stage.Traverse():
            for attribute in prim.GetAttributes():
                value = attribute.Get()
                values = (
                    [value]
                    if isinstance(value, Sdf.AssetPath)
                    else (
                        list(value)
                        if attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray
                        else []
                    )
                )
                for asset in values:
                    resolved = Path(asset.resolvedPath) if asset.resolvedPath else None
                    if asset.path and (
                        resolved is None
                        or not resolved.is_file()
                        or not resolved.resolve().is_relative_to(root)
                    ):
                        if (
                            prim.IsA(UsdShade.Shader)
                            and attribute.GetName().endswith("sourceAsset")
                            and asset.path.endswith(".mdl")
                        ):
                            shader_dependencies.append(asset.path)
                            continue
                        raise DomainValidationError(
                            f"unresolved or external USD asset: {asset.path}"
                        )
        bodies = {
            str(p.GetPath()): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        if not bodies:
            raise DomainValidationError("reference USD has no explicit rigid-body parts")
        if len({body.GetName() for body in bodies.values()}) != len(bodies):
            raise DomainValidationError("reference body names must be unique")
        meshes: dict[str, list[Any]] = {key: [] for key in bodies}
        material_conversions = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            owner = prim.GetParent()
            while owner and str(owner.GetPath()) not in bodies:
                owner = owner.GetParent()
            if not owner:
                raise DomainValidationError(
                    f"mesh is not assigned to a rigid-body part: {prim.GetPath()}"
                )
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not np.isfinite(vertices).all()
                or counts.sum() != len(indices)
                or (counts < 3).any()
                or (indices < 0).any()
                or (indices >= len(vertices)).any()
            ):
                raise DomainValidationError("USD mesh has invalid vertices or polygon topology")
            transform = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=float,
            ).T
            vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) @ basis.T * scale
            triangles: list[tuple[int, int, int]] = []
            corners: list[tuple[int, int, int]] = []
            offset = 0
            for count in counts:
                polygon = indices[offset : offset + count]
                if count > 3:
                    points = vertices[polygon]
                    normals = np.cross(points[1:-1] - points[0], points[2:] - points[0])
                    total = normals.sum(axis=0)
                    if np.linalg.norm(total) <= 1e-12 or (normals @ total < -1e-12).any():
                        raise DomainValidationError("USD polygons must be convex and nondegenerate")
                    if np.max(np.abs((points - points[0]) @ total)) > np.linalg.norm(total) * 1e-6:
                        raise DomainValidationError("USD polygons must be planar")
                triangles.extend(
                    (int(polygon[0]), int(polygon[i]), int(polygon[i + 1]))
                    for i in range(1, count - 1)
                )
                corners.extend((offset, offset + i, offset + i + 1) for i in range(1, count - 1))
                offset += count
            if str(mesh.GetOrientationAttr().Get()) == "leftHanded":
                triangles = [(face[2], face[1], face[0]) for face in triangles]
                corners = [(face[2], face[1], face[0]) for face in corners]
            visual, conversion = usd_visual_mesh(prim, vertices, np.asarray(triangles), corners)
            meshes[str(owner.GetPath())].append(visual)
            if conversion is not None:
                material_conversions.append({"mesh": str(prim.GetPath()), **conversion})
        links = []
        all_vertices = []
        for key, pieces in sorted(meshes.items()):
            if not pieces:
                raise DomainValidationError(f"USD body has no visual geometry: {key}")
            combined = trimesh.Scene(pieces)
            ref = artifacts.put_bytes(
                combined.export(file_type="glb"), media_type="model/gltf-binary"
            )
            all_vertices.append(np.concatenate([mesh.vertices for mesh in pieces]))
            links.append(
                {
                    "name": bodies[key].GetName(),
                    "usd_path": key,
                    "geometry": ref.model_dump(mode="json"),
                    "vertex_count": sum(len(mesh.vertices) for mesh in pieces),
                    "triangle_count": sum(len(mesh.faces) for mesh in pieces),
                }
            )
        joints = []
        children: set[str] = set()
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents, targets = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if prim.IsA(UsdPhysics.FixedJoint) and not parents:
                continue
            if (
                len(parents) != 1
                or len(targets) != 1
                or str(parents[0]) not in bodies
                or str(targets[0]) not in bodies
            ):
                raise DomainValidationError("USD joint must connect exactly two imported bodies")
            parent, child = str(parents[0]), str(targets[0])
            if child in children or parent == child:
                raise DomainValidationError("USD articulation must be a tree")
            children.add(child)
            if prim.IsA(UsdPhysics.PrismaticJoint):
                typed, kind, factor = UsdPhysics.PrismaticJoint(prim), "slide", scale
            elif prim.IsA(UsdPhysics.RevoluteJoint):
                typed, kind, factor = UsdPhysics.RevoluteJoint(prim), "hinge", math.pi / 180
            else:
                raise DomainValidationError("reference import supports slide and hinge joints")
            lower, upper = (
                float(typed.GetLowerLimitAttr().Get()) * factor,
                float(typed.GetUpperLimitAttr().Get()) * factor,
            )
            if not math.isfinite(lower + upper) or not lower < upper:
                raise DomainValidationError("reference joint requires finite increasing limits")
            parent_matrix = np.asarray(
                UsdGeom.Xformable(bodies[parent]).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ),
                dtype=float,
            ).T
            rotation = joint.GetLocalRot0Attr().Get()
            from pxr import Gf

            joint_rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(rotation)), dtype=float).T
            axis = (
                basis
                @ parent_matrix[:3, :3]
                @ joint_rotation
                @ np.eye(3)["XYZ".index(str(typed.GetAxisAttr().Get()))]
            )
            axis /= np.linalg.norm(axis)
            position = np.asarray(joint.GetLocalPos0Attr().Get(), dtype=float)
            origin = basis @ (parent_matrix[:3, :3] @ position + parent_matrix[:3, 3]) * scale
            joints.append(
                {
                    "name": bodies[child].GetName() + ".joint",
                    "parent": bodies[parent].GetName(),
                    "child": bodies[child].GetName(),
                    "joint_type": kind,
                    "axis": axis.tolist(),
                    "origin_m": origin.tolist(),
                    "lower_limit": lower,
                    "upper_limit": upper,
                    "limit_unit": "m" if kind == "slide" else "rad",
                    "usd_path": str(prim.GetPath()),
                }
            )
        roots = set(bodies) - children
        if len(roots) != 1 or len(joints) != len(bodies) - 1:
            raise DomainValidationError("reference articulation must have one connected root")
        # Check connectivity rather than relying on the edge count (cycles can coexist with a root).
        reached = {bodies[next(iter(roots))].GetName()}
        for _ in bodies:
            reached.update(j["child"] for j in joints if j["parent"] in reached)
        if reached != {p.GetName() for p in bodies.values()}:
            raise DomainValidationError(
                "reference articulation contains disconnected bodies or cycles"
            )
        vertices = np.concatenate(all_vertices)
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        dimensions = bounds[1] - bounds[0]
        conflicts = []
        info_ref = attachments.get("sim/info.json")
        if info_ref:
            from physrealx.evidence import strict_json_object

            info = strict_json_object(artifacts.get_bytes(info_ref), "ArtModeling metadata")
            numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(info.get("dimension(meter)", "")))
            if len(numbers) == 3 and not np.allclose(
                sorted(map(float, numbers)), sorted(dimensions), rtol=0.05
            ):
                conflicts.append(
                    {
                        "field": "dimensions",
                        "metadata_m": list(map(float, numbers)),
                        "usd_m": dimensions.tolist(),
                        "resolution": "explicitly_selected_usd",
                    }
                )
        kinematics_ref = attachments.get("sim/kinematics.json")
        if kinematics_ref:
            import json

            metadata = json.loads(artifacts.get_bytes(kinematics_ref))
            if isinstance(metadata, list):
                names = sorted(
                    item["name"]
                    for item in metadata
                    if isinstance(item, dict) and isinstance(item.get("name"), str)
                )
                usd_names = sorted(link["name"] for link in links)
                if names != usd_names:
                    conflicts.append(
                        {
                            "field": "part_names",
                            "metadata_names": names,
                            "usd_names": usd_names,
                            "resolution": "explicitly_selected_usd",
                        }
                    )
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(attachments):
                if name.startswith(str(path.parent) + "/"):
                    archive.writestr(name, (root / name).read_bytes())
        bundle = artifacts.put_bytes(archive_buffer.getvalue(), media_type="application/zip")
    return artifacts.put_json(
        {
            "schema": "physrealx.usd-reference/1",
            "status": "IMPORTED",
            "evidence_scope": "supplied_reverse_engineered_model",
            "independent_physical_validation": False,
            "source_observations": observations.reference().model_dump(mode="json"),
            "source_usd": attachments[usd_path].model_dump(mode="json"),
            "usd_entry": usd_path,
            "bundle": bundle.model_dump(mode="json"),
            "unit": "m",
            "coordinate_frame": "reference_z_up",
            "source_meters_per_unit": scale,
            "source_up_axis": up,
            "bounds_m": bounds.tolist(),
            "dimensions_m": dimensions.tolist(),
            "root_link": bodies[next(iter(roots))].GetName(),
            "links": links,
            "joints": joints,
            "metadata_conflicts": conflicts,
            "material_runtime_dependencies": sorted(set(shader_dependencies)),
            "material_conversions": material_conversions,
        }
    )


def evaluate_reference_kinematics(
    reference: ArtifactRef, artifacts: ArtifactStore, positions: dict[str, float]
) -> ArtifactRef:
    """Evaluate authored joints in the canonical reference frame, without dynamics."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from physrealx.integrations._artifacts import read_json

    model = read_json(artifacts, reference, "physrealx.usd-reference/1")
    joints = model["joints"]
    names = {joint["name"] for joint in joints}
    if set(positions) != names:
        raise DomainValidationError(
            "reference joint positions must cover exactly the authored joints"
        )
    transforms = {model["root_link"]: np.eye(4)}
    remaining = list(joints)
    while remaining:
        ready = [joint for joint in remaining if joint["parent"] in transforms]
        if not ready:
            raise DomainValidationError("reference joint graph is disconnected or cyclic")
        for joint in ready:
            q = positions[joint["name"]]
            if (
                isinstance(q, bool)
                or not isinstance(q, (int, float))
                or not math.isfinite(q)
                or not joint["lower_limit"] <= q <= joint["upper_limit"]
            ):
                raise DomainValidationError(
                    f"joint position exceeds finite authored limits: {joint['name']}"
                )
            axis = np.asarray(joint["axis"], dtype=float)
            origin = np.asarray(joint["origin_m"], dtype=float)
            if (
                axis.shape != (3,)
                or origin.shape != (3,)
                or not np.isfinite(origin).all()
                or not np.isclose(np.linalg.norm(axis), 1)
            ):
                raise DomainValidationError(
                    "reference joint requires a finite unit axis and origin"
                )
            motion = np.eye(4)
            if joint["joint_type"] == "slide":
                motion[:3, 3] = axis * q
            elif joint["joint_type"] == "hinge":
                motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                motion[:3, 3] = origin - motion[:3, :3] @ origin
            else:
                raise DomainValidationError("reference kinematics supports slide or hinge joints")
            transforms[joint["child"]] = transforms[joint["parent"]] @ motion
            remaining.remove(joint)
    return artifacts.put_json(
        {
            "schema": "physrealx.reference-kinematics/1",
            "status": "COMPLETED",
            "evidence_scope": "authored_usd_kinematics",
            "independent_motion_validation": False,
            "source_reference": reference.model_dump(mode="json"),
            "coordinate_frame": model["coordinate_frame"],
            "positions": positions,
            "link_transforms": {name: matrix.tolist() for name, matrix in transforms.items()},
        }
    )
