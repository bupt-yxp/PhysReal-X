"""Measured-unit convex collision proxies using the official CoACD binding."""

from __future__ import annotations

import io
import math
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import ArtifactRef, DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import dependency_probe, input_artifact, publish
from physrealx.methods import builtin_method


def mesh_reference(uri: str, store: ArtifactStore) -> Any:
    from physrealx.domain import ArtifactRef

    if not uri.startswith("artifact://"):
        raise DomainValidationError("simulation meshes require immutable artifact:// URIs")
    identifier = uri[len("artifact://") :]
    if not store.verify(identifier):
        raise DomainValidationError("simulation mesh artifact is unavailable")
    payload = store.get_bytes(identifier)
    # Assembly publishes GLB collision and model GLB/PLY visual geometry.
    media = (
        "model/gltf-binary"
        if payload.startswith(b"glTF")
        else "model/ply"
        if payload.startswith(b"ply")
        else "model/obj"
    )
    return ArtifactRef(artifact_id=identifier, size_bytes=len(payload), media_type=media)


def load_meshes(store: ArtifactStore, reference: ArtifactRef) -> list[Any]:
    from importlib import import_module

    import numpy as np

    trimesh = import_module("trimesh")

    formats = {"model/gltf-binary": "glb", "model/ply": "ply", "model/obj": "obj"}
    if reference.media_type not in formats or reference.size_bytes > 512 * 1024 * 1024:
        raise DomainValidationError("mesh must be a self-contained GLB, PLY or OBJ under 512 MiB")
    try:
        scene = trimesh.load(
            io.BytesIO(store.get_bytes(reference)),
            file_type=formats[reference.media_type],
            force="scene",
            resolver={},
        )
        meshes = list(scene.dump())
    except (ValueError, OSError, KeyError) as error:
        raise DomainValidationError("mesh artifact cannot be decoded") from error
    if not meshes:
        raise DomainValidationError("mesh artifact is empty")
    for mesh in meshes:
        if (
            not isinstance(mesh, trimesh.Trimesh)
            or len(mesh.vertices) < 4
            or len(mesh.faces) < 4
            or not np.isfinite(mesh.vertices).all()
        ):
            raise DomainValidationError("mesh must contain finite triangulated geometry")
    return meshes


class CollisionConfig(DomainModel):
    source_unit: Literal["m"]
    link: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    threshold_m: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    max_parts: int = Field(default=64, ge=1, le=1024)
    max_volume_error: float = Field(default=0.15, gt=0, le=1)
    seed: int = Field(default=1, ge=1, le=2_147_483_647)
    weld_tolerance_m: float = Field(default=0.0, ge=0, le=0.001, allow_inf_nan=False)
    preprocessing: Literal["strict", "coacd"] = "strict"


class CollisionAdapter:
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "coacd",
        input_contract="VisualMesh/1",
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
        python=">=3.10,<3.13",
        version="1.2.0",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "numpy": ("numpy", "1.26.4"),
                "trimesh": ("trimesh", "4.5.1"),
                "coacd": ("coacd", "1.0.9"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        from importlib import import_module

        import coacd
        import numpy as np

        trimesh = import_module("trimesh")

        config = CollisionConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "asset", artifacts)
        meshes = load_meshes(artifacts, source)
        # UV and normal seams duplicate vertices without creating a physical gap.
        # Collision topology ignores appearance and merges only exactly equal XYZ.
        topology = []
        for mesh in meshes:
            vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
            topology.append(
                trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
            )
        meshes = topology
        welded_vertices = 0
        if config.weld_tolerance_m:
            # Explicitly bounded seam welding; never fill missing surfaces or cabinet openings.
            digits = math.ceil(-math.log10(config.weld_tolerance_m / math.sqrt(3)))
            for mesh in meshes:
                original_count = len(mesh.vertices)
                mesh.merge_vertices(digits_vertex=digits)
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.remove_unreferenced_vertices()
                welded_vertices += original_count - len(mesh.vertices)
        source_closed = all(mesh.is_volume for mesh in meshes)
        if not source_closed and config.preprocessing == "strict":
            raise DomainValidationError(
                "collision input must be watertight, consistently wound positive volumes"
            )
        parts = []
        for mesh in (part for original in meshes for part in original.split(only_watertight=False)):
            hull = mesh.convex_hull
            convex_volume_agrees = mesh.volume > 0 and math.isclose(
                float(hull.volume), float(mesh.volume), rel_tol=1e-6
            )
            if mesh.is_convex or convex_volume_agrees:
                parts.append(hull)
            else:
                decomposition = coacd.run_coacd(
                    coacd.Mesh(mesh.vertices, mesh.faces),
                    threshold=config.threshold_m,
                    real_metric=True,
                    max_convex_hull=config.max_parts,
                    preprocess_mode="off" if mesh.is_volume else "on",
                    seed=config.seed,
                )
                # Rebuild each convex hull from its vertices. Native triangulation can
                # contain coplanar degeneracies after float conversion; hulls are the
                # actual collision representation, not the tessellation of that hull.
                parts.extend(
                    trimesh.Trimesh(vertices=v, faces=f, process=True).convex_hull
                    for v, f in decomposition
                )
        invalid = sum(not part.is_volume or not part.is_convex for part in parts)
        if not parts or len(parts) > config.max_parts or invalid:
            raise MethodExecutionError(
                f"CoACD collision gates failed: {len(parts)} parts (limit {config.max_parts}), "
                f"{invalid} invalid convex volumes"
            )
        volume = sum(float(mesh.volume) for mesh in meshes)
        volume_error = abs(sum(float(part.volume) for part in parts) - volume) / volume
        if volume <= 0 or not np.isfinite(volume_error):
            raise MethodExecutionError("collision source has no positive oriented volume estimate")
        if volume_error > config.max_volume_error:
            raise MethodExecutionError("collision proxy volume error exceeds tolerance")
        scene = trimesh.Scene()
        for index, part in enumerate(parts):
            scene.add_geometry(
                part, node_name=f"{config.link}_{index}", geom_name=f"convex_{index}"
            )
        collision = artifacts.put_bytes(
            scene.export(file_type="glb"), media_type="model/gltf-binary"
        )
        # Reopen the actual exported bytes, rather than reporting only in-memory quality.
        if len(load_meshes(artifacts, collision)) != len(parts):
            raise MethodExecutionError("collision GLB roundtrip changed the convex part count")
        metrics = {
            "convex_parts": float(len(parts)),
            "relative_volume_error": volume_error,
            "welded_vertices": float(welded_vertices),
        }
        report = {
            "schema": "physrealx.collision-build-report/1",
            "status": "PASSED",
            "unit": "m",
            "link": config.link,
            "source_geometry": source.model_dump(mode="json"),
            "collision_geometry": collision.model_dump(mode="json"),
            "method": self.descriptor.model_dump(mode="json"),
            "preprocessing": {
                "mode": config.preprocessing,
                "weld_tolerance_m": config.weld_tolerance_m,
                "source_closed_positive_volume": source_closed,
                "volume_comparison_scope": "closed_surface"
                if source_closed
                else "oriented_surface_estimate",
            },
            "metrics": metrics,
            "gates": {
                "convex": True,
                "closed_positive_volume": True,
                "volume_error": True,
                "roundtrip": True,
            },
        }
        return publish(
            spec,
            artifacts,
            {"collision_build_report": report},
            artifacts={"collision_geometry": collision},
            metrics=metrics,
        )
