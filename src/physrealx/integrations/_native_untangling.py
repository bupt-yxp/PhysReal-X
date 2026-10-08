"""Shared canonical, multi-pose Mesh-Untangling input and output contracts.

The native flow restores feasibility before IPC shape recovery. Neither a native
exit code nor zero sampled intersections is a publishable asset: callers must
validate exported geometry, observations and continuous articulation separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from physrealx.reconstruction import joint_transform

UPSTREAM_REVISION = "8cc1557f23562803c347b95ec24b7c445afe1d54"


@dataclass(frozen=True)
class PoseBatch:
    """A rigidly transformed copy at each pose, with shared vertex identity."""

    names: tuple[str, str]
    vertices: NDArray[Any]
    faces: NDArray[Any]
    rotations: NDArray[Any]
    positions: NDArray[Any]
    source_vertices: NDArray[Any]
    moving_begin: int
    scale: float
    pins: NDArray[Any]

    def native_constraints(self) -> dict[str, Any]:
        return {
            "canonical_vertices": len(self.source_vertices),
            "moving_begin": self.moving_begin,
            "rotations": self.rotations.ravel().tolist(),
            "pinned_canonical_vertices": self.pins.tolist(),
        }

    def decode(self, vertices: NDArray[Any], faces: NDArray[Any]) -> NDArray[Any]:
        """Reject independent pose deformation, reordered topology and pin drift."""
        if (
            vertices.shape != self.vertices.shape
            or not np.isfinite(vertices).all()
            or not np.array_equal(faces, self.faces)
        ):
            raise ValueError("native output changed topology or has invalid coordinates")
        count = len(self.source_vertices)
        delta = ((vertices - self.vertices) / self.scale).reshape(-1, count, 3)
        for i, rotation in enumerate(self.rotations):
            delta[i, self.moving_begin :] = delta[i, self.moving_begin :] @ rotation
        mean = delta.mean(axis=0)
        if np.max(np.linalg.norm(delta - mean, axis=-1)) > 1e-8:
            raise ValueError("native poses no longer share one canonical deformation")
        if len(self.pins) and np.max(np.linalg.norm(delta[:, self.pins], axis=-1)) > 1e-9:
            raise ValueError("native flow moved protected vertices")
        return np.asarray(self.source_vertices + mean, dtype=float)


def pose_batch(
    meshes: dict[str, Any], motion: dict[str, Any], *, poses: int, hinge_band_m: float
) -> PoseBatch:
    if type(poses) is not int or not 3 <= poses <= 361:
        raise ValueError("at least three articulation poses are required")
    if not np.isfinite(hinge_band_m) or hinge_band_m <= 0:
        raise ValueError("hinge band must be positive and finite")
    names = (motion["parent"], motion["child"])
    if names[0] == names[1] or set(names) != set(meshes) or motion["kind"] != "hinge":
        raise ValueError("native unfolding requires two distinct links and one hinge")
    axis, origin = np.asarray(motion["axis"], float), np.asarray(motion["origin_m"], float)
    interval = np.asarray(motion["range"], float)
    if (
        axis.shape != (3,)
        or origin.shape != (3,)
        or interval.shape != (2,)
        or not all(np.isfinite(x).all() for x in (axis, origin, interval))
        or not np.isclose(np.linalg.norm(axis), 1, rtol=0, atol=1e-8)
        or interval[0] >= interval[1]
    ):
        raise ValueError("invalid metric hinge parameters")
    for mesh in meshes.values():
        if (
            not len(mesh.vertices)
            or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()
            or mesh.faces.ndim != 2
            or mesh.faces.shape[1] != 3
            or mesh.faces.min() < 0
            or mesh.faces.max() >= len(mesh.vertices)
        ):
            raise ValueError("native unfolding requires finite triangular surfaces")
    moving = len(meshes[names[0]].vertices)
    source = np.vstack([meshes[name].vertices for name in names]).copy()
    faces = np.vstack([meshes[names[0]].faces, meshes[names[1]].faces + moving])
    extent = float(np.ptp(source, axis=0).max())
    if extent <= 0:
        raise ValueError("source geometry has no spatial extent")
    scale = 1 / extent
    relative = source - origin
    radius = np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1)
    pins = np.flatnonzero(radius > hinge_band_m)
    positions = np.linspace(float(interval[0]), float(interval[1]), poses)
    points, rotations = [], []
    # Copies cannot interact: spacing exceeds the swept bounding diameter and
    # Gaussian cutoff. This offset is removed on decode, never a part pose edit.
    spacing = max(float(np.linalg.norm(relative, axis=1).max()) * 4, extent * 4)
    for i, q in enumerate(positions):
        transform = joint_transform("hinge", axis, origin, float(q))
        v = source.copy()
        v[moving:] = trimesh.transform_points(v[moving:], transform)
        points.append((v + np.array([i * spacing, 0, 0])) * scale)
        rotations.append(transform[:3, :3])
    return PoseBatch(
        names,
        np.vstack(points),
        np.vstack([faces + i * len(source) for i in range(poses)]),
        np.asarray(rotations),
        positions,
        source,
        moving,
        scale,
        pins,
    )


def read_obj(path: Path) -> tuple[NDArray[Any], NDArray[Any]]:
    """Read the native triangle OBJ without a loader welding/reordering vertices."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        row = line.split()
        if row and row[0] == "v":
            if len(row) != 4:
                raise ValueError("native vertex must be one XYZ triple")
            vertices.append([float(x) for x in row[1:]])
        elif row and row[0] == "f":
            if len(row) != 4:
                raise ValueError("native output contains non-triangular faces")
            faces.append([int(x.split("/")[0]) - 1 for x in row[1:]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def write_case(batch: PoseBatch, directory: Path, *, iterations: int = 1200) -> Path:
    """Create a new, isolated run; never overwrite a previous checkpoint."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / "input.obj"
    trimesh.Trimesh(batch.vertices, batch.faces, process=False).export(
        input_path, include_color=False, include_normals=False, include_texture=False, digits=17
    )
    config = {
        "filePath": [str(input_path.resolve())],
        "outputFolder": str((directory / "forward").resolve()),
        "physrealx_articulation": batch.native_constraints(),
        "gaussianEpsilon": 0.01,
        "gaussianCutoffRadius": {"relativeTo": "median_edge_length", "scale": 20},
        "solverType": "LimitedMemoryBFGS",
        "exportMeshPerIteration": False,
        "solverRunSpec": {
            "gradientTerminationNormCriterion": 1e-8,
            "lineSearchInitialAlpha": 1.0,
            "lineSearchEpsilon": 1e-10,
            "maxIterationCount": iterations,
            "memoryLength": 10,
            "beta0": 1e-4,
            "beta1": 0.9,
            "enforceArmijoCondition": True,
            "acceptOnLineSearchFailure": False,
            "enforceCurvatureCondition": False,
            "pinnedVertices": np.concatenate(
                [batch.pins + i * len(batch.source_vertices) for i in range(len(batch.positions))]
            ).tolist(),
        },
        "solverPasses": [
            {
                "name": "unfold",
                "energy": [
                    {
                        "name": "GaussianRegularizedBundle",
                        "coefficient": 1.0,
                        "energy": [
                            {"name": "GaussianLocalAdaptiveEnergy", "coefficient": 0.04},
                            {"name": "ArapEdgeEnergy", "coefficient": 0.96},
                            {"name": "MinkowskiPenaltyEnergy", "coefficient": 0.00001},
                        ],
                    }
                ],
            }
        ],
    }
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2))
    (directory / "provenance.json").write_text(
        json.dumps(
            {
                "upstream_revision": UPSTREAM_REVISION,
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "scale": batch.scale,
                "positions": batch.positions.tolist(),
                "publication_approved": False,
            },
            indent=2,
        )
    )
    return path


def run_flow(runtime: Path, config: Path, *, gpu: int, timeout_s: float = 1800) -> Path:
    """Run pinned native code; failure never falls back to the old local solver."""
    if type(gpu) is not int or gpu < 0 or not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("invalid GPU index or timeout")
    executable = (runtime / "mesh-untangle").resolve(strict=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    with (config.parent / "forward.log").open("w") as log:
        subprocess.run(
            [str(executable), "--headless", str(config.resolve())],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=True,
        )
    return Path(json.loads(config.read_text())["outputFolder"])


def run_recovery(
    runtime: Path, config: Path, *, gpu: int, iterations: int = 100, timeout_s: float = 1800
) -> Path:
    """IPC recovery retains the same canonical projection and protected vertices."""
    if type(gpu) is not int or gpu < 0 or type(iterations) is not int or iterations < 1:
        raise ValueError("invalid GPU index or recovery iteration count")
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be positive and finite")
    document = json.loads(config.read_text())
    forward = Path(document["outputFolder"])
    if (forward / "reverse_flow").exists():
        raise FileExistsError("recovery checkpoint already exists; use an isolated run")
    original, faces = read_obj(Path(document["filePath"][0]))
    reference, reference_faces = read_obj(forward / "reference.obj")
    if not np.array_equal(faces, reference_faces) or not np.allclose(
        original, reference, rtol=0, atol=1e-12
    ):
        raise ValueError("native reference no longer matches the immutable source input")
    from ._mesh_validity import self_intersections, surface_topology

    vertices, candidate_faces = read_obj(forward / "deformed.obj")
    if not np.array_equal(faces, candidate_faces) or not np.isfinite(vertices).all():
        raise ValueError("native unfolding produced invalid recovery input")
    initial = trimesh.Trimesh(vertices, candidate_faces, process=False)
    if not surface_topology(initial)["valid"] or self_intersections(initial):
        raise ValueError("IPC recovery requires valid, nonintersecting initial surfaces")
    (forward / "physrealx.json").write_text(
        json.dumps({"physrealx_articulation": document["physrealx_articulation"]})
    )
    executable = (runtime / "reverse-flow").resolve(strict=True)
    with (config.parent / "recovery.log").open("w") as log:
        subprocess.run(
            [
                str(executable),
                str(forward.resolve()),
                "--no-tui",
                "--verbose",
                "--newton-iters",
                str(iterations),
                "--cg-iters",
                "300",
                "--rho-idp",
                "10",
                "--ipc-kappa",
                "100000",
                "--ipc-dhat-fraction",
                "0.15",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=True,
        )
    return forward / "reverse_flow" / "final.obj"


def decode_candidate(batch: PoseBatch, source: dict[str, Any], output: Path) -> dict[str, Any]:
    vertices, faces = read_obj(output)
    canonical = batch.decode(vertices, faces)
    result = {}
    for name, section in zip(
        batch.names, (slice(None, batch.moving_begin), slice(batch.moving_begin, None)), strict=True
    ):
        mesh = source[name].copy()
        mesh.vertices = canonical[section]
        result[name] = mesh
    return result


def load_unfolded_candidate(
    store: Any,
    candidate: Any,
    source_ref: Any,
    meshes: dict[str, Any],
    motion: dict[str, Any],
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Recompute validity before admitting an external solver result to the gates."""
    from physrealx.domain import ArtifactRef

    from ._artifacts import read_json
    from ._mesh_validity import self_intersections, surface_topology
    from .collision import load_meshes

    document = read_json(store, candidate, "physrealx.articulated-unfolding/1")
    if (
        ArtifactRef.model_validate(document["source_geometry"]) != source_ref
        or document["motion"] != motion
        or set(document["geometries"]) != set(meshes)
    ):
        raise ValueError("unfolding candidate differs from source geometry or articulation")
    output, links = {}, {}
    valid = True
    for name, reference in document["geometries"].items():
        values = load_meshes(store, ArtifactRef.model_validate(reference), process=False)
        if len(values) != 1:
            raise ValueError("unfolding requires exactly one surface per source link")
        mesh, source = values[0], meshes[name]
        if mesh.vertices.shape != source.vertices.shape or not np.array_equal(
            mesh.faces, source.faces
        ):
            raise ValueError("unfolding candidate changed source topology")
        displacement = np.linalg.norm(mesh.vertices - source.vertices, axis=1)
        relative = source.vertices - np.asarray(motion["origin_m"])
        axis = np.asarray(motion["axis"])
        radius = np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1)
        protected = displacement[radius > config.hinge_band_m]
        protected_drift = float(protected.max()) if len(protected) else 0.0
        collapsed = int(np.count_nonzero(mesh.area_faces <= 0.1 * source.area_faces))
        topology = surface_topology(mesh)
        intersects = self_intersections(mesh)
        maximum = float(displacement.max())
        valid &= (
            topology["valid"]
            and not intersects
            and collapsed == 0
            and maximum <= config.max_surface_displacement_m + 1e-9
            and protected_drift <= 1e-9
        )
        output[name] = mesh
        links[name] = {
            "max_displacement_m": maximum,
            "protected_vertex_displacement_m": protected_drift,
            "collapsed_faces": collapsed,
            "self_intersections": intersects,
            "topology": topology,
            "geometry_changed": maximum > 1e-9,
        }
    return (
        output,
        motion.copy(),
        {
            "method": "native_shared_canonical_untangling",
            "solver_identity": f"mesh-untangling/{UPSTREAM_REVISION}",
            "geometry_valid": bool(valid),
            "links": links,
            "source_artifacts_mutated": False,
            "native_provenance": document.get("provenance", {}),
            "heldout_used_for_optimization": False,
            "motion_terms": ["shared_canonical_vertices_fixed_source_hinge"],
            "image_terms": [],
        },
    )
