"""CoACD collision decomposition worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

from physrealx.external_workers import WorkerProducedFile, WorkerRequest, WorkerResponse, run_worker
from physrealx_workers.common import (
    config,
    declared_input_path,
    input_file,
    is_probe,
    report_probe,
)

_OUTPUTS = {
    "collision_geometry": "CollisionGeometry/1",
    "collision_build_report": "CollisionBuildReport/1",
}
_ALLOWED_OPTIONS = {
    "link",
    "max_convex_hull",
    "mcts_iterations",
    "mcts_max_depth",
    "mcts_nodes",
    "preprocess_mode",
    "preprocess_resolution",
    "resolution",
    "seed",
    "source_unit",
    "threshold",
}


def _artifact(*, digest: str, size_bytes: int, media_type: str) -> dict[str, object]:
    return {
        "digest": digest,
        "size_bytes": size_bytes,
        "media_type": media_type,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_request(request: WorkerRequest) -> dict[str, Any]:
    if request.operation.method.capability.value != "collision.build":
        raise ValueError("CoACD worker requires collision.build capability")
    actual_outputs = {item.name: item.contract for item in request.operation.outputs}
    if actual_outputs != _OUTPUTS:
        raise ValueError("CoACD worker requires its exact versioned output contracts")
    options = config(request)
    unknown = sorted(set(options) - _ALLOWED_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported CoACD options: {', '.join(unknown)}")
    if options.get("source_unit") != "m":
        raise ValueError("CoACD source_unit must explicitly be 'm'")
    link = options.get("link")
    if not isinstance(link, str) or not link.strip():
        raise ValueError("CoACD link must be a non-empty string")
    options["link"] = link.strip()
    return options


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    import coacd
    import numpy as np
    import trimesh

    options = _validate_request(request)
    source_input = input_file(request, "asset")
    file_types = {
        "model/ply": "ply",
        "model/stl": "stl",
        "model/obj": "obj",
        "model/gltf-binary": "glb",
    }
    file_type = file_types.get(source_input.media_type)
    if file_type is None:
        raise ValueError(f"CoACD does not support media type {source_input.media_type!r}")
    source = trimesh.load(
        declared_input_path(request, "asset"),
        file_type=file_type,
        force="mesh",
        process=False,
    )
    if not isinstance(source, trimesh.Trimesh) or len(source.faces) == 0:
        raise ValueError("CoACD requires one non-empty triangle mesh")
    source.remove_unreferenced_vertices()
    vertices = np.asarray(source.vertices, dtype=np.float64)
    faces = np.asarray(source.faces, dtype=np.int32)
    if not np.isfinite(vertices).all():
        raise ValueError("CoACD source mesh vertices must be finite")
    mesh = coacd.Mesh(
        vertices,
        faces,
    )
    parts = coacd.run_coacd(
        mesh,
        threshold=float(options.get("threshold", 0.05)),
        max_convex_hull=int(options.get("max_convex_hull", 32)),
        preprocess_mode=str(options.get("preprocess_mode", "auto")),
        preprocess_resolution=int(options.get("preprocess_resolution", 30)),
        resolution=int(options.get("resolution", 2_000)),
        mcts_nodes=int(options.get("mcts_nodes", 20)),
        mcts_iterations=int(options.get("mcts_iterations", 80)),
        mcts_max_depth=int(options.get("mcts_max_depth", 3)),
        seed=int(options.get("seed", 0)),
    )
    scene = trimesh.Scene()
    output_faces = 0
    output_vertices = 0
    output_volume = 0.0
    if not parts:
        raise RuntimeError("CoACD did not produce any convex parts")
    for index, (vertices, faces) in enumerate(parts):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(part.vertices) == 0 or len(part.faces) == 0:
            raise RuntimeError("CoACD produced an empty convex part")
        if not np.isfinite(np.asarray(part.vertices, dtype=np.float64)).all():
            raise RuntimeError("CoACD produced non-finite collision vertices")
        scene.add_geometry(part, node_name=f"convex_part_{index:04d}")
        output_faces += len(part.faces)
        output_vertices += len(part.vertices)
        output_volume += abs(float(part.volume))
    destination = output_directory / "collision.glb"
    destination.write_bytes(scene.export(file_type="glb"))
    if destination.stat().st_size <= 0:
        raise RuntimeError("CoACD produced an empty GLB")
    source_volume = abs(float(source.volume)) if source.is_volume else 0.0
    metrics = {
        "input_vertices": float(len(source.vertices)),
        "input_faces": float(len(source.faces)),
        "convex_parts": float(len(parts)),
        "output_vertices": float(output_vertices),
        "output_faces": float(output_faces),
        "source_watertight": float(source.is_watertight),
        "source_volume": source_volume,
        "collision_volume": output_volume,
    }
    if source_volume > 1e-12:
        metrics["volume_ratio"] = output_volume / source_volume
    if any(not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("CoACD metrics must be finite")
    collision_digest = _sha256(destination)
    descriptor = request.operation.method.descriptor
    report = {
        "schema": "physrealx.collision-build-report/1",
        "status": "PASSED",
        "method": {
            "name": descriptor.name,
            "version": descriptor.version,
            "source_revision": descriptor.source_revision,
        },
        "unit": "m",
        "link": options["link"],
        "source_geometry": _artifact(
            digest=source_input.digest,
            size_bytes=source_input.size_bytes,
            media_type=source_input.media_type,
        ),
        "collision_geometry": _artifact(
            digest=collision_digest,
            size_bytes=destination.stat().st_size,
            media_type="model/gltf-binary",
        ),
        "metrics": metrics,
        "gates": {
            "source_mesh_nonempty": True,
            "decomposition_nonempty": True,
            "output_mesh_nonempty": True,
            "finite_metrics": True,
            "metric_unit_declared": True,
        },
    }
    report_path = output_directory / "collision-build-report.json"
    report_path.write_text(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        encoding="utf-8",
    )
    return WorkerResponse(
        outputs=(
            WorkerProducedFile(
                name="collision_geometry",
                contract="CollisionGeometry/1",
                path=destination.name,
                media_type="model/gltf-binary",
            ),
            WorkerProducedFile(
                name="collision_build_report",
                contract="CollisionBuildReport/1",
                path=report_path.name,
                media_type="application/json",
            ),
        ),
        metrics=metrics,
    )


def main() -> int:
    if is_probe():
        import coacd
        import numpy
        import trimesh

        _ = coacd.Mesh
        if numpy.__version__ != "1.26.4":
            raise RuntimeError("CoACD worker requires NumPy 1.26.4")
        coacd_version = importlib.metadata.version("coacd")
        return report_probe(
            capability="collision.build",
            name="coacd",
            version=coacd_version,
            source_revision=f"v{coacd_version}",
            dependencies={
                "coacd": coacd_version,
                "numpy": numpy.__version__,
                "trimesh": trimesh.__version__,
            },
            self_test_checks={
                "coacd_mesh_api": True,
                "numpy_1_26_4": True,
                "trimesh_import": True,
            },
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
