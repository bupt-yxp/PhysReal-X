"""CoACD collision decomposition worker."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from physrealx.external_workers import WorkerProducedFile, WorkerRequest, WorkerResponse, run_worker
from physrealx_workers.common import config, input_file, input_path, is_probe, report_probe


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    import coacd
    import numpy as np
    import trimesh

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
        input_path(request, "asset"), file_type=file_type, force="mesh", process=False
    )
    if not isinstance(source, trimesh.Trimesh) or len(source.faces) == 0:
        raise ValueError("CoACD requires one non-empty triangle mesh")
    options = config(request)
    source.remove_unreferenced_vertices()
    mesh = coacd.Mesh(
        np.asarray(source.vertices, dtype=np.float64),
        np.asarray(source.faces, dtype=np.int32),
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
    for index, (vertices, faces) in enumerate(parts):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        scene.add_geometry(part, node_name=f"convex_part_{index:04d}")
        output_faces += len(part.faces)
        output_vertices += len(part.vertices)
        output_volume += abs(float(part.volume))
    destination = output_directory / "collision.glb"
    destination.write_bytes(scene.export(file_type="glb"))
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
    return WorkerResponse(
        outputs=(
            WorkerProducedFile(
                name="collision_geometry",
                contract="CollisionGeometry",
                path=destination.name,
                media_type="model/gltf-binary",
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
        return report_probe(
            "coacd",
            {
                "coacd": importlib.metadata.version("coacd"),
                "numpy": numpy.__version__,
                "trimesh": trimesh.__version__,
            },
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
