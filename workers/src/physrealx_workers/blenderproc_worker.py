# ruff: noqa: E402
import blenderproc as bproc

"""BlenderProc RGB/depth/segmentation rendering worker."""

import json
import sys
from pathlib import Path

from physrealx.external_workers import WorkerProducedFile, WorkerRequest, WorkerResponse, run_worker
from physrealx_workers.common import input_path, is_probe, report_probe


def _protocol_arguments() -> list[str] | None:
    """Discard BlenderProc's script/temp-dir prefix before protocol parsing."""

    try:
        start = sys.argv.index("--request")
    except ValueError:
        return None
    return sys.argv[start:]


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    import numpy as np

    trajectory = json.loads(input_path(request, "state_trajectory").read_text(encoding="utf-8"))
    samples = trajectory.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("BlenderProc requires a StateTrajectory with non-empty samples")
    bproc.init()
    floor = bproc.object.create_primitive("PLANE", scale=[2.0, 2.0, 1.0])
    floor.set_cp("category_id", 1)
    body = bproc.object.create_primitive("CUBE", scale=[0.22, 0.18, 0.04])
    body.set_cp("category_id", 2)
    body.set_location([float(samples[-1]["position"]), 0.0, 0.35])
    material = bproc.material.create("cabinet-blue")
    material.set_principled_shader_value("Base Color", [0.05, 0.32, 0.8, 1.0])
    body.replace_materials(material)
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_energy(700)
    light.set_location([1.2, -1.2, 2.2])
    camera_location = np.asarray([1.4, -1.6, 1.1])
    point_of_interest = np.asarray([0.15, 0.0, 0.25])
    camera_rotation = bproc.camera.rotation_from_forward_vec(point_of_interest - camera_location)
    camera_pose = bproc.math.build_transformation_mat(camera_location, camera_rotation)
    bproc.camera.add_camera_pose(camera_pose)
    bproc.camera.set_resolution(512, 384)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])
    rendered = bproc.renderer.render()
    colors = np.asarray(rendered["colors"])
    depth = np.asarray(rendered["depth"])
    segmentation_key = next(
        key for key in rendered if key in {"category_id_segmaps", "instance_segmaps"}
    )
    segmentation = np.asarray(rendered[segmentation_key])
    streams = output_directory / "render_streams.npz"
    np.savez_compressed(streams, rgb=colors, depth=depth, segmentation=segmentation)
    metadata = output_directory / "camera_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "physrealx.camera-metadata/1",
                "renderer": f"blenderproc-{bproc.__version__}",
                "resolution": [512, 384],
                "frames": int(colors.shape[0]),
                "modalities": ["rgb", "depth", "segmentation"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    valid_depth_mask = np.isfinite(depth) & (depth > 0.0) & (depth < 1e9)
    valid_depth = depth[valid_depth_mask]
    if valid_depth.size == 0:
        raise RuntimeError("BlenderProc did not produce any valid depth pixels")
    produced = {
        "render_streams": WorkerProducedFile(
            name="render_streams",
            contract="RenderStreams",
            path=streams.name,
            media_type="application/x-npz",
        ),
        "camera_metadata": WorkerProducedFile(
            name="camera_metadata",
            contract="CameraMetadata",
            path=metadata.name,
            media_type="application/json",
        ),
    }
    return WorkerResponse(
        outputs=tuple(produced[item.name] for item in request.operation.outputs),
        metrics={
            "frames": float(colors.shape[0]),
            "width": float(colors.shape[2]),
            "height": float(colors.shape[1]),
            "depth_min": float(valid_depth.min()),
            "depth_max": float(valid_depth.max()),
            "valid_depth_ratio": float(valid_depth_mask.mean()),
            "segmentation_labels": float(len(np.unique(segmentation))),
        },
    )


def main() -> int:
    if is_probe():
        import numpy

        if not numpy.__version__.startswith("1."):
            raise RuntimeError("BlenderProc's embedded runtime requires NumPy 1.x")
        return report_probe(
            "blenderproc",
            {
                "blenderproc": bproc.__version__,
                "numpy": numpy.__version__,
                "python": sys.version.split()[0],
            },
        )
    return run_worker(_handle, argv=_protocol_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
