"""Run and visualize the real external-worker chain on ArtModeling cabinet2."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from physrealx import PhysRealX
from physrealx.domain import ArtifactRef, OperationManifest
from physrealx.errors import PhysRealXError
from physrealx.reference import reference_asset, reference_scene, reference_task


def _output(manifest: OperationManifest, name: str) -> ArtifactRef:
    return next(item.artifact for item in manifest.outputs if item.name == name)


def _materialize(platform: PhysRealX, reference: ArtifactRef, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir() or destination.is_symlink():
            raise RuntimeError(f"refusing to replace unsafe experiment output: {destination}")
        destination.unlink()
    platform.artifacts.materialize(reference, destination)
    return destination


def _plot_collision(source_path: Path, collision_path: Path, destination: Path) -> None:
    source = trimesh.load(source_path, force="mesh", process=False)
    collision = trimesh.load(collision_path, force="scene", process=False)
    source_points = source.sample(min(20_000, max(2_000, len(source.vertices))))
    geometries = list(collision.geometry.values())
    count = max(100, 20_000 // max(1, len(geometries)))
    collision_points = np.concatenate([geometry.sample(count) for geometry in geometries])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, dimensions, title in (
        (axes[0], (0, 1), "XY projection"),
        (axes[1], (0, 2), "XZ projection"),
    ):
        axis.scatter(
            source_points[:, dimensions[0]],
            source_points[:, dimensions[1]],
            s=0.3,
            alpha=0.18,
            label="visual mesh",
        )
        axis.scatter(
            collision_points[:, dimensions[0]],
            collision_points[:, dimensions[1]],
            s=0.5,
            alpha=0.35,
            label="CoACD hulls",
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    axes[0].legend(markerscale=8)
    figure.suptitle("ArtModeling cabinet2: visual geometry vs collision decomposition")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_trajectory(trajectory_path: Path, destination: Path) -> None:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    samples = trajectory["samples"]
    time_values = np.asarray([item["time"] for item in samples])
    position = np.asarray([item["position"] for item in samples])
    control = np.asarray([item["control"] for item in samples])
    target = np.asarray([item["target"] for item in samples])
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time_values, position, label="joint position", linewidth=2.2)
    axes[0].plot(time_values, target, "--", label="target", linewidth=1.5)
    axes[0].set_ylabel("position (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(time_values, control, color="#f59e0b", linewidth=1.8)
    axes[1].set_xlabel("simulation time (s)")
    axes[1].set_ylabel("control force (N)")
    axes[1].grid(alpha=0.2)
    figure.suptitle("MuJoCo prismatic-joint validation")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_render(streams_path: Path, destination: Path) -> None:
    streams = np.load(streams_path)
    rgb = streams["rgb"][0]
    if rgb.dtype.kind == "f":
        rgb = np.clip(rgb, 0.0, 1.0)
    depth = streams["depth"][0]
    depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0.0) | (depth >= 1e9), depth)
    segmentation = streams["segmentation"][0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    depth_image = axes[1].imshow(depth, cmap="viridis")
    axes[1].set_title("Depth")
    figure.colorbar(depth_image, ax=axes[1], fraction=0.046)
    axes[2].imshow(segmentation, cmap="tab20")
    axes[2].set_title("Segmentation")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("BlenderProc synchronized modalities")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_summary(summary: dict[str, object], destination: Path) -> None:
    stages = summary["stages"]
    names = list(stages)
    durations = [stages[name]["duration_seconds"] for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(names, durations, color=["#2563eb", "#0891b2", "#7c3aed", "#0f766e"])
    axes[0].set_ylabel("wall time (s)")
    axes[0].set_title("Isolated worker execution time")
    axes[0].tick_params(axis="x", rotation=20)
    metrics = {
        "CoACD hulls": stages["coacd"]["metrics"]["convex_parts"],
        "MuJoCo error x1e4": stages["mujoco"]["metrics"]["final_joint_error"] * 10_000,
        "Render labels": stages["blenderproc"]["metrics"]["segmentation_labels"],
        "Episode frames /100": stages["lerobot"]["metrics"]["frames"] / 100,
    }
    axes[1].bar(list(metrics), list(metrics.values()), color="#1d4ed8")
    axes[1].set_title("Key verification metrics (scaled where labelled)")
    axes[1].tick_params(axis="x", rotation=25)
    figure.suptitle("PhysReal-X external worker pipeline")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs-images", type=Path)
    arguments = parser.parse_args()
    source = arguments.source.expanduser().resolve(strict=True)
    workspace = arguments.workspace.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    platform = PhysRealX.open(workspace)
    readiness = {item.descriptor.name: item.to_dict() for item in platform.methods.installations()}
    required = ("coacd", "mujoco", "blenderproc", "lerobot_v2_1")
    unavailable = [
        name for name in required if name not in readiness or not readiness[name]["ready"]
    ]
    if unavailable:
        raise RuntimeError(f"required workers are unavailable: {', '.join(unavailable)}")

    visual_mesh = platform.artifacts.put_file(source, media_type="model/ply")
    started = time.perf_counter()
    collision = platform.assets.build_collision(
        visual_mesh,
        method="coacd",
        config={"threshold": 0.08, "max_convex_hull": 24, "resolution": 2_000},
    )
    collision_wall = time.perf_counter() - started

    asset = reference_asset().model_copy(update={"resource_id": "cabinet2.external_validation"})
    asset_reference = platform.assets.register(asset)
    scene_template = reference_scene(asset)
    scene = scene_template.model_copy(
        update={
            "resource_id": "cabinet2.external_validation_scene",
            "entities": (scene_template.entities[0].model_copy(update={"asset": asset_reference}),),
        }
    )
    scene_reference = platform.scenes.register(scene)
    task = reference_task(scene).model_copy(
        update={
            "resource_id": "cabinet2.external_validation_task",
            "scene": scene_reference,
        }
    )
    task_reference = platform.tasks.register(task)
    started = time.perf_counter()
    simulation = platform.simulations.run(
        scene_reference,
        task_reference,
        method="mujoco",
        config={"joint_type": "slide", "lower": 0.0, "upper": 0.4, "target": 0.32},
    )
    simulation_wall = time.perf_counter() - started
    trajectory_reference = _output(simulation, "state_trajectory")

    started = time.perf_counter()
    render = platform.renders.generate(trajectory_reference, method="blenderproc")
    render_wall = time.perf_counter() - started
    started = time.perf_counter()
    episode = platform.episodes.pack(trajectory_reference, method="lerobot_v2_1")
    episode_wall = time.perf_counter() - started

    artifacts = output / "artifacts"
    collision_path = _materialize(
        platform, _output(collision, "collision_geometry"), artifacts / "collision.glb"
    )
    trajectory_path = _materialize(
        platform, trajectory_reference, artifacts / "state_trajectory.json"
    )
    _materialize(
        platform, _output(simulation, "simulation_report"), artifacts / "simulation_report.json"
    )
    streams_path = _materialize(
        platform, _output(render, "render_streams"), artifacts / "render_streams.npz"
    )
    _materialize(platform, _output(render, "camera_metadata"), artifacts / "camera_metadata.json")
    _materialize(platform, _output(episode, "episode_bundle"), artifacts / "episode_bundle.zip")
    _materialize(
        platform, _output(episode, "episode_manifest"), artifacts / "episode_manifest.json"
    )
    summary: dict[str, object] = {
        "schema": "physrealx.external-worker-experiment/1",
        "source": str(source),
        "source_artifact": visual_mesh.model_dump(mode="json"),
        "workers": {name: readiness[name] for name in required},
        "stages": {
            "coacd": {"duration_seconds": collision_wall, "metrics": collision.metrics},
            "mujoco": {"duration_seconds": simulation_wall, "metrics": simulation.metrics},
            "blenderproc": {"duration_seconds": render_wall, "metrics": render.metrics},
            "lerobot": {"duration_seconds": episode_wall, "metrics": episode.metrics},
        },
        "operation_references": {
            "coacd": collision.reference().model_dump(mode="json"),
            "mujoco": simulation.reference().model_dump(mode="json"),
            "blenderproc": render.reference().model_dump(mode="json"),
            "lerobot": episode.reference().model_dump(mode="json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_collision(source, collision_path, figures / "coacd-collision.png")
    _plot_trajectory(trajectory_path, figures / "mujoco-trajectory.png")
    _plot_render(streams_path, figures / "blenderproc-modalities.png")
    _plot_summary(summary, figures / "external-workers-summary.png")
    if arguments.docs_images:
        docs_images = arguments.docs_images.expanduser().resolve()
        docs_images.mkdir(parents=True, exist_ok=True)
        for figure in figures.glob("*.png"):
            shutil.copy2(figure, docs_images / figure.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PhysRealXError as error:
        print(
            json.dumps(
                {"code": error.code, "message": error.message, "details": error.details},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise
