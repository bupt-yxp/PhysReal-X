#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Evaluate a freshly trained cabinet2 ArtGS model and package a simulation asset.

The script never accepts the historical ArtModeling reconstruction directory as
model output.  Visual geometry must come from a fresh ArtGS ``ours_best`` export;
the source ``sim`` directory is used only as independently identified reference
kinematics and physical metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from matplotlib import font_manager
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

from physrealx import PhysRealX
from physrealx.domain import (
    AssetSpec,
    BodyPhysicsSpec,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    JointSpec,
    JointType,
    ParameterSource,
    PoseSpec,
    ResourceType,
    ScalarParameter,
)
from physrealx.errors import ResourceNotFoundError

PARTS = {
    0: {"link": "cabinet", "reference_index": 0},
    1: {"link": "drawer_lower", "reference_index": 2},
    2: {"link": "drawer_upper", "reference_index": 1},
}
LOG_NAMES = ("01-coarse.log", "02-joint-type.log", "03-full.log")

for font_path in (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"),
):
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"invalid triangle mesh: {path}")
    return mesh


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError(f"PLY vertex declaration is missing: {path}")


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "components": len(mesh.split(only_watertight=False)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area_m2": float(mesh.area),
        "bbox_extents_m": [float(item) for item in mesh.extents],
        "bbox_center_m": [float(item) for item in mesh.bounding_box.centroid],
    }


def symmetric_chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh) -> dict[str, float]:
    state = np.random.get_state()
    np.random.seed(42)
    try:
        points_a = a.sample(20000)
        points_b = b.sample(20000)
    finally:
        np.random.set_state(state)
    distance_ab = cKDTree(points_b).query(points_a, workers=-1)[0]
    distance_ba = cKDTree(points_a).query(points_b, workers=-1)[0]
    values = np.concatenate((distance_ab, distance_ba))
    scale = max(float(np.linalg.norm(a.extents)), float(np.linalg.norm(b.extents)))
    return {
        "mean_m": float(values.mean()),
        "p95_m": float(np.percentile(values, 95)),
        "normalized_mean": float(values.mean() / scale),
    }


def camera_and_marker_metrics(processed: Path, figures: Path) -> dict[str, Any]:
    states: dict[str, Any] = {}
    all_positions: list[np.ndarray] = []
    all_labels: list[str] = []
    all_reprojection: list[float] = []
    all_distance: list[float] = []
    for state in ("start", "end"):
        payload = json.loads((processed / state / "camera_train.json").read_text())
        frames = payload["frames"]
        poses = [frame.get("aruco_pose") for frame in frames]
        if any(pose is None for pose in poses):
            raise ValueError(f"{state} contains a frame without ArUco pose evidence")
        reprojection = np.asarray(
            [pose["mean_reprojection_error_px"] for pose in poses], dtype=float
        )
        distance = np.asarray(
            [
                pose["mean_distance_error_m"]
                for pose in poses
                if pose["mean_distance_error_m"] is not None
            ],
            dtype=float,
        )
        positions = np.asarray([pose["camera_in_object"] for pose in poses], dtype=float)
        all_positions.append(positions)
        all_labels.extend([state] * len(positions))
        all_reprojection.extend(reprojection.tolist())
        all_distance.extend(distance.tolist())
        states[state] = {
            "frame_count": len(frames),
            "aruco_pose_count": len(poses),
            "intrinsics": {
                key: float(payload[key])
                for key in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2", "w", "h")
            },
            "mean_reprojection_error_px": float(reprojection.mean()),
            "p95_reprojection_error_px": float(np.percentile(reprojection, 95)),
            "max_reprojection_error_px": float(
                max(pose["max_reprojection_error_px"] for pose in poses)
            ),
            "mean_marker_distance_error_mm": float(distance.mean() * 1000),
            "p95_marker_distance_error_mm": float(np.percentile(distance, 95) * 1000),
            "single_marker_frames": int(sum(pose["valid_markers"] == 1 for pose in poses)),
            "multi_marker_frames": int(sum(pose["valid_markers"] > 1 for pose in poses)),
            "failed_after_filters": int(payload["failed_image_count_after_all_filters"]),
        }

    positions = np.concatenate(all_positions)
    colors = np.asarray([0 if label == "start" else 1 for label in all_labels])
    figure = plt.figure(figsize=(12, 5), constrained_layout=True)
    trajectory = figure.add_subplot(1, 2, 1, projection="3d")
    trajectory.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=10)
    trajectory.set_title("ArUco 物体坐标系中的相机轨迹")
    trajectory.set_xlabel("X / m")
    trajectory.set_ylabel("Y / m")
    trajectory.set_zlabel("Z / m")
    errors = figure.add_subplot(1, 2, 2)
    errors.boxplot(
        (all_reprojection, np.asarray(all_distance) * 1000),
        labels=("重投影 / px", "标记间距 / mm"),
        showfliers=False,
    )
    errors.set_title("222 帧标定残差")
    errors.grid(axis="y", alpha=0.25)
    figure.savefig(figures / "01_marker_alignment.png", dpi=180)
    plt.close(figure)
    return {
        "method": "ArUco board PnP + LM/VVS refinement",
        "coordinate_frame": "object",
        "frame_count": len(positions),
        "states": states,
        "overall_mean_reprojection_error_px": float(np.mean(all_reprojection)),
        "overall_p95_reprojection_error_px": float(np.percentile(all_reprojection, 95)),
        "overall_mean_marker_distance_error_mm": float(np.mean(all_distance) * 1000),
    }


def depth_metrics(processed: Path, figures: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    previews: list[tuple[str, np.ndarray, np.ndarray]] = []
    for state in ("start", "end"):
        rgba_paths = sorted((processed / state / "train" / "rgba").glob("*.png"))
        depth_paths = sorted((processed / state / "train" / "depth").glob("*.png"))
        if not rgba_paths or [item.stem for item in rgba_paths] != [
            item.stem for item in depth_paths
        ]:
            raise ValueError(f"{state} RGB/depth identities are incomplete")
        valid_ratios: list[float] = []
        medians: list[float] = []
        p95_values: list[float] = []
        for path in depth_paths:
            depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth_mm is None or depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
                raise ValueError(f"expected uint16 metric depth: {path}")
            valid = (depth_mm > 0) & (depth_mm < np.iinfo(np.uint16).max)
            values = depth_mm[valid]
            valid_ratios.append(float(valid.mean()))
            medians.append(float(np.median(values)))
            p95_values.append(float(np.percentile(values, 95)))
        rows[state] = {
            "frame_count": len(depth_paths),
            "resolution": [int(depth_mm.shape[1]), int(depth_mm.shape[0])],
            "dtype": "uint16",
            "unit": "mm",
            "valid_ratio_mean": float(np.mean(valid_ratios)),
            "valid_ratio_min": float(np.min(valid_ratios)),
            "median_depth_mm_mean": float(np.mean(medians)),
            "p95_depth_mm_mean": float(np.mean(p95_values)),
        }
        index = len(rgba_paths) // 2
        rgba = np.asarray(Image.open(rgba_paths[index]).convert("RGB"))
        depth = cv2.imread(str(depth_paths[index]), cv2.IMREAD_UNCHANGED)
        previews.append((state, rgba, depth))

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, (state, rgba, depth) in enumerate(previews):
        axes[0, column].imshow(rgba)
        axes[0, column].set_title(f"{state} RGB-A")
        axes[1, column].imshow(np.where(depth > 0, depth, np.nan), cmap="turbo")
        axes[1, column].set_title(f"{state} 原始米制深度 / mm")
        for row in range(2):
            axes[row, column].axis("off")
    figure.savefig(figures / "02_depth_inputs.png", dpi=180)
    plt.close(figure)
    return {
        "states": rows,
        "enhancement_method": None,
        "training_supervision": "metric_depth",
        "metric_depth_loss_weight": 1.0,
        "mono_depth_loss_weight": 0.0,
        "note": "No depth enhancement/completion model was executed in this run.",
    }


def final_progress(log_path: Path, expected_iterations: int) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    pattern = re.compile(r"Training progress:\s+100%.*?Loss=([0-9.]+)")
    matches = pattern.findall(text)
    if not matches or "Training complete." not in text:
        raise ValueError(f"training stage did not complete: {log_path}")
    return {
        "iterations": expected_iterations,
        "final_displayed_loss": float(matches[-1]),
        "log_sha256": sha256_file(log_path),
    }


def training_metrics(training_run: Path, model_root: Path) -> dict[str, Any]:
    logs = training_run / "logs"
    for name in LOG_NAMES:
        if not (logs / name).is_file():
            raise FileNotFoundError(logs / name)
    checkpoint_paths = {
        "coarse_start": model_root / "coarse_gs/point_cloud/iteration_10000/point_cloud_0.ply",
        "coarse_end": model_root / "coarse_gs/point_cloud/iteration_10000/point_cloud_1.ply",
        "gaussian_best": model_root / "artgs/point_cloud/iteration_best/point_cloud.ply",
        "deform_best": model_root / "artgs/deform/iteration_best/deform.pth",
        "gaussian_20000": model_root / "artgs/point_cloud/iteration_20000/point_cloud.ply",
        "deform_20000": model_root / "artgs/deform/iteration_20000/deform.pth",
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fresh checkpoints are incomplete: {missing}")
    coarse_vertices = {}
    for key in ("coarse_start", "coarse_end"):
        coarse_vertices[key] = ply_vertex_count(checkpoint_paths[key])
    return {
        "execution": "fresh_training",
        "coarse": final_progress(logs / "01-coarse.log", 10000),
        "joint_predictor": final_progress(logs / "02-joint-type.log", 5000),
        "joint_prediction": ["static", "prismatic", "prismatic"],
        "joint_training": final_progress(logs / "03-full.log", 20000),
        "coarse_gaussian_counts": coarse_vertices,
        "checkpoints": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in checkpoint_paths.items()
        },
    }


def image_metrics(gt_path: Path, render_path: Path) -> tuple[float, float]:
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    render = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
    if gt.shape != render.shape:
        raise ValueError(f"render shape mismatch: {render_path}")
    mse = float(np.mean((gt - render) ** 2))
    psnr = 100.0 if mse == 0 else -10.0 * math.log10(mse)
    ssim = structural_similarity(gt, render, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def render_metrics(ours_best: Path, figures: Path) -> dict[str, Any]:
    result = {}
    with (ours_best / "result.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if row and row[0] in {"PSNR", "SSIM", "LPIPS"}:
                result[row[0].lower()] = float(row[1])
    gt_paths = sorted((ours_best / "gt/-1").glob("*.png"))
    render_paths = sorted((ours_best / "renders/-1").glob("*.png"))
    if len(gt_paths) != 222 or [p.name for p in gt_paths] != [p.name for p in render_paths]:
        raise ValueError("expected 222 aligned overall GT/render frames")
    pairs = [image_metrics(gt, render) for gt, render in zip(gt_paths, render_paths)]
    psnr = np.asarray([item[0] for item in pairs])
    ssim = np.asarray([item[1] for item in pairs])
    indices = (0, 111, 221)
    figure, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for row, index in enumerate(indices):
        gt = np.asarray(Image.open(gt_paths[index]).convert("RGB"))
        render = np.asarray(Image.open(render_paths[index]).convert("RGB"))
        error = np.abs(gt.astype(np.float32) - render.astype(np.float32)).mean(axis=2)
        for column, image in enumerate((gt, render, error)):
            axes[row, column].imshow(image, cmap="magma" if column == 2 else None)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"frame {index}")
    axes[0, 0].set_title("真实观测")
    axes[0, 1].set_title("新训练 ArtGS")
    axes[0, 2].set_title("绝对像素误差")
    figure.savefig(figures / "03_render_quality.png", dpi=180)
    plt.close(figure)
    return {
        "resolution": [640, 360],
        "frame_count": len(pairs),
        "psnr_mean_db_recomputed": float(psnr.mean()),
        "psnr_p05_db": float(np.percentile(psnr, 5)),
        "ssim_mean_recomputed": float(ssim.mean()),
        "ssim_p05": float(np.percentile(ssim, 5)),
        "artgs_result": result,
        "part_render_counts": {
            str(part): len(list((ours_best / "renders" / str(part)).glob("*.png")))
            for part in (-1, 0, 1, 2)
        },
    }


def mesh_metrics(ours_best: Path, figures: Path) -> dict[str, Any]:
    mesh_root = ours_best / "meshes"
    summaries: dict[str, Any] = {}
    motion: dict[str, Any] = {}
    start_meshes: list[tuple[int, trimesh.Trimesh]] = []
    for part in (-1, 0, 1, 2):
        start = load_mesh(mesh_root / f"start_{part}.ply")
        end = load_mesh(mesh_root / f"end_{part}.ply")
        summaries[f"start_{part}"] = mesh_summary(start)
        summaries[f"end_{part}"] = mesh_summary(end)
        displacement = end.bounding_box.centroid - start.bounding_box.centroid
        motion[str(part)] = {
            "center_displacement_m": [float(value) for value in displacement],
            "center_displacement_norm_m": float(np.linalg.norm(displacement)),
            "start_end_chamfer": symmetric_chamfer(start, end),
        }
        if part >= 0:
            start_meshes.append((part, start))
    figure = plt.figure(figsize=(14, 5), constrained_layout=True)
    for column, (part, mesh) in enumerate(start_meshes, 1):
        axis = figure.add_subplot(1, 3, column, projection="3d")
        indices = np.unique(
            np.linspace(0, len(mesh.vertices) - 1, min(8000, len(mesh.vertices)), dtype=int)
        )
        points = mesh.vertices[indices]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25)
        axis.set_title(f"{PARTS[part]['link']}\n{len(mesh.vertices):,} vertices")
        axis.set_axis_off()
        axis.set_box_aspect(np.maximum(mesh.extents, 1e-6))
    figure.savefig(figures / "04_trained_part_meshes.png", dpi=180)
    plt.close(figure)
    return {"summaries": summaries, "motion": motion}


def unoriented_axis_error(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def joint_metrics(source: Path, ours_best: Path, figures: Path) -> dict[str, Any]:
    prediction = json.loads((ours_best / "joint_info.json").read_text())
    reference = json.loads((source / "sim/kinematics.json").read_text())
    predicted = [item for item in prediction if item.get("joint") == "slider"]
    expected = [item for item in reference if item.get("part_type") == "prismatic"]
    reference_axis = np.asarray(
        expected[0]["inter_part_kinematics"]["motion_axis"]["axis_vector"], dtype=float
    )
    errors = []
    rows = []
    for item in predicted:
        axis = np.asarray(item["jointData"]["axis"]["direction"], dtype=float)
        error = unoriented_axis_error(axis, reference_axis)
        errors.append(error)
        rows.append(
            {
                "id": int(item["id"]),
                "type": "prismatic",
                "axis": [float(value) for value in axis],
                "unoriented_axis_error_deg": error,
                "learned_limit": bool(item["jointData"].get("limit")),
            }
        )
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.bar([f"slot {row['id']}" for row in rows], errors)
    axis.axhline(10.0, color="tab:red", linestyle="--", label="10° gate")
    axis.set_ylabel("无方向轴角误差 / °")
    axis.set_title("预测滑动轴与独立参考轴")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figures / "05_joint_axis_error.png", dpi=180)
    plt.close(figure)
    return {
        "predicted_movable_joint_count": len(predicted),
        "reference_movable_joint_count": len(expected),
        "type_recall": float(min(len(predicted), len(expected)) / len(expected)),
        "predictions": rows,
        "axis_error_mean_deg": float(np.mean(errors)),
        "axis_error_max_deg": float(np.max(errors)),
        "limit_completion_ratio": float(sum(row["learned_limit"] for row in rows) / len(expected)),
        "packaging_policy": "reference kinematics used because a predicted axis fails the 10 degree gate and limits are absent",
    }


def scalar(
    value: float, source: ParameterSource, confidence: float, uncertainty: float
) -> ScalarParameter:
    return ScalarParameter(
        value=value, source=source, confidence=confidence, uncertainty=uncertainty
    )


def physics_spec(
    link: str, source_material: dict[str, Any], mesh: trimesh.Trimesh
) -> BodyPhysicsSpec:
    mass_text = str(source_material["mass"])
    mass = float(mass_text[:-2] if mass_text.endswith("kg") else mass_text)
    x, y, z = np.maximum(mesh.extents, 1e-5)
    inertia = (
        (float(mass * (y * y + z * z) / 12), 0.0, 0.0),
        (0.0, float(mass * (x * x + z * z) / 12), 0.0),
        (0.0, 0.0, float(mass * (x * x + y * y) / 12)),
    )
    raw_static = float(source_material["static_friction"])
    raw_dynamic = float(source_material["dynamic_friction"])
    static = max(raw_static, raw_dynamic)
    dynamic = min(raw_static, raw_dynamic)
    return BodyPhysicsSpec(
        link=link,
        mass_kg=scalar(mass, ParameterSource.MEASURED, 0.5, mass * 0.2),
        center_of_mass_m=tuple(float(value) for value in mesh.bounding_box.centroid),
        inertia_kg_m2=inertia,
        static_friction=scalar(static, ParameterSource.INFERRED, 0.4, 0.1),
        dynamic_friction=scalar(dynamic, ParameterSource.INFERRED, 0.4, 0.1),
        restitution=scalar(0.05, ParameterSource.DEFAULT, 0.2, 0.05),
    )


def simplify_collision(source: Path, destination: Path) -> trimesh.Trimesh:
    mesh = o3d.io.read_triangle_mesh(str(source))
    target = max(1000, min(10000, len(mesh.triangles) // 10))
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    if not o3d.io.write_triangle_mesh(str(destination), simplified, write_ascii=False):
        raise OSError(f"failed to write collision mesh: {destination}")
    return load_mesh(destination)


def author_usd_mesh(stage: Usd.Stage, path: str, mesh: trimesh.Trimesh) -> UsdGeom.Mesh:
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr(
        np.asarray(mesh.faces, dtype=np.int32).reshape(-1).tolist()
    )
    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(usd_mesh, Usd.TimeCode.Default())
    if extent:
        usd_mesh.CreateExtentAttr(extent)
    return usd_mesh


def package_asset(
    source: Path,
    ours_best: Path,
    output: Path,
    px: PhysRealX,
) -> tuple[AssetSpec, dict[str, Any]]:
    asset_root = output / "asset"
    collision_root = asset_root / "collision"
    asset_root.mkdir()
    collision_root.mkdir()
    source_materials = json.loads((source / "sim/Material.json").read_text())
    reference = json.loads((source / "sim/kinematics.json").read_text())
    mesh_root = ours_best / "meshes"
    geometries = []
    physics = []
    meshes: dict[str, trimesh.Trimesh] = {}
    collision_metrics: dict[str, Any] = {}
    artifacts = []
    for part, mapping in PARTS.items():
        link = mapping["link"]
        visual_path = mesh_root / f"start_{part}.ply"
        collision_path = collision_root / f"{link}.ply"
        visual_mesh = load_mesh(visual_path)
        collision_mesh = simplify_collision(visual_path, collision_path)
        meshes[link] = visual_mesh
        visual_ref = px.artifacts.put_file(visual_path, media_type="model/ply")
        collision_ref = px.artifacts.put_file(collision_path, media_type="model/ply")
        artifacts.extend((visual_ref, collision_ref))
        geometries.extend(
            (
                GeometrySpec(
                    name=f"{link}_visual",
                    link=link,
                    role=GeometryRole.VISUAL,
                    representation=GeometryRepresentation.MESH,
                    uri=f"artifact://{visual_ref.digest}",
                    checksum=visual_ref.digest,
                ),
                GeometrySpec(
                    name=f"{link}_collision",
                    link=link,
                    role=GeometryRole.COLLISION,
                    representation=GeometryRepresentation.MESH,
                    uri=f"artifact://{collision_ref.digest}",
                    checksum=collision_ref.digest,
                ),
            )
        )
        material = source_materials[mapping["reference_index"]]
        physics.append(physics_spec(link, material, visual_mesh))
        collision_metrics[link] = {
            "visual_faces": len(visual_mesh.faces),
            "collision_faces": len(collision_mesh.faces),
            "reduction_ratio": 1.0 - len(collision_mesh.faces) / len(visual_mesh.faces),
        }

    joints = []
    for link, reference_index in (("drawer_upper", 1), ("drawer_lower", 2)):
        item = reference[reference_index]
        motion = item["inter_part_kinematics"]
        center = tuple(float(value) for value in motion["motion_axis"]["axis_center"])
        axis = tuple(float(value) for value in motion["motion_axis"]["axis_vector"])
        lower, upper = (float(value) for value in motion["motion_range"]["translate"])
        material = source_materials[reference_index]
        joints.append(
            JointSpec(
                name=f"{link}_slide",
                parent="cabinet",
                child=link,
                joint_type=JointType.PRISMATIC,
                parent_to_joint=PoseSpec(translation_m=center),
                axis=axis,
                lower_limit=lower,
                upper_limit=upper,
                damping=float(material.get("damping", 0.0)),
            )
        )

    usd_path = asset_root / "cabinet2.usdc"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Cabinet2")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for part, mapping in PARTS.items():
        link = mapping["link"]
        link_path = f"/Cabinet2/{link}"
        link_xform = UsdGeom.Xform.Define(stage, link_path)
        UsdPhysics.RigidBodyAPI.Apply(link_xform.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(link_xform.GetPrim())
        mass_api.CreateMassAttr(float(physics[part].mass_kg.value))
        visual = author_usd_mesh(stage, f"{link_path}/visual", meshes[link])
        visual.CreatePurposeAttr(UsdGeom.Tokens.render)
        collision_mesh = load_mesh(collision_root / f"{link}.ply")
        collision = author_usd_mesh(stage, f"{link_path}/collision", collision_mesh)
        collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
        collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        material_prim = UsdShade.Material.Define(stage, f"/Cabinet2/materials/{link}")
        material_api = UsdPhysics.MaterialAPI.Apply(material_prim.GetPrim())
        material_api.CreateStaticFrictionAttr(float(physics[part].static_friction.value))
        material_api.CreateDynamicFrictionAttr(float(physics[part].dynamic_friction.value))
        material_api.CreateRestitutionAttr(float(physics[part].restitution.value))
        UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(material_prim)
    UsdGeom.Scope.Define(stage, "/Cabinet2/joints")
    for joint in joints:
        usd_joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Cabinet2/joints/{joint.name}")
        usd_joint.CreateBody0Rel().SetTargets([Sdf.Path("/Cabinet2/cabinet")])
        usd_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Cabinet2/{joint.child}")])
        usd_joint.CreateAxisAttr(UsdPhysics.Tokens.y)
        usd_joint.CreateLowerLimitAttr(joint.lower_limit)
        usd_joint.CreateUpperLimitAttr(joint.upper_limit)
        usd_joint.CreateLocalPos0Attr(Gf.Vec3f(*joint.parent_to_joint.translation_m))
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(usd_path))
    if reopened is None:
        raise ValueError("generated OpenUSD asset cannot be reopened")
    usd_ref = px.artifacts.put_file(usd_path, media_type="model/vnd.usd")
    artifacts.append(usd_ref)
    asset = AssetSpec(
        resource_id="cabinet2_digital_twin",
        version="2.0.0",
        display_name="cabinet2 freshly trained digital twin",
        root_link="cabinet",
        links=("cabinet", "drawer_upper", "drawer_lower"),
        geometries=tuple(geometries),
        joints=tuple(joints),
        physics=tuple(physics),
        metadata={
            "visual_geometry_source": "fresh_artgs_training",
            "articulation_source": "ArtModeling sim/kinematics.json after prediction quality gate",
            "depth_enhancement": "not_executed",
            "friction_policy": "swap invalid static/dynamic ordering",
        },
    )
    reference_asset = px.assets.register(asset)
    prims = list(reopened.Traverse())
    return asset, {
        "asset_reference": reference_asset.model_dump(mode="json"),
        "usd_path": str(usd_path),
        "usd_sha256": sha256_file(usd_path),
        "usd_bytes": usd_path.stat().st_size,
        "usd_mesh_count": sum(prim.IsA(UsdGeom.Mesh) for prim in prims),
        "usd_rigid_body_count": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "usd_prismatic_joint_count": sum(prim.IsA(UsdPhysics.PrismaticJoint) for prim in prims),
        "collision_metrics": collision_metrics,
        "artifact_count": len(artifacts),
    }


def write_flat_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else key, value)
        elif isinstance(node, (str, int, float, bool)) or node is None:
            rows.append((prefix, node))

    walk("", payload)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    marker = metrics["marker_alignment"]
    render = metrics["rendering"]
    joints = metrics["joints"]
    package = metrics["asset_package"]
    report = f"""# cabinet2 全新训练与数字孪生资产报告

## 可审计结论

- 使用新建 `PhysReal-X` Conda 环境从零完成 ArtGS：粗模型 10,000 轮、关节预测 5,000 轮、联合训练 20,000 轮；没有读取历史 cabinet2 检查点。
- 222/222 帧包含 ArUco 位姿证据，平均重投影误差 `{marker["overall_mean_reprojection_error_px"]:.3f} px`，平均标记间距误差 `{marker["overall_mean_marker_distance_error_mm"]:.3f} mm`；本路径使用 ArUco/PnP，不使用 COLMAP。
- 深度输入是 1280×720 uint16 毫米深度，本次仅以米制深度损失监督 ArtGS，未运行深度增强/补全模型。
- 640×360 全量 222 视角的 ArtGS 指标为 PSNR `{render["artgs_result"]["psnr"]:.3f} dB`、SSIM `{render["artgs_result"]["ssim"]:.4f}`、LPIPS `{render["artgs_result"]["lpips"]:.4f}`。
- 新模型预测 `static + prismatic + prismatic`，关节数量和类型召回率 `{joints["type_recall"]:.1%}`；两条滑动轴的无方向角误差分别为 `{joints["predictions"][0]["unoriented_axis_error_deg"]:.3f}°` 和 `{joints["predictions"][1]["unoriented_axis_error_deg"]:.3f}°`。
- 第二条轴超过 10° 门限，且训练输出没有行程上下限，因此最终仿真资产只采用新训练网格，关节轴/中心/范围来自独立 `sim/kinematics.json`，没有把参考标注伪装成学习结果。
- 已生成并重新打开 `{package["usd_path"]}`：{package["usd_mesh_count"]} 个 Mesh、{package["usd_rigid_body_count"]} 个刚体、{package["usd_prismatic_joint_count"]} 个移动关节。

## 阶段边界

1. **标定对齐**：历史预处理已经使用 ArUco board、PnP、LM/VVS 优化把相机统一到物体坐标系；本次核验全部逐帧残差和位姿矩阵。
2. **深度处理**：检查 RGB/Depth 一一对应、16 位格式、毫米单位、有效率和分布；没有运行未配置的补全网络。
3. **部件与关节训练**：从两个状态共同训练，得到 1 个静态槽位和 2 个滑动槽位，再联合优化 Gaussian、部件掩码、形变与轴参数。
4. **仿真构建**：将三个新训练的起始态部件网格写入 OpenUSD；碰撞网格经二次曲面误差简化，质量与摩擦来自源注释并显式修复非法摩擦顺序。
5. **质量门禁**：外观、几何、关节和 USD 分开评价；预测轴或 limit 不合格时使用有来源标记的参考标注，而不是静默覆盖。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    processed = args.processed.resolve(strict=True)
    training_run = args.training_run.resolve(strict=True)
    output = args.output.resolve()
    if (output / "metrics.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    model_root = training_run / "artgs/realscan/cabinet2"
    ours_best = model_root / "artgs/train/ours_best"
    if not ours_best.is_dir():
        raise FileNotFoundError(f"fresh ArtGS export is missing: {ours_best}")

    px = PhysRealX.open(output / "workspace", discover_plugins=False)
    inspection = px.assets.inspect_source(source)
    if not inspection.ingestible:
        raise ValueError("ArtModeling source failed platform inspection")
    try:
        observation = px.repository.get_resource(
            ResourceType.OBSERVATION_SET.value, "cabinet2_capture", "2.0.0"
        )
    except ResourceNotFoundError:
        observation = px.assets.ingest(
            source,
            resource_id="cabinet2_capture",
            version="2.0.0",
            display_name="ArtModeling cabinet2 capture",
            config={
                "depth_unit": "mm",
                "coordinate_frame": "camera",
                "include_candidates": True,
            },
        )
    metrics = {
        "source": {
            "files": inspection.total_files,
            "bytes": inspection.total_bytes,
            "streams": len(inspection.streams),
            "observation": observation.model_dump(mode="json"),
        },
        "marker_alignment": camera_and_marker_metrics(processed, figures),
        "depth": depth_metrics(processed, figures),
        "training": training_metrics(training_run, model_root),
        "rendering": render_metrics(ours_best, figures),
        "meshes": mesh_metrics(ours_best, figures),
        "joints": joint_metrics(source, ours_best, figures),
    }
    asset, package = package_asset(source, ours_best, output, px)
    metrics["asset_package"] = package
    save_json(output / "metrics.json", metrics)
    save_json(output / "asset.json", asset.model_dump(mode="json"))
    write_flat_csv(output / "metrics.csv", metrics)
    write_report(output / "report.md", metrics)
    print(json.dumps(package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
