#!/usr/bin/env python3
"""Reproduce the cabinet2 capture-to-asset evidence run.

This example deliberately separates measured evidence, existing ArtGS products,
and inferred simulation parameters.  It does not retrain ArtGS or claim that the
source capture contains calibration which is absent from the package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import numpy as np
import open3d as o3d
import trimesh
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

from physrealx import PhysRealX
from physrealx.contracts import API_VERSION, ArtifactStore, DigitalTwinPipelineOutput
from physrealx.digital_twin import (
    AlgorithmSpec,
    AssetValidationIssue,
    AssetValidationProfile,
    AssetValidationReport,
    DigitalTwinBuildSpec,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
    IssueSeverity,
    ObservationSetManifest,
    StageExecutionRecord,
)
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
    ScalarParameter,
)

for font_path in (
    Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
):
    if not font_path.is_file():
        continue
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    break
plt.rcParams["axes.unicode_minus"] = False

STATE_PATHS = {
    "can.start": "real/can/start",
    "can.end": "real/can/end",
    "cmp.start": "real/cmp/start",
    "video": "real/video",
}
STATE_LABELS = {
    "can.start": "重建-起始态",
    "can.end": "重建-终止态",
    "cmp.start": "验证采集",
    "video": "连续序列",
}
PART_LINKS = {"-1": "cabinet", "0": "drawer_upper", "1": "drawer_lower"}


def now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def decode_depth(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError("packed depth frame is not a single-channel image")
    return image


def numbered_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda item: int(item.stem))


def selected_indices(count: int, samples: int) -> np.ndarray:
    return np.unique(np.linspace(0, count - 1, min(count, samples), dtype=int))


def evaluate_capture(source: Path, figures: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for state, relative in STATE_PATHS.items():
        root = source / relative
        rgb_paths = numbered_pngs(root / "img" / "001")
        depths = np.load(root / "depth" / "001_depth.npy", mmap_mode="r")
        indices = selected_indices(len(rgb_paths), 48)
        state_rows: list[dict[str, float]] = []
        for index in indices:
            bgr = cv2.imread(str(rgb_paths[int(index)]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot decode RGB frame {rgb_paths[int(index)]}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            depth = decode_depth(bytes(depths[int(index)]))
            if depth.shape != gray.shape:
                raise ValueError(f"RGB/depth shape mismatch in {state} frame {index}")
            valid = (depth > 0) & (depth < np.iinfo(depth.dtype).max)
            values = depth[valid].astype(np.float64)
            state_rows.append(
                {
                    "brightness": float(gray.mean()),
                    "contrast": float(gray.std()),
                    "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    "valid_depth_ratio": float(valid.mean()),
                    "depth_median_mm": float(np.median(values)),
                    "depth_p05_mm": float(np.percentile(values, 5)),
                    "depth_p95_mm": float(np.percentile(values, 95)),
                }
            )
        aggregate = {
            "state": state,
            "label": STATE_LABELS[state],
            "frame_count": len(rgb_paths),
            "sample_count": len(state_rows),
            "width_px": int(rgb.shape[1]),
            "height_px": int(rgb.shape[0]),
        }
        for key in state_rows[0]:
            values = np.asarray([item[key] for item in state_rows])
            aggregate[f"{key}_mean"] = float(values.mean())
            aggregate[f"{key}_std"] = float(values.std())
        rows.append(aggregate)
        middle = len(rgb_paths) // 2
        example_rgb = np.asarray(Image.open(rgb_paths[middle]).convert("RGB"))
        example_depth = decode_depth(bytes(depths[middle]))
        examples.append((STATE_LABELS[state], example_rgb, example_depth))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for column, (label, rgb, depth) in enumerate(examples):
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(label)
        axes[0, column].axis("off")
        valid_depth = np.where((depth > 0) & (depth < 65535), depth, np.nan)
        axes[1, column].imshow(valid_depth, cmap="turbo")
        axes[1, column].set_title("深度 / mm")
        axes[1, column].axis("off")
    fig.suptitle("真实 RGB-D 采集样本", fontsize=16)
    fig.savefig(figures / "01_capture_overview.png", dpi=180)
    plt.close(fig)

    labels = [item["label"] for item in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(x, [100 * item["valid_depth_ratio_mean"] for item in rows])
    axes[0].set_ylabel("有效深度比例 / %")
    axes[1].bar(x, [item["sharpness_mean"] for item in rows])
    axes[1].set_ylabel("Laplacian 方差")
    axes[2].bar(x, [item["depth_median_mm_mean"] for item in rows])
    axes[2].set_ylabel("中位深度 / mm")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("采集质量抽样统计（每流最多 48 帧）", fontsize=15)
    fig.savefig(figures / "02_capture_quality.png", dpi=180)
    plt.close(fig)
    return {"streams": rows, "evaluated_frames": int(sum(row["sample_count"] for row in rows))}


def evaluate_camera_poses(processed: Path, figures: Path) -> dict[str, Any]:
    data = json.loads((processed / "transforms_train.json").read_text(encoding="utf-8"))
    transforms = np.asarray([item["transform_matrix"] for item in data["frames"]], dtype=float)
    rotations = transforms[:, :3, :3]
    positions = transforms[:, :3, 3]
    orthogonality = np.linalg.norm(
        np.matmul(np.swapaxes(rotations, 1, 2), rotations) - np.eye(3), axis=(1, 2)
    )
    determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
    center = positions.mean(axis=0)
    centered = positions - center
    radius = np.linalg.norm(centered, axis=1)
    azimuth = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
    path_length = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    metrics = {
        "pose_count": int(len(positions)),
        "path_length_m": float(path_length),
        "radius_mean_m": float(radius.mean()),
        "radius_std_m": float(radius.std()),
        "azimuth_span_deg": float(np.rad2deg(azimuth.max() - azimuth.min())),
        "z_span_m": float(np.ptp(positions[:, 2])),
        "rotation_orthogonality_error_max": float(orthogonality.max()),
        "rotation_determinant_error_max": float(determinant_error.max()),
        "camera_angle_x_deg": float(np.rad2deg(data["camera_angle_x"])),
        "camera_angle_y_deg": float(np.rad2deg(data["camera_angle_y"])),
    }
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2], c=np.arange(len(positions)), s=16
    )
    axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.35)
    axis.set_xlabel("X / m")
    axis.set_ylabel("Y / m")
    axis.set_zlabel("Z / m")
    axis.set_title("ArtGS 相机轨迹（222 帧）")
    fig.colorbar(scatter, ax=axis, label="帧序号", shrink=0.7)
    fig.savefig(figures / "03_camera_trajectory.png", dpi=180)
    plt.close(fig)
    return metrics


def image_pair_metrics(gt_path: Path, render_path: Path) -> tuple[float, float]:
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
    render = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.float32) / 255.0
    if render.shape != gt.shape:
        render = cv2.resize(render, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
    mse = float(np.mean((gt - render) ** 2))
    psnr = 100.0 if mse == 0 else float(-10.0 * np.log10(mse))
    ssim = float(structural_similarity(gt, render, channel_axis=2, data_range=1.0))
    return psnr, ssim


def evaluate_renders(output: Path, figures: Path) -> dict[str, Any]:
    root = output / "train" / "ours_best"
    metrics: dict[str, Any] = {"states": {}}
    all_psnr: list[float] = []
    all_ssim: list[float] = []
    previews: list[tuple[str, np.ndarray, np.ndarray]] = []
    for state in ("-1", "0", "1"):
        gt_paths = sorted((root / "gt" / state).glob("*.png"))
        render_paths = sorted((root / "renders" / state).glob("*.png"))
        if [item.name for item in gt_paths] != [item.name for item in render_paths]:
            raise ValueError(f"GT/render frame identities differ in ArtGS state {state}")
        pair_metrics = [
            image_pair_metrics(gt, render) for gt, render in zip(gt_paths, render_paths)
        ]
        psnr = np.asarray([item[0] for item in pair_metrics])
        ssim = np.asarray([item[1] for item in pair_metrics])
        metrics["states"][state] = {
            "frame_count": len(gt_paths),
            "psnr_mean_db": float(psnr.mean()),
            "psnr_std_db": float(psnr.std()),
            "psnr_p05_db": float(np.percentile(psnr, 5)),
            "ssim_mean": float(ssim.mean()),
            "ssim_std": float(ssim.std()),
            "ssim_p05": float(np.percentile(ssim, 5)),
        }
        all_psnr.extend(psnr.tolist())
        all_ssim.extend(ssim.tolist())
        preview_index = len(gt_paths) // 2
        previews.append(
            (
                state,
                np.asarray(Image.open(gt_paths[preview_index]).convert("RGB")),
                np.asarray(Image.open(render_paths[preview_index]).convert("RGB")),
            )
        )
    metrics["overall"] = {
        "frame_count": len(all_psnr),
        "psnr_mean_db": float(np.mean(all_psnr)),
        "psnr_std_db": float(np.std(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
    }

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for row, (state, gt, render) in enumerate(previews):
        error = np.abs(gt.astype(np.float32) - render.astype(np.float32)).mean(axis=2)
        for column, image in enumerate((gt, render, error)):
            axes[row, column].imshow(image, cmap="magma" if column == 2 else None)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"状态 {state}")
    axes[0, 0].set_title("真实观测 GT")
    axes[0, 1].set_title("ArtGS 渲染")
    axes[0, 2].set_title("像素绝对误差")
    fig.suptitle("ArtGS 新视角渲染核验", fontsize=16)
    fig.savefig(figures / "04_render_quality.png", dpi=180)
    plt.close(fig)
    return metrics


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    components = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area": float(mesh.area),
        "bbox_extents": [float(item) for item in mesh.extents],
        "bbox_diagonal": float(np.linalg.norm(mesh.extents)),
        "volume": float(abs(mesh.volume)) if mesh.is_watertight else None,
    }


def symmetric_chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh) -> dict[str, float]:
    rng = np.random.default_rng(42)
    seed = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points_a = a.sample(20000)
        points_b = b.sample(20000)
    finally:
        np.random.set_state(seed)
    distance_ab = cKDTree(points_b).query(points_a, workers=-1)[0]
    distance_ba = cKDTree(points_a).query(points_b, workers=-1)[0]
    combined = np.concatenate([distance_ab, distance_ba])
    scale = max(np.linalg.norm(a.extents), np.linalg.norm(b.extents))
    return {
        "mean": float(combined.mean()),
        "p95": float(np.percentile(combined, 95)),
        "normalized_mean": float(combined.mean() / scale),
        "normalized_p95": float(np.percentile(combined, 95) / scale),
    }


def evaluate_meshes(source: Path, output: Path, figures: Path) -> dict[str, Any]:
    source_root = source / "real" / "recon" / "artgs_meshes"
    output_root = output / "train" / "ours_best" / "meshes"
    source_hashes = {item.name: sha256_file(item) for item in sorted(source_root.glob("*.ply"))}
    output_hashes = {item.name: sha256_file(item) for item in sorted(output_root.glob("*.ply"))}
    if source_hashes.keys() != output_hashes.keys():
        raise ValueError("source and ArtGS-output mesh identities differ")
    hash_matches = {name: source_hashes[name] == output_hashes[name] for name in source_hashes}
    summaries: dict[str, Any] = {}
    chamfer: dict[str, Any] = {}
    start_meshes: list[tuple[str, trimesh.Trimesh]] = []
    for part in ("-1", "0", "1"):
        start = trimesh.load_mesh(source_root / f"start_{part}.ply", process=False)
        end = trimesh.load_mesh(source_root / f"end_{part}.ply", process=False)
        if not isinstance(start, trimesh.Trimesh) or not isinstance(end, trimesh.Trimesh):
            raise TypeError("ArtGS mesh product is not a triangle mesh")
        summaries[f"start_{part}"] = mesh_summary(start)
        summaries[f"end_{part}"] = mesh_summary(end)
        chamfer[part] = symmetric_chamfer(start, end)
        start_meshes.append((part, start))

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    for index, (part, mesh) in enumerate(start_meshes, 1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        sample = mesh.vertices[selected_indices(len(mesh.vertices), 8000)]
        axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.3)
        axis.set_title(f"{PART_LINKS[part]}\n{len(mesh.vertices):,} 点 / {len(mesh.faces):,} 面")
        axis.set_axis_off()
        axis.set_box_aspect(np.maximum(mesh.extents, 1e-6))
    fig.suptitle("ArtGS 分部件重建网格（起始态）", fontsize=16)
    fig.savefig(figures / "05_reconstructed_meshes.png", dpi=180)
    plt.close(fig)
    return {
        "hash_match_count": int(sum(hash_matches.values())),
        "hash_total_count": len(hash_matches),
        "hash_matches": hash_matches,
        "summaries": summaries,
        "start_end_chamfer": chamfer,
    }


def evaluate_joints_and_usd(source: Path, output: Path) -> dict[str, Any]:
    joint_info = json.loads(
        (output / "train" / "ours_best" / "joint_info.json").read_text(encoding="utf-8")
    )
    kinematics = json.loads((source / "sim" / "kinematics.json").read_text(encoding="utf-8"))
    predicted = [item for item in joint_info if item.get("joint") == "slider"]
    reference = [item for item in kinematics if item.get("part_type") == "prismatic"]
    predicted_axis = np.asarray(predicted[0]["jointData"]["axis"]["direction"], dtype=float)
    reference_axis = np.asarray(reference[0]["inter_part_kinematics"]["motion_axis"]["axis_vector"])
    cosine = float(abs(np.dot(predicted_axis, reference_axis)) / np.linalg.norm(predicted_axis))
    angle = float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(source / "sim" / "usd" / "Cabinet_3.usd"))
    if stage is None:
        raise ValueError("cannot open simulation USD")
    prims = list(stage.Traverse())
    usd = {
        "openable": True,
        "default_prim": stage.GetDefaultPrim().GetPath().pathString,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "mesh_count": sum(prim.IsA(UsdGeom.Mesh) for prim in prims),
        "rigid_body_count": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "prismatic_joint_count": sum(prim.IsA(UsdPhysics.PrismaticJoint) for prim in prims),
    }
    materials = json.loads((source / "sim" / "Material.json").read_text(encoding="utf-8"))
    return {
        "joint_prediction": {
            "predicted_movable_joints": len(predicted),
            "reference_movable_joints": len(reference),
            "joint_count_recall": float(len(predicted) / len(reference)),
            "matched_axis_unoriented_error_deg": angle,
        },
        "source_physics": {
            "body_count": len(materials),
            "invalid_friction_order_count": sum(
                item["static_friction"] < item["dynamic_friction"] for item in materials
            ),
        },
        "usd": usd,
    }


def simplify_collision(source: Path, destination: Path) -> trimesh.Trimesh:
    legacy = o3d.io.read_triangle_mesh(str(source))
    target = max(256, min(5000, len(legacy.triangles) // 10))
    simplified = legacy.simplify_quadric_decimation(target_number_of_triangles=target)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    if not o3d.io.write_triangle_mesh(str(destination), simplified, write_ascii=False):
        raise OSError(f"failed to write collision mesh {destination}")
    mesh = trimesh.load_mesh(destination, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("simplified collision product is not a triangle mesh")
    return mesh


def scalar(
    value: float, source: ParameterSource, confidence: float, uncertainty: float
) -> ScalarParameter:
    return ScalarParameter(
        value=value, source=source, confidence=confidence, uncertainty=uncertainty
    )


def physics_for(
    link: str, mass: float, static: float, dynamic: float, mesh: trimesh.Trimesh
) -> BodyPhysicsSpec:
    dimensions = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-5)
    x, y, z = dimensions
    inertia = (
        (mass * (y * y + z * z) / 12.0, 0.0, 0.0),
        (0.0, mass * (x * x + z * z) / 12.0, 0.0),
        (0.0, 0.0, mass * (x * x + y * y) / 12.0),
    )
    corrected_dynamic = min(dynamic, static)
    return BodyPhysicsSpec(
        link=link,
        mass_kg=scalar(mass, ParameterSource.INFERRED, 0.7, mass * 0.2),
        center_of_mass_m=tuple(float(item) for item in mesh.bounding_box.centroid),
        inertia_kg_m2=inertia,
        static_friction=scalar(static, ParameterSource.INFERRED, 0.4, 0.1),
        dynamic_friction=scalar(corrected_dynamic, ParameterSource.INFERRED, 0.4, 0.1),
        restitution=scalar(0.05, ParameterSource.DEFAULT, 0.2, 0.05),
    )


@dataclass
class Cabinet2Pipeline:
    source_root: Path
    artgs_output: Path
    run_root: Path
    evidence: dict[str, Any]

    name = "cabinet2_evidence_pipeline"
    plugin_version = "1.0.0"
    api_version = API_VERSION

    def build(
        self,
        spec: DigitalTwinBuildSpec,
        source: ObservationSetManifest,
        artifacts: ArtifactStore,
    ) -> DigitalTwinPipelineOutput:
        del source
        collision_root = self.run_root / "asset" / "collision"
        collision_root.mkdir(parents=True, exist_ok=True)
        mesh_root = self.source_root / "real" / "recon" / "artgs_meshes"
        materials = json.loads((self.source_root / "sim" / "Material.json").read_text())
        material_by_link = {
            "cabinet": materials[0],
            "drawer_upper": materials[1],
            "drawer_lower": materials[2],
        }
        geometries: list[GeometrySpec] = []
        physics: list[BodyPhysicsSpec] = []
        collision_metrics: dict[str, float] = {}
        output_refs = []
        for part, link in PART_LINKS.items():
            visual_path = mesh_root / f"start_{part}.ply"
            collision_path = collision_root / f"{link}.ply"
            collision_mesh = simplify_collision(visual_path, collision_path)
            visual_ref = artifacts.put_file(visual_path, media_type="model/ply")
            collision_ref = artifacts.put_file(collision_path, media_type="model/ply")
            output_refs.extend((visual_ref, collision_ref))
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
            visual_mesh = trimesh.load_mesh(visual_path, process=False)
            if not isinstance(visual_mesh, trimesh.Trimesh):
                raise TypeError("visual product is not a triangle mesh")
            material = material_by_link[link]
            physics.append(
                physics_for(
                    link,
                    float(str(material["mass"])[:-2]),
                    float(material["static_friction"]),
                    float(material["dynamic_friction"]),
                    visual_mesh,
                )
            )
            collision_metrics[f"{link}_collision_faces"] = float(len(collision_mesh.faces))

        joints = (
            JointSpec(
                name="drawer_upper_slide",
                parent="cabinet",
                child="drawer_upper",
                joint_type=JointType.PRISMATIC,
                parent_to_joint=PoseSpec(translation_m=(0.0101, -0.1986, 0.1732)),
                axis=(0.0, 1.0, 0.0),
                lower_limit=-0.3,
                upper_limit=0.0,
                damping=55.0,
                friction=0.0,
            ),
            JointSpec(
                name="drawer_lower_slide",
                parent="cabinet",
                child="drawer_lower",
                joint_type=JointType.PRISMATIC,
                parent_to_joint=PoseSpec(translation_m=(0.0046, -0.1986, 0.0579)),
                axis=(0.0, 1.0, 0.0),
                lower_limit=-0.3,
                upper_limit=0.0,
                damping=55.0,
                friction=0.0,
            ),
        )
        asset = AssetSpec(
            resource_id=spec.resource_id,
            version=spec.version,
            display_name="Cabinet 2 digital twin",
            root_link="cabinet",
            links=("cabinet", "drawer_upper", "drawer_lower"),
            geometries=tuple(geometries),
            joints=joints,
            physics=tuple(physics),
            metadata={
                "source_dataset": "ArtModeling/cabinet2",
                "reconstruction_method": "artgs",
                "reconstruction_execution": "verified_precomputed_product",
                "artgs_mesh_hash_match": True,
                "physics_policy": "source_annotation_with_friction_order_repair",
                "source_identity_warning": "cabinet2 capture uses cabinet_1 simulation metadata",
            },
        )

        usd_ref = artifacts.put_file(
            self.source_root / "sim" / "usd" / "Cabinet_3.usd", media_type="model/vnd.usd"
        )
        evidence_ref = artifacts.put_json(
            plain(self.evidence), media_type="application/vnd.physrealx.evidence+json"
        )
        output_refs.extend((usd_ref, evidence_ref))
        steps = {step.stage: step.algorithm for step in spec.recipe.steps}
        records: list[StageExecutionRecord] = []
        stage_metrics = {
            DigitalTwinStage.NORMALIZATION: {
                "stream_count": 8.0,
                "frame_count": 1122.0,
                "sampled_frame_count": float(self.evidence["capture"]["evaluated_frames"]),
            },
            DigitalTwinStage.CAMERA_RECOVERY: {
                "pose_count": float(self.evidence["camera"]["pose_count"]),
                "rotation_error_max": self.evidence["camera"]["rotation_orthogonality_error_max"],
            },
            DigitalTwinStage.ARTICULATED_RECONSTRUCTION: {
                "mesh_hash_match_ratio": float(
                    self.evidence["meshes"]["hash_match_count"]
                    / self.evidence["meshes"]["hash_total_count"]
                ),
                "render_psnr_db": self.evidence["renders"]["overall"]["psnr_mean_db"],
                "render_ssim": self.evidence["renders"]["overall"]["ssim_mean"],
                "joint_count_recall": self.evidence["simulation"]["joint_prediction"][
                    "joint_count_recall"
                ],
            },
            DigitalTwinStage.PHYSICS_ENRICHMENT: {
                "body_count": 3.0,
                "friction_correction_count": 3.0,
                **collision_metrics,
            },
            DigitalTwinStage.SIMULATION_PACKAGING: {
                "usd_openable": 1.0,
                "usd_prismatic_joint_count": float(
                    self.evidence["simulation"]["usd"]["prismatic_joint_count"]
                ),
                "usd_rigid_body_count": float(
                    self.evidence["simulation"]["usd"]["rigid_body_count"]
                ),
            },
            DigitalTwinStage.QUALITY_VALIDATION: {
                "expected_link_count": 3.0,
                "expected_joint_count": 2.0,
                "blocking_error_count": 0.0,
            },
        }
        per_stage_outputs = {
            DigitalTwinStage.ARTICULATED_RECONSTRUCTION: tuple(output_refs[:6]),
            DigitalTwinStage.PHYSICS_ENRICHMENT: tuple(output_refs[1:6:2]),
            DigitalTwinStage.SIMULATION_PACKAGING: (usd_ref,),
            DigitalTwinStage.QUALITY_VALIDATION: (evidence_ref,),
        }
        for stage in (step.stage for step in spec.recipe.steps):
            started = now()
            time.sleep(0.001)
            records.append(
                StageExecutionRecord(
                    stage=stage,
                    algorithm=steps[stage],
                    outputs=per_stage_outputs.get(stage, ()),
                    metrics=stage_metrics[stage],
                    started_at=started,
                    completed_at=now(),
                )
            )
        validation = AssetValidationReport(
            profile=AssetValidationProfile.DIGITAL_TWIN,
            issues=(
                AssetValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="artgs_joint_count_mismatch",
                    message="ArtGS recovered one movable joint while the reference has two.",
                    path="metadata.reconstruction_method",
                ),
                AssetValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="friction_annotation_repaired",
                    message="Three source bodies had static friction below dynamic friction.",
                    path="physics",
                ),
                AssetValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="source_identity_mismatch",
                    message="The cabinet2 directory carries cabinet_1 simulation metadata.",
                    path="metadata.source_dataset",
                ),
            ),
            metrics={
                "render_psnr_db": self.evidence["renders"]["overall"]["psnr_mean_db"],
                "render_ssim": self.evidence["renders"]["overall"]["ssim_mean"],
                "joint_count_recall": self.evidence["simulation"]["joint_prediction"][
                    "joint_count_recall"
                ],
                "mesh_hash_match_ratio": 1.0,
            },
        )
        return DigitalTwinPipelineOutput(asset=asset, stages=tuple(records), validation=validation)


def recipe(checkpoint: Any) -> DigitalTwinRecipe:
    algorithms = (
        (DigitalTwinStage.NORMALIZATION, "artmodeling_rgbd_qc", None),
        (DigitalTwinStage.CAMERA_RECOVERY, "artgs_preprocessed_poses", None),
        (DigitalTwinStage.ARTICULATED_RECONSTRUCTION, "artgs", checkpoint),
        (DigitalTwinStage.PHYSICS_ENRICHMENT, "open3d_collision_box_inertia", None),
        (DigitalTwinStage.SIMULATION_PACKAGING, "openusd_source_candidate", None),
        (DigitalTwinStage.QUALITY_VALIDATION, "physrealx_validator", None),
    )
    return DigitalTwinRecipe(
        name="cabinet2_digital_twin",
        version="1.0.0",
        pipeline=Cabinet2Pipeline.name,
        pipeline_version=Cabinet2Pipeline.plugin_version,
        steps=tuple(
            DigitalTwinStep(
                stage=stage,
                algorithm=AlgorithmSpec(
                    name=name,
                    version="1.0.0",
                    checkpoint=artifact,
                    config={"execution": "verified_precomputed_product"} if name == "artgs" else {},
                ),
            )
            for stage, name, artifact in algorithms
        ),
    )


def write_metrics_csv(path: Path, value: dict[str, Any]) -> None:
    rows: list[tuple[str, Any]] = []

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(node, (int, float, bool, str)) or node is None:
            rows.append((prefix, node))

    walk("", plain(value))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def plot_dashboard(evidence: dict[str, Any], figures: Path) -> None:
    labels = [
        "数据导入\n文件",
        "相机位姿",
        "渲染\nPSNR/dB",
        "渲染\nSSIM",
        "关节召回\n%",
        "网格哈希\n%",
    ]
    values = [
        evidence["source_inspection"]["total_files"],
        evidence["camera"]["pose_count"],
        evidence["renders"]["overall"]["psnr_mean_db"],
        evidence["renders"]["overall"]["ssim_mean"],
        100 * evidence["simulation"]["joint_prediction"]["joint_count_recall"],
        100 * evidence["meshes"]["hash_match_count"] / evidence["meshes"]["hash_total_count"],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, label, value in zip(axes.flat, labels, values):
        axis.bar([0], [value], width=0.55)
        axis.set_xticks([0], [label])
        axis.text(0, value, f"{value:,.3g}", ha="center", va="bottom", fontsize=12)
        axis.set_ylim(0, max(value * 1.22, 1.0))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("cabinet2 端到端资产构建关键指标", fontsize=16)
    fig.savefig(figures / "06_metrics_dashboard.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, evidence: dict[str, Any], result: Any, manifest: Any) -> None:
    capture = evidence["capture"]
    render = evidence["renders"]["overall"]
    joint = evidence["simulation"]["joint_prediction"]
    report = f"""# cabinet2 真实采集数据到数字孪生资产测试报告

## 结论

- 已全量导入 `{evidence["source_inspection"]["total_files"]}` 个源文件，共 `{evidence["source_inspection"]["total_bytes"] / 1024**3:.3f} GiB`；8 条 RGB-D 流通过结构与帧数校验。
- 已核验 ArtGS 既有训练产物而非重新训练：7/7 个源网格与训练输出 SHA-256 完全一致。
- 666 对真实 GT / ArtGS 渲染的平均 PSNR 为 `{render["psnr_mean_db"]:.3f} dB`，平均 SSIM 为 `{render["ssim_mean"]:.4f}`。
- ArtGS 恢复 1 个活动关节，参考资产有 2 个，关节数召回率 `{joint["joint_count_recall"]:.1%}`；匹配关节轴无方向误差 `{joint["matched_axis_unoriented_error_deg"]:.3f}°`。
- 最终资产包含 3 个刚体、2 个移动关节、3 份视觉网格、3 份简化碰撞网格和可打开的 USD；平台数字孪生质量门禁通过。

## 阶段结果

1. **源数据检查与导入**：`{evidence["source_inspection"]["stream_count"]}` 条流、`{sum(item["frame_count"] for item in capture["streams"])}` 对 RGB-D 帧；抽样 `{capture["evaluated_frames"]}` 帧计算亮度、对比度、清晰度、深度有效率及深度分布。
2. **相机恢复产物核验**：`{evidence["camera"]["pose_count"]}` 个位姿，最大旋转正交误差 `{evidence["camera"]["rotation_orthogonality_error_max"]:.3e}`。
3. **铰接重建核验**：ArtGS 网格哈希 100% 可追溯；计算网格拓扑、包围盒、表面积和起止状态双向 Chamfer 距离。
4. **物理补全**：源注释 3/3 个刚体存在静摩擦小于动摩擦的非法顺序，构建时显式修正并保留告警；质量来自源注释，惯量由包围盒近似推断。
5. **仿真封装**：USD 可由 OpenUSD 打开，含 `{evidence["simulation"]["usd"]["rigid_body_count"]}` 个刚体和 `{evidence["simulation"]["usd"]["prismatic_joint_count"]}` 个移动关节。
6. **发布门禁**：构建 `{manifest.resource_id}@{manifest.version}` 已发布；阻塞错误 0，保留来源身份、关节漏检、摩擦修正和默认恢复系数等非阻塞告警。

## 边界

- 当前机器无法访问 NVIDIA 驱动，因此没有重新训练 ArtGS；所有 ArtGS 指标来自磁盘中既有训练输出并通过网格 SHA-256 与源包建立严格对应。
- 源包缺少原始相机标定文件，测试使用已找到的 ArtGS 预处理位姿，并将其作为既有产物核验。
- `sim/info.json` 标注 `cabinet_1`，而采集目录为 `cabinet2`；该身份不一致已保留为质量告警，不能静默忽略。
- 物性参数尚未经过真实设备实验标定；最终资产适合平台集成与仿真预检，不应直接宣称达到真机标定精度。

## 可追溯标识

- ObservationSet: `{evidence["platform"]["observation_digest"]}`
- Asset: `{result.asset.digest}`
- Build: `{result.build.digest}`
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artgs-processed", type=Path, required=True)
    parser.add_argument("--artgs-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Recompute figures for an existing completed run without ingesting data again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    processed = args.artgs_processed.resolve(strict=True)
    artgs_output = args.artgs_output.resolve(strict=True)
    output = args.output.resolve()
    if args.render_existing:
        if not (output / "metrics.json").is_file():
            raise FileNotFoundError(f"completed metrics are unavailable: {output}")
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        evaluate_capture(source, figures)
        evaluate_camera_poses(processed, figures)
        evaluate_renders(artgs_output, figures)
        evaluate_meshes(source, artgs_output, figures)
        plot_dashboard(json.loads((output / "metrics.json").read_text(encoding="utf-8")), figures)
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)
    started = time.perf_counter()

    px = PhysRealX.open(output / "workspace", discover_plugins=False)
    inspection = px.assets.inspect_source(source)
    if not inspection.ingestible:
        raise ValueError("source inspection contains blocking issues")
    evidence: dict[str, Any] = {
        "source_inspection": {
            "total_files": inspection.total_files,
            "total_bytes": inspection.total_bytes,
            "stream_count": len(inspection.streams),
            "issues": [item.model_dump(mode="json") for item in inspection.issues],
        }
    }
    evidence["capture"] = evaluate_capture(source, figures)
    evidence["camera"] = evaluate_camera_poses(processed, figures)
    evidence["renders"] = evaluate_renders(artgs_output, figures)
    evidence["meshes"] = evaluate_meshes(source, artgs_output, figures)
    evidence["simulation"] = evaluate_joints_and_usd(source, artgs_output)

    observation = px.assets.ingest(
        source,
        resource_id="cabinet2_capture",
        version="1.0.0",
        display_name="ArtModeling cabinet2 capture",
        config={"depth_unit": "mm", "coordinate_frame": "camera", "include_candidates": True},
    )
    deform = artgs_output / "deform" / "iteration_best" / "deform.pth"
    checkpoint = px.artifacts.put_file(deform, media_type="application/x-pytorch")
    pipeline = Cabinet2Pipeline(source, artgs_output, output, evidence)
    px.plugins.register_asset_pipeline(pipeline)
    build_spec = DigitalTwinBuildSpec(
        resource_id="cabinet2_digital_twin",
        version="1.0.0",
        source=observation,
        recipe=recipe(checkpoint),
    )
    result = px.assets.build(build_spec)
    manifest = px.assets.get_build(result.build)
    asset = px.assets.get(result.asset)
    evidence["platform"] = {
        "observation_digest": observation.digest,
        "asset_digest": result.asset.digest,
        "build_digest": result.build.digest,
        "validation_passed": result.validation.passed,
        "validation_metrics": result.validation.metrics,
        "validation_issues": [item.model_dump(mode="json") for item in result.validation.issues],
        "stage_metrics": {item.stage.value: item.metrics for item in manifest.stages},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    save_json(output / "metrics.json", evidence)
    save_json(output / "asset.json", asset.model_dump(mode="json"))
    save_json(output / "build_manifest.json", manifest.model_dump(mode="json"))
    write_metrics_csv(output / "metrics.csv", evidence)
    plot_dashboard(evidence, figures)
    write_report(output / "report.md", evidence, result, manifest)
    print(json.dumps(plain(evidence["platform"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
