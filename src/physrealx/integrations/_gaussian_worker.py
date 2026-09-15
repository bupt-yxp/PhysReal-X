"""Isolated native ArtGS rasterization worker, launched by gaussian_render only."""

from __future__ import annotations

from typing import Any


def main() -> None:
    import json
    import math
    import sys
    from argparse import Namespace
    from importlib import import_module
    from pathlib import Path

    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    from physrealx.reconstruction import joint_transform

    scene = import_module("scene")
    cameras = import_module("scene.cameras")
    renderer = import_module("gaussian_renderer")
    graphics = import_module("utils.graphics_utils")
    quaternion_mul = import_module("utils.dual_quaternion").quaternion_mul
    root = Path(sys.argv[1])
    request = json.loads((root / "request.json").read_text())
    configuration = json.loads((root / "config.json").read_text())
    model = scene.DeformModel(Namespace(**configuration))
    model.deform.load_state_dict(torch.load(root / "deform.pth", weights_only=True))
    model.update(20000)
    gaussians = scene.GaussianModel(configuration["sh_degree"])
    gaussians.load_ply(str(root / "point_cloud.ply"))
    camera = request["camera"]
    fovy = math.radians(camera["fovy_degrees"])
    fovx = 2 * math.atan(math.tan(fovy / 2) * camera["width"] / camera["height"])
    view = torch.tensor(request["view_matrix"], dtype=torch.float32, device="cuda").T
    projection = graphics.getProjectionMatrix(0.01, camera["max_depth_m"], fovx, fovy).T.cuda()
    viewpoint = cameras.MiniCam(
        camera["width"],
        camera["height"],
        fovy,
        fovx,
        0.01,
        camera["max_depth_m"],
        view,
        view @ projection,
    )
    outputs: dict[str, list[Any]] = {name: [] for name in ("rgb", "depth", "alpha")}
    with torch.no_grad():
        translations, rotations = model.deform.interpolate(gaussians, [0.0])
        start_points = (gaussians.get_xyz + translations[0]).cpu().numpy()
        aligned = start_points @ np.asarray(request["registration_rotation"]).T
        aligned += np.asarray(request["registration_translation"])
        distance = cKDTree(np.load(root / "reference.npy", allow_pickle=False)).query(aligned)[0]
        visible = torch.tensor(distance <= request["max_surface_distance_m"], device="cuda")
        if int(visible.sum()) < max(1000, len(visible) * 0.75):
            raise ValueError("registered Gaussian appearance has insufficient reference coverage")
        mask = model.deform.get_mask(gaussians.get_xyz, False).argmax(-1)
        moving = mask == request["moving_group"]
        initial_points = gaussians.get_xyz + translations[0]
        for position in request["positions"]:
            motion = joint_transform(
                request["joint_kind"],
                request["capture_axis"],
                request["capture_origin"],
                position - request["start_position"],
            )
            rotation = torch.tensor(motion[:3, :3], device="cuda", dtype=torch.float32)
            shift = torch.tensor(motion[:3, 3], device="cuda", dtype=torch.float32)
            delta = (initial_points @ rotation.T + shift - initial_points) * moving[:, None]
            quaternion = Rotation.from_matrix(motion[:3, :3]).as_quat()[[3, 0, 1, 2]]
            quaternion = torch.tensor(quaternion, device="cuda", dtype=torch.float32).expand(
                len(moving), -1
            )
            posed_rotations = torch.where(
                moving[:, None], quaternion_mul(quaternion, rotations[0]), rotations[0]
            )
            image = renderer.render(
                viewpoint,
                gaussians,
                Namespace(debug=False, compute_cov3D_python=False),
                torch.zeros(3, device="cuda"),
                translations[0] + delta,
                posed_rotations,
                vis_mask=visible,
            )
            outputs["rgb"].append(
                (image["render"].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
            )
            outputs["alpha"].append(image["alpha"][0].cpu().numpy())
            depth = torch.where(
                image["alpha"] > 1e-4, image["depth"] / image["alpha"].clamp_min(1e-4), 0
            )
            outputs["depth"].append(depth[0].cpu().numpy())
        for name, frames in outputs.items():
            np.save(root / (name + ".npy"), np.stack(frames), allow_pickle=False)
        diagnostics = {
            "gaussians": len(mask),
            "moving_gaussians": int(moving.sum()),
            "rejected_background_gaussians": int((~visible).sum()),
            "physical_motion_span": max(request["positions"]) - min(request["positions"]),
            "foreground_pixels": int((np.stack(outputs["alpha"]) > 0.5).sum()),
        }
        (root / "diagnostics.json").write_text(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
