"""Observed RGB, depth and silhouette constraints through the existing mesh renderer."""

from __future__ import annotations

from typing import Any

import numpy as np


class VideoRenderLoss:
    """Fixed observed images and vertex appearance; only geometry/motion receive gradients."""

    def __init__(
        self,
        meshes: dict[str, Any],
        observations: dict[str, Any],
        device: str,
        motion: dict[str, Any],
        end_state: dict[str, Any] | None = None,
        end_position: float | None = None,
    ) -> None:
        import torch
        from scipy.ndimage import distance_transform_edt

        self.device = device
        self.faces = {
            n: torch.as_tensor(np.asarray(m.faces), device=device, dtype=torch.int64)
            for n, m in meshes.items()
        }
        self.colors = {
            n: torch.as_tensor(
                np.asarray(m.visual.vertex_colors)[:, :3] / 255, device=device, dtype=torch.float32
            )
            for n, m in meshes.items()
        }
        self.end_position = motion["range"][1] if end_position is None else end_position
        self.end_vertices = {
            n: torch.as_tensor(np.array(m.vertices), device=device, dtype=torch.float64)
            for n, m in (end_state or {}).items()
        }
        self.end_colors = {
            n: torch.as_tensor(
                np.array(m.visual.vertex_colors[:, :3]) / 255, device=device, dtype=torch.float32
            )
            for n, m in (end_state or {}).items()
        }
        self.parent = motion["parent"]
        self.child = motion["child"]
        self.faces["__object__"] = torch.cat(
            [self.faces[self.parent], self.faces[self.child] + len(meshes[self.parent].vertices)]
        )
        self.colors["__object__"] = torch.cat([self.colors[self.parent], self.colors[self.child]])
        if self.end_colors:
            self.end_colors["__object__"] = torch.cat(
                [self.end_colors[self.parent], self.end_colors[self.child]]
            )
        self.frames = []
        for name, partitions in observations.items():
            for frame in partitions["fit"]:
                if frame.image is None:
                    continue
                rgb, depth, mask, camera, intrinsics = frame.image
                # Distance outside the silhouette supplies a gradient even when
                # a projected vertex is outside the hard rasterized object.
                outside = distance_transform_edt(~mask).astype(np.float32)
                self.frames.append(
                    (
                        name,
                        frame.position,
                        *[
                            torch.as_tensor(
                                np.array(x, copy=True), device=device, dtype=torch.float32
                            )
                            for x in (rgb / 255, depth, mask, outside, camera, intrinsics)
                        ],
                    )
                )

    def __call__(
        self,
        vertices: dict[str, Any],
        transform: Any,
        axis: Any,
        origin: Any,
        child: str,
        iteration: int,
    ) -> Any:
        if not self.frames:
            return axis.sum() * 0
        frame = self.frames[iteration % len(self.frames)]
        q = frame[1]
        posed = {n: transform(v, q, axis, origin) if n == child else v for n, v in vertices.items()}
        loss = self._render(posed, frame, axis)
        if self.end_vertices:
            backward = {
                n: transform(v, q - self.end_position, axis, origin) if n == child else v
                for n, v in self.end_vertices.items()
            }
            loss = 0.5 * (loss + self._render(backward, frame, axis, self.end_colors))
        return loss

    def observed_loss(
        self,
        vertices: Any,
        transform: Any,
        axis: Any,
        origin: Any,
        child: str,
        rgb_tolerance: float,
        mask_tolerance: float,
    ) -> Any:
        """Use all fit RGB-D and part masks during crossed-surface recovery."""
        import torch

        if not self.frames:
            return axis.sum() * 0
        if rgb_tolerance <= 0 or mask_tolerance <= 0:
            raise ValueError("image tolerances must be positive")
        images = []
        for frame in self.frames:
            posed = {
                name: transform(value, frame[1], axis, origin) if name == child else value
                for name, value in vertices.items()
            }
            terms = self.render_terms(posed, frame, axis)
            images.append(
                terms["rgb"] / rgb_tolerance
                + terms["mask"] / mask_tolerance
                + 0.1 * terms["depth"]
                + 0.05 * terms["boundary"]
            )
        return torch.stack(images).mean()

    def _raster(self, vertices: Any, frame: Any, *, soft: bool = False) -> tuple[Any, ...]:
        import torch
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes

        name, _q, _rgb, _depth, mask, _outside, camera, intrinsics = frame
        # Visibility belongs to the complete articulated object. Rendering a
        # link alone incorrectly exposes its pixels through another link.
        v = torch.cat([vertices[self.parent], vertices[self.child]])
        cv = v.float() @ camera[:3, :3].T + camera[:3, 3]
        fx, fy, cx, cy = intrinsics
        height, width = mask.shape
        z = cv[:, 2].clamp_min(1e-5)
        u, w = fx * cv[:, 0] / z + cx, fy * cv[:, 1] / z + cy
        # PyTorch3D: positive X left, positive Y up; shorter image side spans [-1,1].
        scale = min(height, width) / 2
        # OpenCV locates pixel centres at integer coordinates; rasterizer NDC
        # centres lie half a pixel inside the image boundary.
        ndc = torch.stack((((width - 1) / 2 - u) / scale, ((height - 1) / 2 - w) / scale, z), dim=1)
        sigma = (0.5 / scale) ** 2
        scene = Meshes(verts=[ndc], faces=[self.faces["__object__"]])
        pix, zbuf, bary, distances = rasterize_meshes(
            scene,
            image_size=(height, width),
            blur_radius=9 * sigma if soft else 0.0,
            faces_per_pixel=4 if soft else 1,
            clip_barycentric_coords=True,
            perspective_correct=True,
            cull_backfaces=False,
            max_faces_per_bin=len(self.faces["__object__"]),
        )
        if name != "__object__":
            split = len(self.faces[self.parent])

            def owned(ids: Any) -> Any:
                return ((ids >= 0) & (ids < split)) if name == self.parent else ids >= split

            visible = owned(pix)
            if soft:
                # Soft neighbours can sort in front of an interior face. The
                # hard scene supplies occlusion; soft faces supply gradients
                # at the observed link's visible silhouette only.
                hard_ids = rasterize_meshes(
                    scene,
                    image_size=(height, width),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    clip_barycentric_coords=True,
                    perspective_correct=True,
                    cull_backfaces=False,
                    max_faces_per_bin=len(self.faces["__object__"]),
                )[0]
                visible &= owned(hard_ids) | (hard_ids < 0)
            pix = torch.where(visible, pix - (split if name == self.child else 0), -1)
            # Vertex projection terms still belong to the requested part.
            part = (
                slice(0, len(vertices[self.parent]))
                if name == self.parent
                else slice(len(vertices[self.parent]), None)
            )
            cv, u, w = cv[part], u[part], w[part]
        return pix, zbuf, bary, distances, cv, u, w, sigma

    def _render(self, vertices: Any, frame: Any, axis: Any, colors: Any = None) -> Any:
        terms = self.render_terms(vertices, frame, axis, colors)
        return terms["rgb"] + 0.1 * terms["depth"] + 0.05 * (terms["boundary"] + terms["coverage"])

    def render_terms(
        self, vertices: Any, frame: Any, axis: Any, colors: Any = None
    ) -> dict[str, Any]:
        import torch

        colors = self.colors if colors is None else colors
        name, _q, rgb, depth, mask, outside, _camera, _intrinsics = frame
        pix, zbuf, bary, distances, cv, u, w, sigma = self._raster(vertices, frame)
        height, width = mask.shape
        ids, zbuf, bary = pix[0, ..., 0], zbuf[0, ..., 0], bary[0, ..., 0, :]
        visible = (ids >= 0) & (distances[0, ..., 0] <= 0) & (mask > 0)
        if visible.any():
            color = (
                colors[name][self.faces[name][ids[visible]]] * bary[visible].unsqueeze(-1)
            ).sum(1)
            photo = torch.nn.functional.smooth_l1_loss(color, rgb[visible], beta=0.1)
        else:
            photo = axis.sum() * 0
        metric_visible = visible & torch.isfinite(depth) & (depth > 0)
        if metric_visible.any():
            metric = torch.nn.functional.huber_loss(
                (zbuf[metric_visible] - depth[metric_visible]) / 0.001,
                torch.zeros_like(zbuf[metric_visible]),
                delta=3,
            )
        else:
            metric = axis.sum() * 0
        grid = torch.stack((2 * u / (width - 1) - 1, 2 * w / (height - 1) - 1), dim=1)
        distance = torch.nn.functional.grid_sample(
            outside[None, None],
            grid[None, None],
            align_corners=True,
            padding_mode="border",
        )[0, 0, 0]
        observed_depth = torch.nn.functional.grid_sample(
            depth[None, None],
            grid[None, None],
            mode="nearest",
            align_corners=True,
        )[0, 0, 0]
        valid = (
            ((observed_depth <= 0) | (cv[:, 2] <= observed_depth + 0.02))
            & (cv[:, 2] > 0)
            & (u >= 0)
            & (u < width)
            & (w >= 0)
            & (w < height)
        )
        silhouette = distance[valid].square().mean() if valid.any() else axis.sum() * 0
        # Blurred faces are sorted by depth, including faces whose interiors
        # do not cover the pixel. On dense meshes these can occupy every slot
        # in front of its true surface. RGB/depth visibility therefore uses
        # the hard raster above; the soft raster serves only silhouette loss.
        soft_pix, _, _, soft_distances, *_ = self._raster(vertices, frame, soft=True)
        alpha = 1 - (1 - torch.sigmoid(-soft_distances[0] / sigma) * (soft_pix[0] >= 0)).prod(-1)
        foreground = mask > 0
        coverage = (1 - alpha[foreground]).square().mean() if foreground.any() else axis.sum() * 0
        # Pixel penalties have a consistent scale at different render resolutions.
        pixel_scale = 160 / max(height, width)
        intersection = (alpha * mask).sum()
        union = (alpha + mask - alpha * mask).sum().clamp_min(1)
        return {
            "rgb": photo,
            "depth": metric,
            "boundary": silhouette * pixel_scale**2,
            "coverage": coverage,
            "mask": 1 - intersection / union,
        }


def check_images(meshes: Any, observations: Any, motion: Any, device: str) -> list[dict[str, Any]]:
    """Independent held-out image measurements, never optimization correspondences.

    Report each camera/pose separately so a better view cannot hide degradation
    at an endpoint. Missing overlap is a failed measurement, not zero error.
    """
    import torch

    from physrealx.reconstruction import joint_transform

    selected = {n: {"fit": p["check"]} for n, p in observations.items()}
    renderer = VideoRenderLoss(meshes, selected, device, motion)
    ids = [f.frame_id for p in observations.values() for f in p["check"] if f.image is not None]
    rows = []
    with torch.no_grad():
        for frame_id, frame in zip(ids, renderer.frames, strict=True):
            name, q, rgb, depth, mask, _outside, _camera, _intrinsics = frame
            matrix = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            posed = {}
            for n, mesh in meshes.items():
                v = np.asarray(mesh.vertices)
                if n == motion["child"]:
                    v = v @ matrix[:3, :3].T + matrix[:3, 3]
                posed[n] = torch.tensor(np.array(v), device=device, dtype=torch.float32)
            pix, zbuf, bary, distances, *_ = renderer._raster(posed, frame)
            index = pix[0, ..., 0]
            predicted = (index >= 0) & (distances[0, ..., 0] <= 0)
            expected = mask > 0
            overlap = predicted & expected
            union = predicted | expected
            metric = overlap & torch.isfinite(depth) & (depth > 0)
            rgb_error = depth_error = None
            if overlap.any():
                color = (
                    renderer.colors[name][renderer.faces[name][index[overlap]]]
                    * bary[0, ..., 0, :][overlap].unsqueeze(-1)
                ).sum(1)
                rgb_error = float((color - rgb[overlap]).abs().mean())
            if metric.any():
                depth_error = float(
                    torch.quantile((zbuf[0, ..., 0][metric] - depth[metric]).abs(), 0.9)
                )
            rows.append(
                dict(
                    frame_id=frame_id,
                    link=name,
                    position=float(q),
                    mask_iou=float(overlap.sum() / union.sum()) if union.any() else None,
                    rgb_mae=rgb_error,
                    depth_p90_m=depth_error,
                    overlap_pixels=int(overlap.sum()),
                    metric_pixels=int(metric.sum()),
                )
            )
    return rows
