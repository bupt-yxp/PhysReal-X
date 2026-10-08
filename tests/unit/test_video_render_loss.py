"""RGB, mask, and depth must provide real geometry gradients."""

from types import SimpleNamespace

import numpy as np
import torch
import trimesh


def test_real_rgb_and_mask_renderer_supply_distinct_geometry_gradients():
    from physrealx.integrations._video_render_loss import VideoRenderLoss

    body = trimesh.Trimesh(
        vertices=[[-0.02, -0.02, 1.0], [0.02, -0.02, 1.0], [0.0, 0.02, 1.0]],
        faces=[[0, 1, 2]],
        vertex_colors=[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
        process=False,
    )
    cover = body.copy()
    cover.apply_translation([0.2, 0, 0])
    meshes = {"body": body, "cover": cover}
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    mask = np.zeros((64, 64), dtype=bool)
    mask[22:43, 20:44] = True
    frame = SimpleNamespace(
        position=0.0, image=(rgb, np.zeros((64, 64)), mask, np.eye(4), [500.0, 500.0, 32.0, 32.0])
    )
    renderer = VideoRenderLoss(
        meshes,
        {"body": {"fit": [frame]}, "cover": {"fit": []}},
        "cpu",
        {"parent": "body", "child": "cover", "range": [0.0, 0.7]},
    )
    gradients = []
    for term in ("rgb", "mask"):
        vertices = {
            n: torch.tensor(np.array(m.vertices), requires_grad=True) for n, m in meshes.items()
        }
        loss = renderer.render_terms(vertices, renderer.frames[0], torch.ones(3))[term]
        loss.backward()
        assert vertices["body"].grad is not None
        assert vertices["body"].grad.norm() > 0
        assert torch.isfinite(vertices["body"].grad).all()
        gradients.append(vertices["body"].grad.clone())
    assert not torch.allclose(*gradients)
    vertices = {
        n: torch.tensor(np.array(m.vertices), requires_grad=True) for n, m in meshes.items()
    }
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    origin = torch.zeros(3, dtype=torch.float64)

    def rotate(points, q, hinge_axis, hinge_origin):
        return points  # Both test observations are at the same fixed pose.

    first = renderer.observed_loss(vertices, rotate, axis, origin, "cover", 0.01, 0.01)
    changed = list(renderer.frames[0])
    changed[2] = torch.zeros_like(changed[2])
    renderer.frames.append(tuple(changed))
    combined = renderer.observed_loss(vertices, rotate, axis, origin, "cover", 0.01, 0.01)
    assert not torch.allclose(first, combined)
    combined.backward()
    assert vertices["body"].grad is not None
    assert torch.isfinite(vertices["body"].grad).all()
    assert vertices["body"].grad.norm() > 0
