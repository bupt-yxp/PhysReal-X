"""Calibrate the reviewed laptop video against module-one start/end RGB-D cameras.

Run with the PhysReal-X environment. Outputs are calibration candidates until
held-out checks pass; this command does not manufacture physical observations.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def reconstruct(root):
    import cv2
    import numpy as np

    video = root / "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1/real/video"
    output = root / "examples/physical_asset_building/output/video-colmap/masked"
    for name in ("images", "masks", "sparse"):
        (output / name).mkdir(parents=True, exist_ok=True)
    frames = sorted((video / "img").glob("*.png"), key=lambda p: int(p.stem))
    if not frames:
        raise ValueError("video has no RGB frames")
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.read_bytes())
        digest.update((video / "link_mask" / frame.name).read_bytes())
    identity = digest.hexdigest()
    receipt = output / "input.json"
    if receipt.exists() and json.loads(receipt.read_text())["sha256"] != identity:
        raise ValueError("video or masks changed: use a fresh calibration output directory")
    for frame in frames:
        name = f"{int(frame.stem):06d}.png"
        target = output / "images" / name
        if not target.exists():
            # COLMAP resolves symlinks and changes names; hardlinks preserve mask lookup.
            os.link(frame, target)
        labels = cv2.imread(str(video / "link_mask" / frame.name), 0)
        blocked = cv2.dilate((labels == 2).astype(np.uint8), np.ones((21, 21), np.uint8))
        cv2.imwrite(str(output / "masks" / (name + ".png")), (1 - blocked) * 255)
    receipt.write_text(json.dumps({"sha256": identity, "frames": len(frames)}))
    camera_file = root / "examples/real_assets_and_digital_twin/output/registered-cameras.json"
    rig = json.loads(camera_file.read_text())
    params = rig["streams"][0]["components"][0]["cameras"][0]["params"]
    commands = [
        [
            "feature_extractor",
            "--image_path",
            str(output / "images"),
            "--ImageReader.mask_path",
            str(output / "masks"),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            ",".join(map(str, params)),
            "--FeatureExtraction.use_gpu",
            "0",
            "--FeatureExtraction.num_threads",
            "12",
        ],
        [
            "exhaustive_matcher",
            "--FeatureMatching.use_gpu",
            "1",
            "--FeatureMatching.gpu_index",
            "0,1",
            "--FeatureMatching.num_threads",
            "12",
        ],
        [
            "mapper",
            "--image_path",
            str(output / "images"),
            "--output_path",
            str(output / "sparse"),
            "--Mapper.num_threads",
            "12",
            "--Mapper.random_seed",
            "2026",
            "--Mapper.min_focal_length_ratio",
            "0.5",
            "--Mapper.max_focal_length_ratio",
            "1.5",
            "--Mapper.max_extra_param",
            "0.5",
        ],
    ]
    for command in commands:
        with (output / (command[0] + ".log")).open("w") as log:
            subprocess.run(
                ["colmap", *command, "--database_path", str(output / "database.db")],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def localize(root):
    import json
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    base = root / "examples/real_assets_and_digital_twin"
    video = base / "input/ArtModeling/laptop1/real/video"
    out = root / "examples/physical_asset_building/output/video-colmap"
    rig = json.loads((base / "output/registered-cameras.json").read_text())

    def reference(item):
        stream, component, im = item
        state = stream["stream_name"].split(".")[1]
        path = base / "input/ArtModeling/laptop1/real/can" / state
        idx = im["source_frame_index"]
        gray = cv2.imread(str(path / "img" / f"{idx}.png"), 0)
        mask = cv2.imread(str(path / "mask" / f"{idx}.png"), 0)
        depth = np.load(path / "depth" / f"{idx}.npy")
        k, d = cv2.SIFT_create(nfeatures=6000).detectAndCompute(gray, mask)
        if d is None:
            return None
        xy = np.array([a.pt for a in k])
        uv = np.rint(xy).astype(int)
        z = (
            depth[uv[:, 1], uv[:, 0]]
            * rig["depth_calibrations"][stream["stream_name"]]["depth_scale_m"]
        )
        cam = next(c for c in component["cameras"] if c["camera_id"] == im["camera_id"])
        f, cx, cy, rad = cam["params"]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        rays = cv2.undistortPoints(xy[:, None, :], K, np.array([rad, 0, 0, 0.0]))[:, 0, :]
        xyz = np.column_stack([rays, np.ones(len(rays))]) * z[:, None]
        T = np.array(im["cam_from_world"])
        world = (xyz - T[:, 3]) @ T[:, :3]
        valid = np.isfinite(world).all(axis=1) & (z > 0)
        return {"state": state, "frame": idx, "d": d[valid], "xyz": world[valid]}

    items = []
    for stream in rig["streams"]:
        for c in stream["components"]:
            images = c["images"]
            items.extend((stream, c, im) for im in images[::8])
    with ThreadPoolExecutor(max_workers=10) as pool:
        refs = [x for x in pool.map(reference, items) if x is not None]
    print("references", len(refs), flush=True)
    K = np.array([[955.0, 0, 640.0], [0, 955.0, 360.0], [0, 0, 1.0]])
    # Use actual video COLMAP intrinsics once the reconstruction exists.
    models = sorted((out / "masked/sparse").glob("*/cameras.bin"))
    if models:
        import pycolmap

        rec = max(
            (pycolmap.Reconstruction(str(p.parent)) for p in models),
            key=lambda r: r.num_reg_images(),
        )
        cam = next(iter(rec.cameras.values()))
        K = cam.calibration_matrix()
        params = cam.params
        dist = (
            np.array([params[3], 0, 0, 0.0]) if cam.model_name == "SIMPLE_RADIAL" else np.zeros(4)
        )
    else:
        dist = np.zeros(4)

    def localize(idx):
        gray = cv2.imread(str(video / "img" / f"{idx}.png"), 0)
        labels = cv2.imread(str(video / "link_mask" / f"{idx}.png"), 0)
        mask = cv2.erode((labels == 1).astype("uint8") * 255, np.ones((5, 5), np.uint8))
        keys, desc = cv2.SIFT_create(nfeatures=6000).detectAndCompute(gray, mask)
        if desc is None:
            return {"frame": idx, "status": "NO_FEATURES"}
        xy = np.array([k.pt for k in keys])
        candidates = []
        matcher = cv2.BFMatcher()
        for r in refs:
            matches = matcher.knnMatch(desc, r["d"], k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < 10:
                continue
            obj = np.array([r["xyz"][m.trainIdx] for m in good])
            pix = np.array([xy[m.queryIdx] for m in good])
            ok, rv, tv, ins = cv2.solvePnPRansac(
                obj,
                pix,
                K,
                dist,
                iterationsCount=300,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or ins is None or len(ins) < 10:
                continue
            ins = ins[:, 0]
            rv, tv = cv2.solvePnPRefineLM(obj[ins], pix[ins], K, dist, rv, tv)
            proj = cv2.projectPoints(obj[ins], rv, tv, K, dist)[0][:, 0]
            error = np.linalg.norm(proj - pix[ins], axis=1)
            R = cv2.Rodrigues(rv)[0]
            candidates.append(
                {
                    "state": r["state"],
                    "reference_frame": r["frame"],
                    "inliers": len(ins),
                    "matches": len(good),
                    "rmse_px": float(np.sqrt(np.mean(error**2))),
                    "cam_from_reference": np.column_stack([R, tv]).tolist(),
                    "image_points": pix[ins].tolist(),
                    "reference_points": obj[ins].tolist(),
                }
            )
        candidates.sort(key=lambda c: (-c["inliers"], c["rmse_px"]))
        result = {
            "frame": idx,
            "status": "LOCALIZED" if candidates else "FAILED",
            "candidates": candidates[:6],
        }
        print(
            idx,
            result["status"],
            [(c["state"], c["inliers"], round(c["rmse_px"], 2)) for c in candidates[:2]],
            flush=True,
        )
        return result

    available = sorted(int(p.stem) for p in (video / "img").glob("*.png"))
    ids = sorted(set([*available[::4], available[-1]]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(localize, ids))
    (out / "reference-localization.json").write_text(
        json.dumps(
            {
                "method": "static-body SIFT RGB-D PnP against registered start/end cameras",
                "video_intrinsics": K.tolist(),
                "video_distortion": dist.tolist(),
                "video_intrinsics_source": "colmap" if models else "initial_guess_only",
                "frames": results,
            },
            indent=2,
        )
    )


def align(root):
    import json
    from pathlib import Path

    import cv2
    import numpy as np
    import pycolmap
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    out = root / "examples/physical_asset_building/output/video-colmap"
    models = list((out / "masked/sparse").glob("*/cameras.bin"))
    rec = max(
        (pycolmap.Reconstruction(str(p.parent)) for p in models), key=lambda r: r.num_reg_images()
    )
    imgs = {int(Path(i.name).stem): i for i in rec.images.values()}
    local = json.loads((out / "reference-localization.json").read_text())
    cam = next(iter(rec.cameras.values()))
    K = cam.calibration_matrix()
    dist = np.array([cam.params[3], 0, 0, 0.0])
    video = root / "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1/real/video"
    ratios = []
    for idx, img in imgs.items():
        depth = np.load(video / "depth" / f"{idx}.npy")
        T = img.cam_from_world().matrix()
        for p in img.points2D:
            if not p.has_point3D():
                continue
            x, y = np.rint(p.xy).astype(int)
            if not (0 <= x < 1280 and 0 <= y < 720) or depth[y, x] <= 0:
                continue
            z = (T[:, :3] @ rec.points3D[p.point3D_id].xyz + T[:, 3])[2]
            if z > 0:
                ratios.append(float(depth[y, x]) * 0.001 / z)
    scale = float(np.median(ratios))
    anchors = []
    for frame in local["frames"]:
        if frame["frame"] not in imgs or not frame.get("candidates"):
            continue
        c = frame["candidates"][0]
        anchors.append(
            {
                "frame": frame["frame"],
                "T": imgs[frame["frame"]].cam_from_world().matrix(),
                "xyz": np.array(c["reference_points"]),
                "uv": np.array(c["image_points"]),
                "pnp": np.array(c["cam_from_reference"]),
                "state": c["state"],
            }
        )
    print("images", len(imgs), "anchors", len(anchors), "scale_initial", scale, flush=True)

    def residual(x, anchors):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        t = x[3:6]
        s = np.exp(x[6])
        pieces = []
        for a in anchors:
            T = a["T"]
            cr = T[:, :3] @ R.T
            ct = s * T[:, 3] - cr @ t
            xyz = a["xyz"] @ cr.T + ct
            proj = cv2.projectPoints(a["xyz"], cv2.Rodrigues(cr)[0], ct, K, dist)[0][:, 0, :]
            r = proj - a["uv"]
            r[xyz[:, 2] <= 0] = 1000
            pieces.append(r.ravel())
        return np.concatenate(pieces)

    fit_anchors = []
    check_anchors = []
    for state in ("start", "end"):
        group = [a for a in anchors if a["state"] == state]
        fit_anchors.extend(group[::2])
        check_anchors.extend(group[1::2])
    if len(fit_anchors) < 3 or len(check_anchors) < 3:
        raise ValueError("alignment requires at least three fit and three check frames")
    fits = []
    for a in fit_anchors:
        R = a["pnp"][:, :3].T @ a["T"][:, :3]
        cref = -a["pnp"][:, :3].T @ a["pnp"][:, 3]
        ccol = -a["T"][:, :3].T @ a["T"][:, 3]
        x = np.r_[Rotation.from_matrix(R).as_rotvec(), cref - scale * R @ ccol, np.log(scale)]
        fit = least_squares(
            residual, x, args=(fit_anchors,), loss="soft_l1", f_scale=3, max_nfev=150
        )
        errors = [
            float(np.median(np.linalg.norm(residual(fit.x, [a]).reshape(-1, 2), axis=1)))
            for a in fit_anchors
        ]
        fits.append((float(np.median(errors)), fit.x, errors))
    score, x, _ = min(fits, key=lambda z: z[0])
    errors = [
        float(np.median(np.linalg.norm(residual(x, [a]).reshape(-1, 2), axis=1))) for a in anchors
    ]
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    t = x[3:6]
    s = float(np.exp(x[6]))
    print("fit", score, s, [(a["frame"], e) for a, e in zip(anchors, errors)], flush=True)
    views = []
    for idx, img in sorted(imgs.items()):
        T = img.cam_from_world().matrix()
        cr = T[:, :3] @ R.T
        ct = s * T[:, 3] - cr @ t
        views.append(
            {
                "frame": idx,
                "cam_from_reference": np.vstack([np.column_stack([cr, ct]), [0, 0, 0, 1]]).tolist(),
            }
        )
    (out / "alignment-candidate.json").write_text(
        json.dumps(
            {
                "status": "DIAGNOSTIC_ONLY",
                "scale": s,
                "rotation": R.tolist(),
                "translation_m": t.tolist(),
                "median_anchor_error_px": score,
                "anchors": [
                    {
                        "frame": a["frame"],
                        "state": a["state"],
                        "median_px": e,
                        "points": len(a["xyz"]),
                    }
                    for a, e in zip(anchors, errors)
                ],
                "camera": {"model": cam.model_name, "params": cam.params.tolist()},
                "frames": views,
                "fit_frames": [a["frame"] for a in fit_anchors],
                "check_frames": [a["frame"] for a in check_anchors],
                "validation": (
                    "State-stratified frames held out from Sim3 fitting; "
                    "acceptance pending depth and start/end consistency."
                ),
            },
            indent=2,
        )
    )


def audit(root):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import cv2
    import numpy as np
    import pycolmap
    from scipy.optimize import minimize_scalar
    from scipy.spatial import cKDTree

    from physrealx import ArtifactRef, PhysRealX
    from physrealx.integrations.collision import load_meshes
    from physrealx.reconstruction import joint_transform

    out = root / "examples/physical_asset_building/output/video-colmap"
    v = root / "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1/real/video"
    a = json.loads((out / "alignment-candidate.json").read_text())
    s = a["scale"]
    models = list((out / "masked/sparse").glob("*/cameras.bin"))
    rec = max(
        (pycolmap.Reconstruction(str(p.parent)) for p in models), key=lambda r: r.num_reg_images()
    )
    ratios = {}
    for img in rec.images.values():
        idx = int(Path(img.name).stem)
        d = np.load(v / "depth" / f"{idx}.npy")
        mask = cv2.imread(str(v / "link_mask" / f"{idx}.png"), 0)
        T = img.cam_from_world().matrix()
        rr = []
        for p in img.points2D:
            if not p.has_point3D():
                continue
            x, y = np.rint(p.xy).astype(int)
            if not (1 <= x < 1279 and 1 <= y < 719) or d[y, x] <= 0 or mask[y, x] != 1:
                continue
            z = (T[:, :3] @ rec.points3D[p.point3D_id].xyz + T[:, 3])[2] * s
            if z > 0 and np.ptp(d[y - 1 : y + 2, x - 1 : x + 2]) < 30:
                rr.append((float(d[y, x]), float(z)))
        ratios[idx] = rr
    fit = np.array([p for k, ps in ratios.items() if k % 2 == 0 for p in ps])
    check = np.array([p for k, ps in ratios.items() if k % 2 == 1 for p in ps])
    scale = float(np.median(fit[:, 1] / fit[:, 0]))
    error = np.abs(check[:, 0] * scale - check[:, 1])
    print("depth scale", scale, "check median/p90", np.quantile(error, [0.5, 0.9]), flush=True)
    px = PhysRealX.open(root / ".physrealx")
    g = json.loads((out.parent / "registered-reconstruction-geometry.json").read_text())
    mesh = load_meshes(
        px.artifacts, ArtifactRef.model_validate(g["geometries"]["cover"]), process=False
    )[0]
    tree = cKDTree(np.concatenate([mesh.vertices, mesh.triangles_center]))
    motion = g["reconstructed_motion"]
    lower, upper = motion["range"]
    f, cx, cy, rad = a["camera"]["params"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

    def measure(view):
        idx = view["frame"]
        d = np.load(v / "depth" / f"{idx}.npy")
        mask = cv2.imread(str(v / "link_mask" / f"{idx}.png"), 0)
        keep = cv2.erode((mask == 2).astype("uint8"), np.ones((9, 9), np.uint8)).astype(bool) & (
            d > 0
        )
        y, x = np.nonzero(keep)
        sel = np.linspace(0, len(x) - 1, min(800, len(x)), dtype=int)
        x = x[sel]
        y = y[sel]
        z = d[y, x] * scale
        rays = cv2.undistortPoints(
            np.column_stack([x, y]).astype(float)[:, None, :], K, np.array([rad, 0, 0, 0.0])
        )[:, 0, :]
        points = np.column_stack([rays, np.ones(len(rays))]) * z[:, None]
        T = np.array(view["cam_from_reference"])
        points = (points - T[:3, 3]) @ T[:3, :3]

        def err(q):
            M = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            local = (points - M[:3, 3]) @ M[:3, :3]
            dd = tree.query(local)[0]
            return float(np.median(dd))

        grid = np.linspace(lower - 0.5, upper + 0.5, 91)
        cost = np.array([err(q) for q in grid])
        j = int(np.argmin(cost))
        fitted = minimize_scalar(
            err, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]), method="bounded"
        )
        q = float(fitted.x)
        return {
            "frame": idx,
            "position": q,
            "surface_median_m": float(fitted.fun),
            "within_source_range": bool(lower <= q <= upper),
            "support": len(points),
        }

    with ThreadPoolExecutor(max_workers=10) as pool:
        frames = list(pool.map(measure, a["frames"]))
    report = {
        "depth_scale_m": scale,
        "depth_check_median_m": float(np.median(error)),
        "depth_check_p90_m": float(np.quantile(error, 0.9)),
        "source_range": motion["range"],
        "frames": frames,
        "note": (
            "Motion estimates are measured without clamping; "
            "out-of-range frames cannot silently change the source range."
        ),
    }
    (out / "depth-motion-audit.json").write_text(json.dumps(report, indent=2))
    print(
        "q range",
        min(x["position"] for x in frames),
        max(x["position"] for x in frames),
        "valid",
        sum(x["within_source_range"] for x in frames),
        "error p50p90",
        np.quantile([x["surface_median_m"] for x in frames], [0.5, 0.9]),
        flush=True,
    )


def preview(root):
    import json

    import cv2
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from physrealx import ArtifactRef, PhysRealX
    from physrealx.integrations.collision import load_meshes

    out = root / "examples/physical_asset_building/output/video-colmap"
    a = json.loads((out / "alignment-candidate.json").read_text())
    local = json.loads((out / "reference-localization.json").read_text())
    px = PhysRealX.open(root / ".physrealx")
    geo = json.loads((out.parent / "registered-reconstruction-geometry.json").read_text())
    mesh = load_meshes(
        px.artifacts, ArtifactRef.model_validate(geo["geometries"]["body"]), process=False
    )[0]
    f, cx, cy, k = a["camera"]["params"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for ax, idx in zip(axes.flat, [16, 64, 72, 160]):
        p = (
            root
            / "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1/real/video/img"
            / f"{idx}.png"
        )
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        T = np.array(next(x for x in a["frames"] if x["frame"] == idx)["cam_from_reference"])
        uv = cv2.projectPoints(
            mesh.vertices, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, np.array([k, 0, 0, 0.0])
        )[0][:, 0]
        ax.imshow(rgb)
        ax.scatter(uv[::4, 0], uv[::4, 1], s=0.6, c="#00ffff", alpha=0.35)
        c = next(x for x in local["frames"] if x["frame"] == idx)["candidates"][0]
        pix = np.array(c["image_points"])
        proj = cv2.projectPoints(
            np.array(c["reference_points"]),
            cv2.Rodrigues(T[:3, :3])[0],
            T[:3, 3],
            K,
            np.array([k, 0, 0, 0.0]),
        )[0][:, 0]
        ax.scatter(pix[:, 0], pix[:, 1], s=10, c="yellow")
        ax.scatter(proj[:, 0], proj[:, 1], s=10, c="red", marker="+")
        ax.set(
            xlim=(0, 1280),
            ylim=(720, 0),
            title=f"Frame {idx}: cyan = module-one body; yellow/red = matched/projected",
        )
        ax.axis("off")
    fig.savefig(out / "alignment-preview.png", dpi=130)
    plt.close(fig)


def export(root):
    import io
    import json

    import cv2
    import numpy as np

    from physrealx import ArtifactRef, PhysRealX
    from physrealx.integrations.video_articulation import VideoArticulationEvidence

    out = root / "examples/physical_asset_building/output"
    work = out / "video-colmap"
    v = root / "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1/real/video"
    a = json.loads((work / "alignment-candidate.json").read_text())
    audit = json.loads((work / "depth-motion-audit.json").read_text())
    up = json.loads(
        (root / "examples/real_assets_and_digital_twin/output/results.json").read_text()
    )
    prior = json.loads((out / "vlm-physics-prior.json").read_text())
    geometry = json.loads((out / "registered-reconstruction-geometry.json").read_text())
    px = PhysRealX.open(root / ".physrealx")
    checks = [x["median_px"] for x in a["anchors"] if x["frame"] in a["check_frames"]]
    if (
        len(checks) < 3
        or not np.isfinite(checks).all()
        or np.quantile(checks, 0.9) > 5
        or max(checks) > 8
    ):
        raise ValueError("held-out camera alignment exceeds 5px p90 / 8px maximum")
    if not np.isfinite(audit["depth_scale_m"]) or audit["depth_scale_m"] <= 0:
        raise ValueError("invalid metric depth scale")
    # Preserve all frame diagnostics; only observations consistent with the source
    # articulation interval are eligible, never clamp measured angles into it.
    valid = [
        f for f in audit["frames"] if f["within_source_range"] and f["surface_median_m"] <= 0.02
    ]
    if len(valid) < 4:
        raise ValueError("Too few source-range observations")
    selected = [
        valid[i] for i in np.unique(np.linspace(0, len(valid) - 1, min(24, len(valid)), dtype=int))
    ]
    f, cx, cy, k = a["camera"]["params"]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, np.array([k, 0, 0, 0.0]), None, K, (1280, 720), cv2.CV_32FC1
    )
    views = {v["frame"]: v for v in a["frames"]}
    frames = []
    for index, m in enumerate(selected):
        idx = m["frame"]
        rgb = cv2.imread(str(v / "img" / f"{idx}.png"))
        depth = np.load(v / "depth" / f"{idx}.npy")
        mask = cv2.imread(str(v / "link_mask" / f"{idx}.png"), 0)
        rgb = cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR)
        depth = cv2.remap(depth, mapx, mapy, cv2.INTER_NEAREST)
        mask = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)

        def png(x):
            ok, data = cv2.imencode(".png", x)
            if not ok:
                raise ValueError("PNG encoding failed")
            return px.artifacts.put_bytes(data.tobytes(), media_type="image/png")

        stream = io.BytesIO()
        np.save(stream, depth, allow_pickle=False)
        frames.append(
            {
                "frame_id": str(idx),
                "partition": "fit" if index % 2 == 0 else "check",
                "rgb": png(rgb),
                "depth": px.artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy"),
                "link_mask": png(mask),
                "intrinsics": [f, f, cx, cy],
                "camera_from_reference": views[idx]["cam_from_reference"],
                "positions": {geometry["reconstructed_motion"]["joint"]: m["position"]},
            }
        )
    # Original two-state images constrain shape/joint changes directly. Their
    # masks are foreground masks, never relabelled as invented part supervision.
    base = root / "examples/real_assets_and_digital_twin"
    rig = json.loads((base / "output/registered-cameras.json").read_text())
    import zipfile

    depth_archive = zipfile.ZipFile(
        io.BytesIO(px.artifacts.get_bytes(ArtifactRef.model_validate(up["enhanced-depth.zip"])))
    )
    depth_manifest = json.loads(depth_archive.read("manifest.json"))
    if depth_manifest["depth_encoding"]["unit"] != "m":
        raise ValueError("module-one training depth must be metric")
    training_frames = {
        (frame["view"]["stream_name"], frame["view"]["source_frame_index"]): frame
        for frame in depth_manifest["frames"]
    }
    for stream_info in rig["streams"]:
        state = stream_info["stream_name"].split(".")[1]
        if state not in {"start", "end"}:
            continue
        capture = base / "input/ArtModeling/laptop1/real/can" / state
        component = max(stream_info["components"], key=lambda c: len(c["images"]))
        selected_images = [
            component["images"][int(i)]
            for i in np.unique(np.linspace(0, len(component["images"]) - 1, 6, dtype=int))
        ]
        for index, image_info in enumerate(selected_images):
            camera = next(
                c for c in component["cameras"] if c["camera_id"] == image_info["camera_id"]
            )
            if camera["model"] != "SIMPLE_RADIAL":
                raise ValueError("endpoint export requires the calibrated SIMPLE_RADIAL model")
            focal, cx0, cy0, radial = camera["params"]
            intrinsics = np.array([[focal, 0, cx0], [0, focal, cy0], [0, 0, 1.0]])
            maps = cv2.initUndistortRectifyMap(
                intrinsics,
                np.array([radial, 0, 0, 0.0]),
                None,
                intrinsics,
                (camera["width"], camera["height"]),
                cv2.CV_32FC1,
            )
            frame_index = image_info["source_frame_index"]
            training = training_frames[(stream_info["stream_name"], frame_index)]
            measured = np.load(
                io.BytesIO(depth_archive.read(training["depth"]["path"])), allow_pickle=False
            )
            raw_rgb = cv2.imread(str(capture / "img" / f"{frame_index}.png"))
            height, width = raw_rgb.shape[:2]
            sy, sx = (
                (height + measured.shape[0] - 1) // measured.shape[0],
                (width + measured.shape[1] - 1) // measured.shape[1],
            )
            training_rgb = np.load(
                io.BytesIO(px.artifacts.get_bytes(ArtifactRef.model_validate(training["rgb"]))),
                allow_pickle=False,
            )
            if not np.array_equal(raw_rgb[::sy, ::sx, ::-1], training_rgb):
                raise ValueError("endpoint pixels differ from module-one training input")
            # Reuse the exact accepted training depths at their original sensor
            # pixels. Unobserved pixels remain invalid; no interpolated depth or
            # candidate-dependent outlier rejection enters the endpoint evidence.
            sparse_depth = np.zeros((height, width), dtype=np.float32)
            sparse_depth[::sy, ::sx] = measured
            rgb = cv2.remap(raw_rgb, *maps, cv2.INTER_LINEAR)
            depth = cv2.remap(sparse_depth, *maps, cv2.INTER_NEAREST)
            mask = cv2.remap(
                cv2.imread(str(capture / "mask" / f"{frame_index}.png"), 0),
                *maps,
                cv2.INTER_NEAREST,
            )
            array = io.BytesIO()
            np.save(array, depth, allow_pickle=False)
            matrix = np.eye(4)
            matrix[:3] = image_info["cam_from_world"]
            frames.append(
                {
                    "frame_id": f"can.{state}.{frame_index}",
                    "partition": "fit" if index % 2 == 0 else "check",
                    "rgb": png(rgb),
                    "depth": px.artifacts.put_bytes(
                        array.getvalue(), media_type="application/x-npy"
                    ),
                    "object_mask": png(mask),
                    "depth_scale_m": 1.0,
                    "intrinsics": [focal, focal, cx0, cy0],
                    "camera_from_reference": matrix.tolist(),
                    "positions": {
                        geometry["reconstructed_motion"]["joint"]: geometry["reconstructed_motion"][
                            "positions"
                        ][0 if state == "start" else 1]
                    },
                }
            )
    depth_archive.close()
    evidence = VideoArticulationEvidence(
        reference=ArtifactRef.model_validate(up["reference-model.json"]),
        registration=ArtifactRef.model_validate(up["reference-registration.json"]),
        source_geometry=ArtifactRef.model_validate(prior["source_geometry"]),
        capture_id="laptop1.video.colmap.start-end-aligned",
        depth_scale_m=audit["depth_scale_m"],
        link_ids={"body": 1, "cover": 2},
        frames=frames,
    )
    (out / "video-articulation-evidence.json").write_text(
        evidence.model_dump_json(by_alias=True, indent=2)
    )
    checks = [x["median_px"] for x in a["anchors"] if x["frame"] in a["check_frames"]]
    summary = {
        "schema": "physrealx.video-calibration-diagnostic/1",
        "status": "ESTIMATED_NOT_INDEPENDENT_GROUND_TRUTH",
        "registered_frames": len(a["frames"]),
        "total_frames": len(audit["frames"]),
        "source_states": ["start", "end"],
        "endpoint_depth_source": up["enhanced-depth.zip"],
        "endpoint_depth_policy": "exact_module_one_training_pixels_and_validity",
        "fit_frames": a["fit_frames"],
        "check_frames": a["check_frames"],
        "check_frame_median_error_px": float(np.median(checks)),
        "check_frame_max_error_px": float(max(checks)),
        "depth_scale_m": audit["depth_scale_m"],
        "depth_check_median_m": audit["depth_check_median_m"],
        "depth_check_p90_m": audit["depth_check_p90_m"],
        "range_preserved": audit["source_range"],
        "out_of_range_frames": [
            x["frame"] for x in audit["frames"] if not x["within_source_range"]
        ],
        "selected_frames": [x["frame"] for x in selected],
        "rejected_frame_count": len(audit["frames"]) - len(valid),
        "evidence": "video-articulation-evidence.json",
        "physical_validation": False,
        "limitations": [
            (
                "Depth and reference alignment are model-conditioned estimates, "
                "not independent metric ground truth."
            ),
            "Joint positions are inferred from observed depth against module-one surfaces.",
            (
                "No verified timing or external force is available; "
                "this evidence supports geometry only."
            ),
            "Open source meshes still require valid collision geometry before SDF acceptance.",
        ],
    }
    (out / "video-calibration.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--stage",
        choices=("all", "reconstruct", "localize", "align", "audit", "preview", "export"),
        default="all",
    )
    args = parser.parse_args()
    stages = {
        "reconstruct": reconstruct,
        "localize": localize,
        "align": align,
        "audit": audit,
        "preview": preview,
        "export": export,
    }
    for name, action in stages.items():
        if args.stage in ("all", name):
            action(args.root.resolve())


if __name__ == "__main__":
    main()
