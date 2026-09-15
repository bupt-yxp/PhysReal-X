"""PhysReal-X local DTArt runner; no reference geometry or online experiment tracking."""

import argparse
import json
from pathlib import Path


def correspondences(data, weights, top_k):
    import cv2
    import numpy as np
    import torch
    import yaml
    from kornia.feature import LoFTR

    matcher = LoFTR(pretrained=None).eval().cuda()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    matcher.load_state_dict(state["state_dict"])
    frames = yaml.safe_load((data / "init_keyframes.yml").read_text())
    names = [name.removeprefix("frame_") for name in frames]
    positions = {
        name: np.asarray(frames[f"frame_{name}"]["cam_in_ob"])[:3, 3] for name in names
    }
    folder = data / "correspondence_loftr" / "no_filter"
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    matched_views = 0
    for i, name in enumerate(names):
        image0 = cv2.imread(
            str(data / "color_segmented" / f"{name}.png"), cv2.IMREAD_GRAYSCALE
        )
        mask0 = cv2.imread(str(data / "mask" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        targets = sorted(
            (n for n in names if n.split("_")[0] != name.split("_")[0]),
            key=lambda n: float(np.linalg.norm(positions[n] - positions[name])),
        )[:top_k]
        matches = []
        for target in targets:
            image1 = cv2.imread(
                str(data / "color_segmented" / f"{target}.png"), cv2.IMREAD_GRAYSCALE
            )
            mask1 = cv2.imread(
                str(data / "mask" / f"{target}.png"), cv2.IMREAD_GRAYSCALE
            )
            with torch.inference_mode():
                result = matcher(
                    {
                        "image0": torch.from_numpy(image0.copy())
                        .cuda()
                        .float()[None, None]
                        / 255,
                        "image1": torch.from_numpy(image1.copy())
                        .cuda()
                        .float()[None, None]
                        / 255,
                    }
                )
            points = [
                result[k].cpu().numpy().round().astype(np.int64)
                for k in ("keypoints0", "keypoints1")
            ]
            valid = np.ones(len(points[0]), dtype=bool)
            for pts, mask in zip(points, (mask0, mask1)):
                h, w = mask.shape
                inside = (
                    (pts[:, 0] >= 0)
                    & (pts[:, 0] < w)
                    & (pts[:, 1] >= 0)
                    & (pts[:, 1] < h)
                )
                keep = inside.copy()
                keep[inside] &= mask[pts[inside, 1], pts[inside, 0]] > 0
                valid &= keep
            if valid.any():
                matches.append({name: points[0][valid], target: points[1][valid]})
                count += int(valid.sum())
        if matches:
            matched_views += 1
            np.savez_compressed(folder / f"{name}.npz", data=matches)
        print(f"LoFTR {i + 1}/{len(names)}; cumulative matches {count}", flush=True)
    if matched_views < 4 or count < 100:
        raise RuntimeError("Insufficient cross-state object correspondences for DTArt")
    del matcher
    torch.cuda.empty_cache()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import yaml
    from main import run

    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    count = correspondences(args.data, args.weights, config["correspondence_neighbors"])
    cfg = yaml.safe_load(
        (Path(__file__).parent / "config/release/config.yml").read_text()
    )
    cfg.update(
        n_step=config["iterations"],
        random_seed=config["seed"],
        i_mesh=config["iterations"],
        i_weights=config["iterations"],
        use_gt_joint_type=False,
        use_gt_transform=False,
    )
    cfg_dir = args.output / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yml").write_text(yaml.safe_dump(cfg))
    run(
        str(cfg_dir),
        str(args.data),
        str(args.output),
        num_parts=len(config["joint_types"]) + 1,
        no_wandb=True,
        denoise=True,
    )
    (args.output / "matching.json").write_text(json.dumps({"match_count": count}))


if __name__ == "__main__":
    main()
