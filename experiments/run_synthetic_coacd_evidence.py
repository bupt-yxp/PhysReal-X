#!/usr/bin/env python3
"""Execute the installed CoACD worker on a deterministic synthetic concave mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physrealx import PhysRealX

_SOURCE_OBJ = """\
# Deterministic synthetic L-shaped prism in metres.
v 0.00 0.00 0.00
v 0.40 0.00 0.00
v 0.40 0.12 0.00
v 0.12 0.12 0.00
v 0.12 0.40 0.00
v 0.00 0.40 0.00
v 0.00 0.00 0.10
v 0.40 0.00 0.10
v 0.40 0.12 0.10
v 0.12 0.12 0.10
v 0.12 0.40 0.10
v 0.00 0.40 0.10
f 1 3 2
f 1 4 3
f 1 5 4
f 1 6 5
f 7 8 9
f 7 9 10
f 7 10 11
f 7 11 12
f 1 2 8
f 1 8 7
f 2 3 9
f 2 9 8
f 3 4 10
f 3 10 9
f 4 5 11
f 4 11 10
f 5 6 12
f 5 12 11
f 6 1 7
f 6 7 12
""".encode("ascii")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    arguments = _arguments()
    workspace = arguments.workspace.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or next(output.iterdir(), None) is not None):
        raise FileExistsError(f"output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    platform = PhysRealX.open(workspace)
    source = platform.artifacts.put_bytes(_SOURCE_OBJ, media_type="model/obj")
    operation = platform.assets.build_collision(
        source,
        method="coacd",
        config={
            "source_unit": "m",
            "link": "synthetic_l_prism",
            "threshold": 0.05,
            "max_convex_hull": 8,
            "preprocess_resolution": 30,
            "resolution": 2_000,
            "mcts_nodes": 20,
            "mcts_iterations": 80,
            "mcts_max_depth": 3,
            "seed": 0,
        },
    )
    (output / "source_visual_mesh.obj").write_bytes(_SOURCE_OBJ)
    destinations = {
        "collision_geometry": output / "collision.glb",
        "collision_build_report": output / "collision_build_report.json",
    }
    for item in operation.outputs:
        platform.artifacts.materialize(item.artifact, destinations[item.name])
    _write_json(output / "operation_manifest.json", operation.model_dump(mode="json"))
    print(json.dumps(dict(operation.metrics), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
