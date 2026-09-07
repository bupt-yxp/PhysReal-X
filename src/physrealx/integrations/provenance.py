"""Record actual source bytes and installed versions for reproducible learning outputs."""

from __future__ import annotations

import io
import platform
import sys
import zipfile
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import canonical_record_id
from physrealx.integrations.learning_sources import algorithm_module
from physrealx.training_lifecycle import CodePin, EnvironmentPin


def capture_runtime(
    store: ArtifactStore, *, algorithms: tuple[str, ...] = ()
) -> tuple[CodePin, EnvironmentPin]:
    roots = {"physrealx": Path(__file__).resolve().parents[1]}
    for name in algorithms:
        module = algorithm_module(name, package=name == "dinov2")
        if module.__file__ is None:
            raise ValueError("algorithm source has no materialized origin")
        roots[f"algorithms/{name}"] = Path(module.__file__).parent
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, root in roots.items():
            for path in sorted(root.rglob("*")):
                if path.is_file() and (
                    path.suffix in {".py", ".json", ".toml"} or path.name.startswith("LICENSE")
                ):
                    relative = path.relative_to(root)
                    if any(part in {"__pycache__", "_embedded"} for part in relative.parts):
                        continue
                    info = zipfile.ZipInfo(
                        f"{label}/{relative.as_posix()}", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
    source = store.put_bytes(buffer.getvalue(), media_type="application/zip")
    code = CodePin(
        source_uri="https://github.com/bupt-yxp/PhysReal-X",
        revision=source.artifact_id,
        source_tree=source,
    )
    lock: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": sorted(
            (item.metadata["Name"] or "unknown", item.version) for item in distributions()
        ),
    }
    environment = EnvironmentPin(
        runtime=f"Python {platform.python_version()}",
        lockfile=store.put_json(lock),
        environment_record_id=canonical_record_id(lock),
    )
    return code, environment
