"""Small shared helpers for independently installed worker processes."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from physrealx.external_workers import (
    WORKER_PROTOCOL_VERSION,
    WorkerInputFile,
    WorkerProbeCheckpoint,
    WorkerProbeEnvironment,
    WorkerProbeRecord,
    WorkerProbeSelfTest,
    WorkerRequest,
    WorkerResourceArtifact,
)


def input_file(request: WorkerRequest, name: str) -> WorkerInputFile:
    """Return one declared input by semantic name."""

    item = next((candidate for candidate in request.inputs if candidate.name == name), None)
    if item is None:
        raise ValueError(f"required worker input is missing: {name}")
    return item


def _materialized_path(relative_path: str) -> Path:
    root = Path.cwd().resolve()
    raw_path = root / relative_path
    if raw_path.is_symlink():
        raise ValueError(f"materialized worker file cannot be a symbolic link: {relative_path}")
    path = raw_path.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"materialized worker file is unsafe: {relative_path}")
    return path


def declared_input_path(request: WorkerRequest, name: str) -> Path:
    """Return a verified materialized input path by semantic name."""

    item = input_file(request, name)
    return _materialized_path(item.path)


def resource_json(request: WorkerRequest, owner: str) -> dict[str, Any]:
    """Load one verified resource manifest without accepting duplicate JSON keys."""

    item = input_file(request, owner)
    if item.kind != "resource":
        raise ValueError(f"worker input is not a resource manifest: {owner}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, member in pairs:
            if key in value:
                raise ValueError(f"resource manifest contains duplicate JSON key: {key!r}")
            value[key] = member
        return value

    document = json.loads(
        declared_input_path(request, owner).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(document, dict):
        raise ValueError(f"resource manifest must be a JSON object: {owner}")
    return document


def dependency_file(
    request: WorkerRequest,
    owner: str,
    digest: str,
) -> WorkerResourceArtifact:
    """Resolve one resource dependency by owning input and immutable digest."""

    matches = tuple(
        item for item in request.resource_artifacts if item.owner == owner and item.digest == digest
    )
    if len(matches) != 1:
        raise ValueError(
            f"resource dependency must resolve exactly once: owner={owner!r}, digest={digest!r}"
        )
    return matches[0]


def dependency_path(request: WorkerRequest, owner: str, digest: str) -> Path:
    """Return a verified dependency path from a resource's ``ArtifactRef`` digest."""

    return _materialized_path(dependency_file(request, owner, digest).path)


def config(request: WorkerRequest) -> dict[str, Any]:
    """Copy the JSON-compatible method configuration."""

    return dict(request.operation.method.config)


def _executable_digest() -> str:
    executable = Path(sys.executable).resolve(strict=True)
    digest = hashlib.sha256()
    with executable.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def report_probe(
    *,
    capability: str,
    name: str,
    version: str,
    source_revision: str,
    dependencies: dict[str, str],
    self_test_checks: dict[str, Any],
    ready: bool = True,
    reason: str | None = None,
    checkpoint_digest: str | None = None,
    checkpoint_source_revision: str | None = None,
    details: dict[str, Any] | None = None,
    failure_exit_code: int = 1,
) -> int:
    """Emit one strict ``WorkerProbe/1`` identity and readiness record."""

    if (checkpoint_digest is None) != (checkpoint_source_revision is None):
        raise ValueError("checkpoint digest and source revision must be supplied together")
    checkpoint = None
    if checkpoint_digest is not None and checkpoint_source_revision is not None:
        checkpoint = WorkerProbeCheckpoint(
            digest=checkpoint_digest,
            source_revision=checkpoint_source_revision,
        )
    record = WorkerProbeRecord(
        protocol_version=WORKER_PROTOCOL_VERSION,
        capability=capability,
        name=name,
        version=version,
        source_revision=source_revision,
        ready=ready,
        self_test=WorkerProbeSelfTest(
            status="passed" if ready else "failed",
            checks=self_test_checks,
        ),
        checkpoint=checkpoint,
        environment=WorkerProbeEnvironment(
            python=platform.python_version(),
            executable_digest=_executable_digest(),
            dependencies=dependencies,
        ),
        reason=reason,
        details=details or {},
    )
    print(record.model_dump_json(by_alias=True))
    return 0 if ready else failure_exit_code


def is_probe(argv: list[str] | None = None) -> bool:
    return "--probe" in (sys.argv[1:] if argv is None else argv)
