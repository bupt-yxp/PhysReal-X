from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from physrealx.domain import (
    ArtifactRef,
    EpisodeSummary,
    RunResult,
    RunSpec,
    RunStatus,
    VersionedRef,
)
from physrealx.errors import (
    ArtifactIntegrityError,
    DomainValidationError,
    InvalidStateTransitionError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from physrealx.storage import LocalArtifactStore, SQLiteRepository

_RUN_ID = "00000000-0000-0000-0000-000000000001"
_EPISODE_ID = "00000000-0000-0000-0000-000000000002"


def _reference(resource_type: str, resource_id: str, digest: str) -> VersionedRef:
    return VersionedRef(
        resource_type=resource_type,
        resource_id=resource_id,
        version="1.0.0",
        digest=digest,
    )


def _run_spec(digest: str, *, seed: int = 7) -> RunSpec:
    return RunSpec(
        run_id=_RUN_ID,
        asset=_reference("asset", "drawer", digest),
        task=_reference("task", "open_drawer", digest),
        backend="reference",
        policy="scripted",
        episode_count=1,
        base_seed=seed,
        dataset_id="drawer_dataset",
        dataset_version="1.0.0",
        record_signals=("joint.position",),
    )


def _episode(artifact: ArtifactRef) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=_EPISODE_ID,
        run_id=_RUN_ID,
        index=0,
        step_count=3,
        success=True,
        digest=artifact.digest,
        artifact=artifact,
    )


@pytest.mark.integration
def test_artifacts_are_content_addressed_and_json_is_canonical(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    first = store.put_bytes(b"physreal-x", media_type="application/octet-stream")
    duplicate = store.put_bytes(b"physreal-x", media_type="application/octet-stream")

    assert first == duplicate
    assert first.digest == f"sha256:{hashlib.sha256(b'physreal-x').hexdigest()}"
    assert first.size_bytes == len(b"physreal-x")
    assert store.get_bytes(first) == b"physreal-x"
    assert store.verify(first)

    json_a = store.put_json({"z": [1, 2], "a": "\u5177\u8eab"})
    json_b = store.put_json({"a": "\u5177\u8eab", "z": [1, 2]})
    assert json_a.digest == json_b.digest
    assert store.get_bytes(json_a) == b'{"a":"\xe5\x85\xb7\xe8\xba\xab","z":[1,2]}'

    assert not list(store.root.rglob("*.tmp"))


@pytest.mark.integration
def test_artifact_reads_reject_bad_references_and_corruption(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    reference = store.put_bytes(b"immutable")

    with pytest.raises(DomainValidationError):
        store.get_bytes("../not-a-digest")
    with pytest.raises(ResourceNotFoundError):
        store.get_bytes(f"sha256:{'0' * 64}")

    object_path = (
        store.root
        / "objects"
        / "sha256"
        / reference.digest[len("sha256:") :][:2]
        / reference.digest[len("sha256:") :]
    )
    object_path.write_bytes(b"corrupt")

    assert not store.verify(reference)
    with pytest.raises(ArtifactIntegrityError):
        store.get_bytes(reference)


@pytest.mark.integration
def test_repository_enables_durable_sqlite_settings_and_schema(tmp_path: Path) -> None:
    database = tmp_path / "metadata.sqlite3"
    repository = SQLiteRepository(database, busy_timeout_ms=1_234)

    assert repository.schema_version == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "schema_metadata",
            "resources",
            "runs",
            "episodes",
        } <= tables
        assert {"datasets", "models", "evaluations"}.isdisjoint(tables)
        episode_foreign_keys = connection.execute("PRAGMA foreign_key_list(episodes)").fetchall()
        assert any(row[2] == "runs" for row in episode_foreign_keys)

    # These pragmas are connection-local and must be installed on every
    # repository connection, not merely during schema creation.
    with repository._connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1_234


@pytest.mark.integration
def test_repository_rejects_unknown_schema_version(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(RuntimeError, match="unsupported database schema version 2"):
        SQLiteRepository(database)


@pytest.mark.integration
def test_versioned_registration_is_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "metadata.sqlite3")
    reference = _reference("asset", "drawer", f"sha256:{'a' * 64}")

    assert repository.register_resource(reference) == reference
    assert repository.register_resource(reference) == reference
    assert repository.get_resource("asset", "drawer", "1.0.0") == reference
    assert repository.list_resources(resource_type="asset") == [reference]

    conflicting = reference.model_copy(update={"digest": f"sha256:{'b' * 64}"})
    with pytest.raises(ResourceConflictError):
        repository.register_resource(conflicting)
    assert repository.get_resource("asset", "drawer", "1.0.0") == reference

    unresolved = reference.model_copy(update={"resource_id": "unresolved", "digest": None})
    with pytest.raises(DomainValidationError):
        repository.register_resource(unresolved)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("resource_type", "register_name", "get_name", "list_name"),
    [
        ("dataset", "register_dataset", "get_dataset", "list_datasets"),
        ("model", "register_model", "get_model", "list_models"),
        ("evaluation", "register_evaluation", "get_evaluation", "list_evaluations"),
    ],
)
def test_specialized_registries_share_resource_invariants(
    tmp_path: Path,
    resource_type: str,
    register_name: str,
    get_name: str,
    list_name: str,
) -> None:
    repository = SQLiteRepository(tmp_path / f"{resource_type}.sqlite3")
    reference = _reference(resource_type, f"{resource_type}_one", f"sha256:{'c' * 64}")

    register = getattr(repository, register_name)
    get = getattr(repository, get_name)
    list_items = getattr(repository, list_name)

    assert register(reference) == reference
    assert register(reference) == reference
    assert get(reference.resource_id, reference.version) == reference
    assert list_items() == [reference]
    assert (
        repository.get_resource(resource_type, reference.resource_id, reference.version)
        == reference
    )

    wrong_type = reference.model_copy(update={"resource_type": "asset"})
    with pytest.raises(DomainValidationError):
        register(wrong_type)


@pytest.mark.integration
def test_concurrent_identical_resource_registration_publishes_one_row(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "metadata.sqlite3", busy_timeout_ms=10_000)
    reference = _reference("asset", "drawer", f"sha256:{'d' * 64}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(repository.register_resource, [reference] * 16))

    assert results == [reference] * 16
    assert repository.list_resources() == [reference]


@pytest.mark.integration
def test_run_episode_and_result_are_published_through_strict_lifecycle(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    repository = SQLiteRepository(tmp_path / "metadata.sqlite3")
    artifact = store.put_bytes(b"episode")
    spec = _run_spec(artifact.digest)

    pending = repository.create_run(spec)
    assert pending.status is RunStatus.PENDING
    assert repository.create_run(spec) == pending

    with pytest.raises(ResourceConflictError):
        repository.create_run(_run_spec(artifact.digest, seed=8))
    with pytest.raises(InvalidStateTransitionError):
        repository.transition_run(_RUN_ID, RunStatus.SUCCEEDED)

    running = repository.transition_run(_RUN_ID, RunStatus.RUNNING)
    assert running.status is RunStatus.RUNNING
    assert running.started_at is not None

    summary = _episode(artifact)
    assert repository.publish_episode(summary) == summary
    assert repository.publish_episode(summary) == summary
    assert repository.get_episode(_RUN_ID, summary.episode_id) == summary
    assert repository.list_episodes(_RUN_ID) == [summary]

    conflict = _episode(store.put_bytes(b"different episode"))
    with pytest.raises(ResourceConflictError):
        repository.publish_episode(conflict)
    duplicate_index = summary.model_copy(
        update={"episode_id": "00000000-0000-0000-0000-000000000003"}
    )
    with pytest.raises(ResourceConflictError):
        repository.publish_episode(duplicate_index)

    dataset = _reference("dataset", "drawer_dataset", store.put_json({"ok": True}).digest)
    wrong_count = RunResult(
        run_id=_RUN_ID,
        status=RunStatus.SUCCEEDED,
        episode_count=0,
        dataset=dataset,
    )
    with pytest.raises(DomainValidationError):
        repository.finalize_success(_RUN_ID, dataset, wrong_count)
    assert repository.get_run(_RUN_ID).status is RunStatus.RUNNING
    with pytest.raises(ResourceNotFoundError):
        repository.get_dataset(dataset.resource_id, dataset.version)

    result = RunResult(
        run_id=_RUN_ID,
        status=RunStatus.SUCCEEDED,
        episode_count=1,
        dataset=dataset,
    )
    succeeded = repository.finalize_success(_RUN_ID, dataset, result)
    assert succeeded.status is RunStatus.SUCCEEDED
    assert succeeded.result == result
    assert repository.get_dataset(dataset.resource_id, dataset.version) == dataset
    assert succeeded.completed_at is not None
    assert repository.list_runs(status=RunStatus.SUCCEEDED) == [succeeded]

    with pytest.raises(InvalidStateTransitionError):
        repository.transition_run(_RUN_ID, RunStatus.CANCELLED, result=result)
    with pytest.raises(InvalidStateTransitionError):
        repository.publish_episode(summary)


@pytest.mark.integration
def test_pending_run_can_be_cancelled_atomically(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "metadata.sqlite3")
    repository.create_run(_run_spec(f"sha256:{'e' * 64}"))
    result = RunResult(
        run_id=_RUN_ID,
        status=RunStatus.CANCELLED,
        episode_count=0,
        error_code="cancelled",
        error_message="cancelled before execution",
    )

    cancelled = repository.transition_run(_RUN_ID, RunStatus.CANCELLED, result=result)

    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.started_at == cancelled.completed_at


@pytest.mark.integration
def test_missing_queries_do_not_conflate_absence_with_empty_results(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "metadata.sqlite3")

    with pytest.raises(ResourceNotFoundError):
        repository.get_resource("asset", "missing", "1.0.0")
    with pytest.raises(ResourceNotFoundError):
        repository.get_run("missing")
    with pytest.raises(ResourceNotFoundError):
        repository.list_episodes("missing")
    with pytest.raises(ResourceNotFoundError):
        repository.get_episode("missing", _EPISODE_ID)
