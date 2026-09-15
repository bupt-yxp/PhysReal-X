"""Durable local artifact and metadata storage.

The storage layer deliberately has no knowledge of a simulation backend.  It
persists immutable, identifier-addressed artifacts and the small amount of
transactional metadata needed by the application layer.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeVar

from pydantic import BaseModel

from .domain import (
    ArtifactRef,
    EpisodeSummary,
    RunRecord,
    RunResult,
    RunSpec,
    RunStatus,
    VersionedRef,
)
from .errors import (
    ArtifactIntegrityError,
    DomainValidationError,
    InvalidStateTransitionError,
    ResourceConflictError,
    ResourceNotFoundError,
)

_SCHEMA_VERSION: Final = 2
_COPY_CHUNK_SIZE: Final = 1024 * 1024
_DEFAULT_BUSY_TIMEOUT_MS: Final = 5_000
_MINIMUM_SQLITE_VERSION: Final = (3, 37, 0)
_T = TypeVar("_T", bound=BaseModel)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(value: Any) -> str:
    """Serialize metadata deterministically and reject non-finite numbers."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _enum_value(value: str | Enum) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _validate_artifact_id(artifact_id: str) -> str:
    if not isinstance(artifact_id, str) or not artifact_id.startswith("art_"):
        raise DomainValidationError("artifact_id must use the platform-generated 'art_' form")
    try:
        uuid.UUID(hex=artifact_id[4:])
    except ValueError as error:
        raise DomainValidationError("artifact_id is malformed") from error
    return artifact_id


def _model_from_json(model: type[_T], payload: str) -> _T:
    return model.model_validate_json(payload)


class LocalArtifactStore:
    """Immutable objects addressed by platform-generated identifiers.

    Files are atomically published and never overwritten.  The platform does
    not calculate or expose file hashes; backend durability and the
    immutable identifier are the storage contract.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        if not root_path.is_dir():
            raise DomainValidationError(f"artifact root is not a directory: {root_path}")
        self._root = root_path.resolve()
        self._objects = self._root / "objects"
        self._objects.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def put_bytes(
        self,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        if not isinstance(media_type, str) or not media_type.strip():
            raise DomainValidationError("media_type must be a non-empty string")
        content = bytes(data)
        artifact_id = self._new_artifact_id()
        destination = self._path_for_artifact_id(artifact_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        return ArtifactRef(
            artifact_id=artifact_id,
            size_bytes=len(content),
            media_type=media_type,
        )

    def put_json(self, value: Any, *, media_type: str = "application/json") -> ArtifactRef:
        encoded = _canonical_json(value).encode("utf-8")
        return self.put_bytes(encoded, media_type=media_type)

    def put_file(
        self,
        source: Path,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        if not isinstance(source, Path):
            raise TypeError("source must be a pathlib.Path")
        if not isinstance(media_type, str) or not media_type.strip():
            raise DomainValidationError("media_type must be a non-empty string")
        expanded = source.expanduser()
        if expanded.is_symlink():
            raise DomainValidationError(f"artifact source cannot be a symbolic link: {expanded}")
        source_path = expanded.resolve(strict=True)
        if not source_path.is_file():
            raise DomainValidationError(f"artifact source is not a regular file: {source_path}")

        before = source_path.stat()
        artifact_id = self._new_artifact_id()
        destination = self._path_for_artifact_id(artifact_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        size = 0
        try:
            with source_path.open("rb") as source_stream, os.fdopen(descriptor, "wb") as output:
                for chunk in iter(lambda: source_stream.read(_COPY_CHUNK_SIZE), b""):
                    size += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = source_path.stat()
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if identity_before != identity_after or size != before.st_size:
                raise ArtifactIntegrityError(
                    f"artifact source changed during ingestion: {source_path}"
                )
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
            return ArtifactRef(
                artifact_id=artifact_id,
                size_bytes=size,
                media_type=media_type,
            )
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()

    def get_bytes(self, reference: ArtifactRef | str) -> bytes:
        artifact_id, expected_size = self._reference_parts(reference)
        path = self._path_for_artifact_id(artifact_id)
        if not path.is_file():
            raise ResourceNotFoundError(f"artifact not found: {artifact_id}")
        content = path.read_bytes()
        if expected_size is not None and len(content) != expected_size:
            raise ArtifactIntegrityError(
                f"artifact size mismatch for {artifact_id}: "
                f"expected {expected_size}, got {len(content)}"
            )
        return content

    def export(
        self,
        destination: Path,
        results: Mapping[str, Any],
        *,
        files: Mapping[str, ArtifactRef] | None = None,
    ) -> ArtifactRef:
        """Stage verified files and publish results.json last as the completion marker."""
        files = dict(files or {})
        for name in files:
            if name in {"", ".", "..", "results.json"} or Path(name).name != name:
                raise DomainValidationError(
                    "export filenames must be simple names other than results.json"
                )
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in (*files, "results.json"):
            target = destination / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise DomainValidationError(f"export target must be a regular file: {target}")
        reference = self.put_json(results)
        with tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as directory:
            stage = Path(directory)
            for name, artifact in files.items():
                self.materialize(artifact, stage / name)
            self.materialize(reference, stage / "results.json")
            # Once replacement starts, a failed export must not retain a stale success marker.
            (destination / "results.json").unlink(missing_ok=True)
            for name in files:
                (stage / name).replace(destination / name)
            (stage / "results.json").replace(destination / "results.json")
        return reference

    def materialize(self, reference: ArtifactRef | str, destination: Path) -> None:
        if not isinstance(destination, Path):
            raise TypeError("destination must be a pathlib.Path")
        artifact_id, expected_size = self._reference_parts(reference)
        source = self._path_for_artifact_id(artifact_id)
        if not source.is_file():
            raise ResourceNotFoundError(f"artifact not found: {artifact_id}")
        self._assert_valid_file(source, artifact_id, expected_size)
        if destination.is_symlink() or destination.exists():
            raise DomainValidationError(
                f"artifact materialization destination must not exist: {destination}"
            )
        parent = destination.parent.resolve(strict=True)
        if not parent.is_dir():
            raise DomainValidationError(
                f"artifact materialization parent is not a directory: {parent}"
            )
        size = 0
        try:
            with source.open("rb") as source_stream, destination.open("xb") as output:
                for chunk in iter(lambda: source_stream.read(_COPY_CHUNK_SIZE), b""):
                    size += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if expected_size is not None and size != expected_size:
                raise ArtifactIntegrityError(
                    f"materialized artifact size does not match {artifact_id}"
                )
        except Exception:
            with suppress(FileNotFoundError):
                destination.unlink()
            raise

    def verify(self, reference: ArtifactRef | str) -> bool:
        artifact_id, expected_size = self._reference_parts(reference)
        path = self._path_for_artifact_id(artifact_id)
        if not path.is_file():
            return False
        try:
            self._assert_valid_file(path, artifact_id, expected_size)
        except ArtifactIntegrityError:
            return False
        return True

    def _reference_parts(self, reference: ArtifactRef | str) -> tuple[str, int | None]:
        if isinstance(reference, ArtifactRef):
            return _validate_artifact_id(reference.artifact_id), reference.size_bytes
        return _validate_artifact_id(reference), None

    def _path_for_artifact_id(self, artifact_id: str) -> Path:
        artifact_id = _validate_artifact_id(artifact_id)
        hexadecimal = artifact_id[4:]
        return self._objects / hexadecimal[:2] / artifact_id

    def _new_artifact_id(self) -> str:
        while True:
            artifact_id = f"art_{uuid.uuid4().hex}"
            if not self._path_for_artifact_id(artifact_id).exists():
                return artifact_id

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _assert_valid_file(path: Path, artifact_id: str, expected_size: int | None) -> None:
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            raise ArtifactIntegrityError(
                f"cannot inspect artifact {artifact_id}: {error}"
            ) from error
        if expected_size is not None and actual_size != expected_size:
            raise ArtifactIntegrityError(
                f"artifact size mismatch for {artifact_id}: "
                f"expected {expected_size}, got {actual_size}"
            )


class SQLiteRepository:
    """SQLite metadata repository with explicit transactional invariants."""

    _ALLOWED_TRANSITIONS: Final[Mapping[RunStatus, frozenset[RunStatus]]] = {
        RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
        RunStatus.RUNNING: frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}),
        RunStatus.SUCCEEDED: frozenset(),
        RunStatus.FAILED: frozenset(),
        RunStatus.CANCELLED: frozenset(),
    }

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if (
            isinstance(busy_timeout_ms, bool)
            or not isinstance(busy_timeout_ms, int)
            or busy_timeout_ms <= 0
        ):
            raise DomainValidationError("busy_timeout_ms must be a positive integer")
        if sqlite3.sqlite_version_info < _MINIMUM_SQLITE_VERSION:
            detected = ".".join(str(value) for value in sqlite3.sqlite_version_info)
            minimum = ".".join(str(value) for value in _MINIMUM_SQLITE_VERSION)
            raise RuntimeError(
                f"SQLite >= {minimum} is required for the STRICT metadata schema; "
                f"detected {detected}"
            )
        database_path = Path(database).expanduser()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database = database_path.resolve()
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._initialize_schema()

    @property
    def database(self) -> Path:
        return self._database

    @property
    def schema_version(self) -> int:
        return _SCHEMA_VERSION

    def register_resource(self, reference: VersionedRef) -> VersionedRef:
        """Idempotently register an immutable versioned resource."""

        resource_type, resource_id, version, artifact_id = self._versioned_parts(reference)
        with self._transaction() as connection:
            self._insert_versioned(
                connection,
                resource_type=resource_type,
                resource_id=resource_id,
                version=version,
                artifact_id=artifact_id,
            )
        return reference

    def register_resources(
        self,
        references: Iterable[VersionedRef],
    ) -> tuple[VersionedRef, ...]:
        """Atomically register a group of immutable versioned resources."""

        items = tuple(references)
        parts = tuple(self._versioned_parts(reference) for reference in items)
        with self._transaction() as connection:
            for resource_type, resource_id, version, artifact_id in parts:
                self._insert_versioned(
                    connection,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    version=version,
                    artifact_id=artifact_id,
                )
        return items

    def get_resource(self, resource_type: str, resource_id: str, version: str) -> VersionedRef:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT resource_type, resource_id, version, artifact_id
                FROM resources
                WHERE resource_type = ? AND resource_id = ? AND version = ?
                """,
                (resource_type, resource_id, version),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError(
                f"resource not found: {resource_type}/{resource_id}@{version}"
            )
        return self._versioned_from_row(row)

    def list_resources(
        self,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> list[VersionedRef]:
        clauses: list[str] = []
        parameters: list[str] = []
        if resource_type is not None:
            clauses.append("resource_type = ?")
            parameters.append(resource_type)
        if resource_id is not None:
            clauses.append("resource_id = ?")
            parameters.append(resource_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT resource_type, resource_id, version, artifact_id
                FROM resources
                """
                + where
                + " ORDER BY resource_type, resource_id, version",
                parameters,
            ).fetchall()
        return [self._versioned_from_row(row) for row in rows]

    def register_dataset(self, reference: VersionedRef) -> VersionedRef:
        return self._register_specialized("dataset", reference)

    def get_dataset(self, resource_id: str, version: str) -> VersionedRef:
        return self._get_specialized("dataset", resource_id, version)

    def list_datasets(self, *, resource_id: str | None = None) -> list[VersionedRef]:
        return self._list_specialized("dataset", resource_id)

    def register_model(self, reference: VersionedRef) -> VersionedRef:
        return self._register_specialized("model", reference)

    def get_model(self, resource_id: str, version: str) -> VersionedRef:
        return self._get_specialized("model", resource_id, version)

    def list_models(self, *, resource_id: str | None = None) -> list[VersionedRef]:
        return self._list_specialized("model", resource_id)

    def register_evaluation(self, reference: VersionedRef) -> VersionedRef:
        return self._register_specialized("evaluation", reference)

    def get_evaluation(self, resource_id: str, version: str) -> VersionedRef:
        return self._get_specialized("evaluation", resource_id, version)

    def list_evaluations(self, *, resource_id: str | None = None) -> list[VersionedRef]:
        return self._list_specialized("evaluation", resource_id)

    def create_run(self, spec: RunSpec) -> RunRecord:
        """Create a pending run, idempotently for an identical specification."""

        run_id = spec.run_id
        serialized_spec = _canonical_json(spec)
        now = _utc_now()
        pending = RunRecord(
            run_id=run_id,
            spec=spec,
            status=RunStatus.PENDING,
            result=None,
            created_at=now,
            updated_at=now,
        )
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT spec_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None:
                if row["spec_json"] != serialized_spec:
                    raise ResourceConflictError(
                        f"run {run_id!r} already exists with a different specification"
                    )
                return self._get_run(connection, run_id)

            connection.execute(
                """
                INSERT INTO runs(
                    run_id, spec_json, status, result_json, created_at, updated_at,
                    started_at, completed_at
                ) VALUES (?, ?, ?, NULL, ?, ?, NULL, NULL)
                """,
                (
                    run_id,
                    serialized_spec,
                    RunStatus.PENDING.value,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            return pending

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            return self._get_run(connection, run_id)

    def list_runs(self, *, status: RunStatus | None = None) -> list[RunRecord]:
        parameters: tuple[str, ...] = ()
        where = ""
        if status is not None:
            where = " WHERE status = ?"
            parameters = (status.value,)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs" + where + " ORDER BY created_at, run_id", parameters
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def transition_run(
        self,
        run_id: str,
        target: RunStatus,
        *,
        result: RunResult | None = None,
    ) -> RunRecord:
        """Apply one valid edge of the run state machine atomically."""

        if not isinstance(target, RunStatus):
            raise DomainValidationError("target must be a RunStatus")
        with self._transaction() as connection:
            return self._transition_run(connection, run_id, target, result=result)

    def finalize_success(
        self,
        run_id: str,
        dataset: VersionedRef,
        result: RunResult,
    ) -> RunRecord:
        """Publish a Dataset and terminal Run state in one SQLite transaction."""

        resource_type, resource_id, version, artifact_id = self._versioned_parts(dataset)
        if resource_type != "dataset":
            raise DomainValidationError("finalize_success requires a dataset reference")
        if result.dataset != dataset:
            raise DomainValidationError("RunResult dataset does not match the finalized dataset")
        with self._transaction() as connection:
            self._insert_versioned(
                connection,
                resource_type=resource_type,
                resource_id=resource_id,
                version=version,
                artifact_id=artifact_id,
            )
            return self._transition_run(
                connection,
                run_id,
                RunStatus.SUCCEEDED,
                result=result,
            )

    def _transition_run(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        target: RunStatus,
        *,
        result: RunResult | None,
    ) -> RunRecord:
        current = self._get_run(connection, run_id)
        if target not in self._ALLOWED_TRANSITIONS[current.status]:
            raise InvalidStateTransitionError(
                f"run {run_id!r} cannot transition from {current.status.value} to {target.value}"
            )

        is_terminal = target in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
        if is_terminal and result is None:
            raise DomainValidationError(f"transition to {target.value} requires a RunResult")
        if not is_terminal and result is not None:
            raise DomainValidationError("a non-terminal transition cannot store a RunResult")
        if result is not None:
            if result.run_id != run_id:
                raise DomainValidationError("RunResult.run_id does not match the target run")
            if result.status is not target:
                raise DomainValidationError("RunResult.status does not match the target status")
            published_episode_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM episodes WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            )
            if result.episode_count != published_episode_count:
                raise DomainValidationError(
                    "RunResult.episode_count does not match the number of published episodes"
                )
            if (
                target is RunStatus.SUCCEEDED
                and published_episode_count != current.spec.episode_count
            ):
                raise DomainValidationError("a successful run must publish every requested episode")

        now = _utc_now()
        started_at = current.started_at
        completed_at = current.completed_at
        if target is RunStatus.RUNNING:
            started_at = now
        if is_terminal:
            if started_at is None:
                started_at = now
            completed_at = now

        cursor = connection.execute(
            """
            UPDATE runs
            SET status = ?, result_json = ?, updated_at = ?, started_at = ?, completed_at = ?
            WHERE run_id = ? AND status = ?
            """,
            (
                target.value,
                _canonical_json(result) if result is not None else None,
                now.isoformat(),
                started_at.isoformat() if started_at is not None else None,
                completed_at.isoformat() if completed_at is not None else None,
                run_id,
                current.status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise ResourceConflictError(f"run {run_id!r} was concurrently modified")
        return self._get_run(connection, run_id)

    def publish_episode(self, summary: EpisodeSummary) -> EpisodeSummary:
        """Record an immutable Episode summary for a running job."""

        run_id = summary.run_id
        serialized = _canonical_json(summary)
        if summary.artifact is not None:
            _validate_artifact_id(summary.artifact.artifact_id)
        with self._transaction() as connection:
            run = self._get_run(connection, run_id)
            if run.status is not RunStatus.RUNNING:
                raise InvalidStateTransitionError(
                    f"episodes may only be published while a run is running; "
                    f"{run_id!r} is {run.status.value}"
                )
            row = connection.execute(
                "SELECT summary_json FROM episodes WHERE run_id = ? AND episode_id = ?",
                (run_id, summary.episode_id),
            ).fetchone()
            if row is not None:
                if row["summary_json"] != serialized:
                    raise ResourceConflictError(
                        f"episode {summary.episode_id!r} for run {run_id!r} already exists "
                        "with different content"
                    )
                return summary

            row = connection.execute(
                "SELECT episode_id FROM episodes WHERE run_id = ? AND episode_index = ?",
                (run_id, summary.index),
            ).fetchone()
            if row is not None:
                raise ResourceConflictError(
                    f"episode index {summary.index} for run {run_id!r} is already used by "
                    f"episode {row['episode_id']!r}"
                )

            connection.execute(
                """
                INSERT INTO episodes(
                    run_id, episode_id, episode_index, summary_json, artifact_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    summary.episode_id,
                    summary.index,
                    serialized,
                    summary.artifact.artifact_id if summary.artifact is not None else None,
                    _utc_now().isoformat(),
                ),
            )
        return summary

    def get_episode(self, run_id: str, episode_id: str) -> EpisodeSummary:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary_json FROM episodes WHERE run_id = ? AND episode_id = ?",
                (run_id, episode_id),
            ).fetchone()
        if row is None:
            raise ResourceNotFoundError(
                f"episode not found: run={run_id!r}, episode={episode_id!r}"
            )
        return _model_from_json(EpisodeSummary, row["summary_json"])

    def list_episodes(self, run_id: str) -> list[EpisodeSummary]:
        # Distinguish a run with no episodes from a misspelled/non-existent run.
        with self._connect() as connection:
            self._get_run(connection, run_id)
            rows = connection.execute(
                """
                SELECT summary_json FROM episodes
                WHERE run_id = ? ORDER BY episode_index
                """,
                (run_id,),
            ).fetchall()
        return [_model_from_json(EpisodeSummary, row["summary_json"]) for row in rows]

    def _register_specialized(self, expected_type: str, reference: VersionedRef) -> VersionedRef:
        resource_type, resource_id, version, artifact_id = self._versioned_parts(reference)
        if resource_type != expected_type:
            raise DomainValidationError(
                f"expected a {expected_type!r} reference, got {resource_type!r}"
            )
        with self._transaction() as connection:
            self._insert_versioned(
                connection,
                resource_type=resource_type,
                resource_id=resource_id,
                version=version,
                artifact_id=artifact_id,
            )
        return reference

    def _get_specialized(self, expected_type: str, resource_id: str, version: str) -> VersionedRef:
        return self.get_resource(expected_type, resource_id, version)

    def _list_specialized(self, expected_type: str, resource_id: str | None) -> list[VersionedRef]:
        return self.list_resources(resource_type=expected_type, resource_id=resource_id)

    @staticmethod
    def _versioned_parts(reference: VersionedRef) -> tuple[str, str, str, str]:
        resource_type = _enum_value(reference.resource_type)
        if reference.artifact_id is None:
            raise DomainValidationError("registered resources require a manifest artifact_id")
        return (
            resource_type,
            reference.resource_id,
            reference.version,
            _validate_artifact_id(reference.artifact_id),
        )

    @staticmethod
    def _insert_versioned(
        connection: sqlite3.Connection,
        *,
        resource_type: str,
        resource_id: str,
        version: str,
        artifact_id: str,
    ) -> None:
        select_query = """
            SELECT artifact_id FROM resources
            WHERE resource_type = ? AND resource_id = ? AND version = ?
            """
        row = connection.execute(
            select_query,
            (resource_type, resource_id, version),
        ).fetchone()
        if row is not None:
            if row["artifact_id"] != artifact_id:
                raise ResourceConflictError(
                    f"{resource_type}/{resource_id}@{version} is already registered "
                    "with a different manifest artifact"
                )
            return
        insert_query = """
            INSERT INTO resources(resource_type, resource_id, version, artifact_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """
        connection.execute(
            insert_query,
            (resource_type, resource_id, version, artifact_id, _utc_now().isoformat()),
        )

    @staticmethod
    def _versioned_from_row(row: sqlite3.Row) -> VersionedRef:
        return VersionedRef(
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            version=row["version"],
            artifact_id=row["artifact_id"],
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        result_json = row["result_json"]
        return RunRecord(
            run_id=row["run_id"],
            spec=_model_from_json(RunSpec, row["spec_json"]),
            status=RunStatus(row["status"]),
            result=_model_from_json(RunResult, result_json) if result_json is not None else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"]) if row["started_at"] is not None else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"] is not None
                else None
            ),
        )

    def _get_run(self, connection: sqlite3.Connection, run_id: str) -> RunRecord:
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ResourceNotFoundError(f"run not found: {run_id}")
        return self._run_from_row(row)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._database,
            timeout=self._busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms:d}")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RuntimeError(f"SQLite refused WAL mode: {journal_mode}")
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if user_version not in {0, _SCHEMA_VERSION}:
                raise RuntimeError(
                    f"unsupported database schema version {user_version}; "
                    f"expected {_SCHEMA_VERSION}"
                )
            connection.execute("PRAGMA synchronous = FULL")
            try:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;

                    CREATE TABLE IF NOT EXISTS schema_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    ) STRICT;

                    CREATE TABLE IF NOT EXISTS resources (
                        resource_type TEXT NOT NULL,
                        resource_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        artifact_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(resource_type, resource_id, version)
                    ) STRICT;

                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        spec_json TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(
                            status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')
                        ),
                        result_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT
                    ) STRICT;

                    CREATE TABLE IF NOT EXISTS episodes (
                        run_id TEXT NOT NULL,
                        episode_id TEXT NOT NULL,
                        episode_index INTEGER NOT NULL CHECK(episode_index >= 0),
                        summary_json TEXT NOT NULL,
                        artifact_id TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(run_id, episode_id),
                        UNIQUE(run_id, episode_index),
                        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
                    ) STRICT;

                    CREATE INDEX IF NOT EXISTS episodes_by_run
                    ON episodes(run_id, created_at);
                    """
                )
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                        (str(_SCHEMA_VERSION),),
                    )
                elif row["value"] != str(_SCHEMA_VERSION):
                    raise RuntimeError(
                        f"unsupported database schema version {row['value']}; "
                        f"expected {_SCHEMA_VERSION}"
                    )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION:d}")
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


__all__ = ["LocalArtifactStore", "SQLiteRepository"]
