"""Atomic local checkpoints for long articulated collision searches."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifact_identity import artifact_graph_sha256
from physrealx.integrations._collision_refit import SurfacePatch
from physrealx.integrations.collision import load_meshes


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


_CHECKPOINT_IDENTITY_IGNORED_FIELDS = frozenset({"resource_id"})


class RefinementCheckpointStore:
    """Persist accepted search states without making artifact storage mutable."""

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        source_assembly: ArtifactRef,
        source_feasibility: ArtifactRef,
        config: dict[str, Any],
        mode: str,
        resume_from: ArtifactRef | None,
        lock_writer: bool = False,
    ) -> None:
        if mode not in {"auto", "disabled", "required"}:
            raise ValueError("invalid checkpoint mode")
        self.artifacts = artifacts
        self.source_assembly = source_assembly
        self.source_feasibility = source_feasibility
        self.config = dict(config)
        self.mode = mode
        self.resume_from = resume_from
        identity_cache: dict[str, str] = {}
        self.source_identity = {
            "assembly": artifact_graph_sha256(
                artifacts,
                source_assembly,
                ignored_fields=_CHECKPOINT_IDENTITY_IGNORED_FIELDS,
                cache=identity_cache,
            ),
            "feasibility": artifact_graph_sha256(
                artifacts,
                source_feasibility,
                ignored_fields=_CHECKPOINT_IDENTITY_IGNORED_FIELDS,
                cache=identity_cache,
            ),
        }
        identity = {
            "source_identity": self.source_identity,
            "config": self.config,
        }
        self.key = hashlib.sha256(_canonical(identity)).hexdigest()
        root = getattr(artifacts, "root", None)
        self.pointer: Path | None = None
        if root is not None and mode != "disabled":
            directory = Path(root) / "refinement-checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            self.pointer = directory / f"{self.key}.json"
        elif mode == "required" and resume_from is None:
            raise MethodExecutionError(
                "required refinement checkpoints need a local artifact root or resume artifact"
            )
        self.latest: ArtifactRef | None = None
        self._lock_handle: Any | None = None
        if lock_writer and self.pointer is not None:
            self._acquire_writer_lock()

    def _acquire_writer_lock(self) -> None:
        """Prevent concurrent optimizers from racing one mutable pointer."""
        import fcntl

        if self.pointer is None or self._lock_handle is not None:
            return
        lock_path = self.pointer.with_suffix(".lock")
        handle = lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise MethodExecutionError(
                "another articulated collision optimizer is writing this checkpoint"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        self._lock_handle = handle

    def close(self) -> None:
        """Release this process's checkpoint writer lock."""
        if self._lock_handle is None:
            return
        import fcntl

        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._lock_handle = None

    def __del__(self) -> None:
        self.close()

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def _pointer_reference(self) -> ArtifactRef | None:
        if self.resume_from is not None:
            return self.resume_from
        if self.pointer is None or not self.pointer.exists():
            return None
        try:
            value = json.loads(self.pointer.read_text(encoding="utf-8"))
            if value.get("key") != self.key:
                return None
            return ArtifactRef.model_validate(value["checkpoint"])
        except (OSError, ValueError, KeyError):
            return None

    def load(self, visuals: dict[str, Any]) -> dict[str, Any] | None:
        reference = self._pointer_reference()
        if reference is None:
            return None
        if not self.artifacts.verify(reference):
            raise DomainValidationError("refinement checkpoint artifact is unavailable")
        document = json.loads(self.artifacts.get_bytes(reference))
        identity_cache: dict[str, str] = {}
        document_source_identity = {
            "assembly": artifact_graph_sha256(
                self.artifacts,
                ArtifactRef.model_validate(document.get("source_assembly")),
                ignored_fields=_CHECKPOINT_IDENTITY_IGNORED_FIELDS,
                cache=identity_cache,
            ),
            "feasibility": artifact_graph_sha256(
                self.artifacts,
                ArtifactRef.model_validate(document.get("source_feasibility")),
                ignored_fields=_CHECKPOINT_IDENTITY_IGNORED_FIELDS,
                cache=identity_cache,
            ),
        }
        if (
            document.get("schema") != "physrealx.articulated-refinement-checkpoint/2"
            or document.get("key") != self.key
            or document.get("source_identity") != document_source_identity
            or document.get("source_identity") != self.source_identity
            or document.get("config") != self.config
        ):
            raise DomainValidationError("refinement checkpoint does not match immutable inputs")
        ownership_ref = ArtifactRef.model_validate(document["ownership"])
        if not self.artifacts.verify(ownership_ref):
            raise DomainValidationError("refinement checkpoint ownership is unavailable")
        archive = np.load(io.BytesIO(self.artifacts.get_bytes(ownership_ref)), allow_pickle=False)

        def restore_candidate(
            geometry_key: str, array_prefix: str
        ) -> tuple[dict[str, list[SurfacePatch]], dict[str, ArtifactRef]]:
            references = {
                link: ArtifactRef.model_validate(value)
                for link, value in document[geometry_key].items()
            }
            if set(references) != set(visuals):
                raise DomainValidationError("checkpoint links differ from visual geometry")
            patches: dict[str, list[SurfacePatch]] = {}
            for link, visual in visuals.items():
                meshes = load_meshes(self.artifacts, references[link], process=False)
                owners = np.asarray(archive[f"{array_prefix}{link}.owners"], dtype=np.int64)
                depths = np.asarray(archive[f"{array_prefix}{link}.depths"], dtype=np.int64)
                volume_key = f"{array_prefix}{link}.baseline_volumes_m3"
                thickness_key = f"{array_prefix}{link}.baseline_thicknesses_m"
                baseline_volumes = np.asarray(archive[volume_key], dtype=float)
                baseline_thicknesses = np.asarray(archive[thickness_key], dtype=float)
                if (
                    len(owners) != len(visual.faces)
                    or len(depths) != len(meshes)
                    or len(baseline_volumes) != len(meshes)
                    or len(baseline_thicknesses) != len(meshes)
                    or np.any(owners < 0)
                    or np.any(owners >= len(meshes))
                ):
                    raise DomainValidationError("checkpoint source-face ownership is invalid")
                patches[link] = [
                    SurfacePatch(
                        mesh,
                        np.flatnonzero(owners == index),
                        int(depths[index]),
                        baseline_volume_m3=(
                            float(baseline_volumes[index])
                            if np.isfinite(baseline_volumes[index])
                            else None
                        ),
                        baseline_minimum_thickness_m=(
                            float(baseline_thicknesses[index])
                            if np.isfinite(baseline_thicknesses[index])
                            else None
                        ),
                    )
                    for index, mesh in enumerate(meshes)
                ]
                # CoACD may emit a valid convex part that owns no nearest source face.
                # Preserve it exactly: deleting such a part during resume would make a
                # checkpoint alter collision geometry merely to simplify the search.
                if any(
                    not patch.mesh.is_volume or not patch.mesh.is_convex for patch in patches[link]
                ):
                    raise DomainValidationError("checkpoint contains an invalid collision part")
            return patches, references

        patches, references = restore_candidate("collision_geometries", "")
        contact_faces: dict[str, NDArray[Any]] = {}
        for link, visual in visuals.items():
            fixed_contact_faces = np.asarray(archive[f"{link}.contact_faces"], dtype=np.int64)
            if np.any(fixed_contact_faces < 0) or np.any(fixed_contact_faces >= len(visual.faces)):
                raise DomainValidationError("checkpoint contact-face evidence is invalid")
            contact_faces[link] = fixed_contact_faces
        best = None
        if document.get("best") is not None:
            best_patches, best_references = restore_candidate("best_collision_geometries", "best.")
            best = {
                **document["best"],
                "patches": best_patches,
                "references": best_references,
            }
        self.latest = reference
        return {
            **document["state"],
            "patches": patches,
            "contact_faces": contact_faces,
            "references": references,
            "best": best,
            "checkpoint": reference,
        }

    def save(
        self,
        *,
        patches: dict[str, list[SurfacePatch]],
        contact_faces: dict[str, Any],
        references: dict[str, ArtifactRef],
        state: dict[str, Any],
        best: dict[str, Any] | None = None,
    ) -> ArtifactRef | None:
        if not self.enabled:
            return None
        arrays: dict[str, NDArray[Any]] = {}

        def encode_candidate(values_by_link: dict[str, list[SurfacePatch]], prefix: str) -> None:
            for link, values in values_by_link.items():
                face_count = sum(len(value.faces) for value in values)
                owners = np.full(face_count, -1, dtype=np.int32)
                for index, patch in enumerate(values):
                    owners[np.asarray(patch.faces, dtype=np.int64)] = index
                if np.any(owners < 0):
                    raise MethodExecutionError("checkpoint would drop source-face ownership")
                arrays[f"{prefix}{link}.owners"] = owners
                arrays[f"{prefix}{link}.depths"] = np.asarray(
                    [value.depth for value in values], dtype=np.int16
                )
                arrays[f"{prefix}{link}.baseline_volumes_m3"] = np.asarray(
                    [
                        value.baseline_volume_m3 if value.baseline_volume_m3 is not None else np.nan
                        for value in values
                    ],
                    dtype=np.float64,
                )
                arrays[f"{prefix}{link}.baseline_thicknesses_m"] = np.asarray(
                    [
                        value.baseline_minimum_thickness_m
                        if value.baseline_minimum_thickness_m is not None
                        else np.nan
                        for value in values
                    ],
                    dtype=np.float64,
                )

        encode_candidate(patches, "")
        for link in patches:
            arrays[f"{link}.contact_faces"] = np.asarray(contact_faces[link], dtype=np.int32)
        if best is not None:
            encode_candidate(best["patches"], "best.")
        payload = io.BytesIO()
        np.savez(payload, **arrays)
        ownership = self.artifacts.put_bytes(
            payload.getvalue(), media_type="application/vnd.physrealx.face-ownership+npz"
        )
        reference = self.artifacts.put_json(
            {
                "schema": "physrealx.articulated-refinement-checkpoint/2",
                "key": self.key,
                "source_identity": self.source_identity,
                "source_assembly": self.source_assembly.model_dump(mode="json"),
                "source_feasibility": self.source_feasibility.model_dump(mode="json"),
                "config": self.config,
                "ownership": ownership.model_dump(mode="json"),
                "collision_geometries": {
                    link: value.model_dump(mode="json") for link, value in references.items()
                },
                "state": state,
                "best_collision_geometries": {
                    link: value.model_dump(mode="json")
                    for link, value in best["references"].items()
                }
                if best is not None
                else None,
                "best": best["state"] if best is not None else None,
            },
            media_type="application/vnd.physrealx.refinement-checkpoint+json",
        )
        if self.pointer is not None:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.pointer.parent, prefix=f".{self.pointer.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(
                        _canonical(
                            {
                                "key": self.key,
                                "checkpoint": reference.model_dump(mode="json"),
                            }
                        )
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.pointer)
            finally:
                if temporary.exists():
                    temporary.unlink()
        self.latest = reference
        return reference
