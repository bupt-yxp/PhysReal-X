"""Stable identities for immutable artifact graphs with opaque storage IDs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError

_ARTIFACT_REFERENCE_FIELDS = frozenset({"artifact_id", "size_bytes", "media_type"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def artifact_graph_sha256(
    artifacts: ArtifactStore,
    reference: ArtifactRef,
    *,
    ignored_fields: frozenset[str] = frozenset(),
    cache: dict[str, str] | None = None,
    stack: set[str] | None = None,
) -> str:
    """Hash content recursively while replacing opaque nested artifact IDs."""
    cache = {} if cache is None else cache
    stack = set() if stack is None else stack
    if reference.artifact_id in cache:
        return cache[reference.artifact_id]
    if reference.artifact_id in stack:
        raise DomainValidationError("artifact graph contains a reference cycle")
    content = artifacts.get_bytes(reference)
    if len(content) != reference.size_bytes:
        raise DomainValidationError("artifact size changed while computing content identity")
    stack.add(reference.artifact_id)
    try:
        if reference.media_type == "application/json" or reference.media_type.endswith("+json"):
            try:
                payload = json.loads(content)
            except (UnicodeDecodeError, ValueError) as error:
                raise DomainValidationError(
                    "JSON artifact is invalid while computing content identity"
                ) from error

            def normalize(value: Any) -> Any:
                if isinstance(value, dict):
                    if set(value) == _ARTIFACT_REFERENCE_FIELDS:
                        nested = ArtifactRef.model_validate(value)
                        return {
                            "content_sha256": artifact_graph_sha256(
                                artifacts,
                                nested,
                                ignored_fields=ignored_fields,
                                cache=cache,
                                stack=stack,
                            ),
                            "media_type": nested.media_type,
                        }
                    return {
                        key: normalize(item)
                        for key, item in value.items()
                        if key not in ignored_fields
                    }
                if isinstance(value, list):
                    return [normalize(item) for item in value]
                return value

            identity_bytes = _canonical(
                {"media_type": reference.media_type, "payload": normalize(payload)}
            )
        else:
            identity_bytes = reference.media_type.encode("utf-8") + b"\0" + content
        identity = hashlib.sha256(identity_bytes).hexdigest()
        cache[reference.artifact_id] = identity
        return identity
    finally:
        stack.remove(reference.artifact_id)
