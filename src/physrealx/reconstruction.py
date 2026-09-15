"""Method-independent reconstructed surfaces and motion in metric capture coordinates."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import model_validator
from typing_extensions import Self

from physrealx.domain import ArtifactRef, DomainModel
from physrealx.errors import DomainValidationError


def joint_transform(kind: str, axis: Any, origin: Any, position: float) -> Any:
    """Rigid joint motion in a common frame; metres for slides, radians for hinges."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    axis, origin = np.asarray(axis, float), np.asarray(origin, float)
    if (
        kind not in {"slide", "hinge", "fixed"}
        or axis.shape != (3,)
        or origin.shape != (3,)
        or not np.isfinite(axis).all()
        or not np.isfinite(origin).all()
        or not math.isfinite(position)
        or not np.isclose(np.linalg.norm(axis), 1)
        or (kind == "fixed" and position != 0)
    ):
        raise DomainValidationError(
            "joint motion requires a finite unit axis, origin and valid position"
        )
    transform = np.eye(4)
    if kind == "slide":
        transform[:3, 3] = axis * position
    elif kind == "hinge":
        transform[:3, :3] = Rotation.from_rotvec(axis * position).as_matrix()
        transform[:3, 3] = origin - transform[:3, :3] @ origin
    return transform


class ReconstructedPart(DomainModel):
    """Axis/origin are in the initial capture frame; motion is radians or metres."""

    parent: str | None = None
    kind: Literal["fixed", "hinge", "slide"] = "fixed"
    axis: tuple[float, float, float] = (1, 0, 0)
    origin: tuple[float, float, float] = (0, 0, 0)
    span: float = 0

    @model_validator(mode="after")
    def valid_motion(self) -> Self:
        if not all(math.isfinite(v) for v in (*self.axis, *self.origin, self.span)):
            raise ValueError("predicted joint contains non-finite motion")
        if math.hypot(*self.axis) < 1e-12:
            raise ValueError("predicted joint axis must be nonzero")
        if self.kind == "fixed" and self.span != 0:
            raise ValueError("fixed part cannot have nonzero motion")
        return self


class ReconstructionPrediction(DomainModel):
    """All states remain available; only motion_pair permits deriving a missing end mesh.

    Unsegmented surfaces still support whole-object CD, never fabricated part/joint scores.
    Hierarchical motions use axes in the common initial capture frame.
    """

    parts: dict[str, ReconstructedPart]
    states: dict[str, dict[str, ArtifactRef]]
    motion_pair: tuple[str, str] | None = None
    segmented: bool = True

    @model_validator(mode="after")
    def valid_graph_and_states(self) -> Self:
        if not self.parts or not self.states or any(not key for key in (*self.parts, *self.states)):
            raise ValueError("reconstruction requires named parts and states")
        roots = [name for name, part in self.parts.items() if part.parent is None]
        if len(roots) != 1 or self.parts[roots[0]].kind != "fixed":
            raise ValueError("reconstruction requires one fixed root motion group")
        for name in self.parts:
            visited: set[str] = set()
            current: str | None = name
            while current is not None:
                if current in visited or current not in self.parts:
                    raise ValueError("predicted joint graph is disconnected or cyclic")
                visited.add(current)
                current = self.parts[current].parent
        if self.motion_pair is not None:
            start, end = self.motion_pair
            if start == end or not {start, end} <= self.states.keys() or not self.segmented:
                raise ValueError("motion_pair requires two distinct segmented states")
        elif any(part.kind != "fixed" for part in self.parts.values()):
            raise ValueError("predicted joint motion requires an explicit motion_pair")
        for state, meshes in self.states.items():
            if not set(meshes) <= self.parts.keys():
                raise ValueError("state mesh references unknown part")
            if set(meshes) != self.parts.keys() and (
                self.motion_pair is None or state != self.motion_pair[1]
            ):
                raise ValueError("every reconstruction part needs a state mesh")
        return self
