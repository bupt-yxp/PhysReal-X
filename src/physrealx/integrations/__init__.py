"""First-party adapters for algorithms vendored in the PhysReal-X repository."""

from physrealx.integrations.artgs import (
    ARTGS_DESCRIPTOR,
    ArtGSMethodAdapter,
    ArtGSRunConfig,
)
from physrealx.integrations.colmap import (
    COLMAP_DESCRIPTOR,
    ColmapMethodAdapter,
    ColmapRunConfig,
    create_colmap_adapter,
)
from physrealx.integrations.dtart import DTART_DESCRIPTOR, DTArtMethodAdapter, DTArtRunConfig

__all__ = [
    "ARTGS_DESCRIPTOR",
    "COLMAP_DESCRIPTOR",
    "DTART_DESCRIPTOR",
    "ArtGSMethodAdapter",
    "ArtGSRunConfig",
    "ColmapMethodAdapter",
    "ColmapRunConfig",
    "DTArtMethodAdapter",
    "DTArtRunConfig",
    "create_colmap_adapter",
]
