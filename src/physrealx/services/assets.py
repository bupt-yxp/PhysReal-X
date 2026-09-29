"""Asset ingestion, reconstruction and physical validation."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote, urlsplit

from physrealx.asset_validation import validate_asset
from physrealx.digital_twin import (
    AssetValidationProfile,
    AssetValidationReport,
    ObservationSetManifest,
    SourceInspection,
)
from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    JSONValue,
    MethodCapability,
    MethodSpec,
    OperationManifest,
    ResourceType,
    VersionedRef,
)
from physrealx.errors import (
    DomainValidationError,
    PhysRealXError,
    ResourceConflictError,
    ResourceNotFoundError,
    SourcePackageError,
)

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


@dataclass(frozen=True)
class SceneSmithAssetCandidate:
    """Registered PM assets for one SceneSmith scene candidate."""

    candidate_id: str
    assets: Mapping[str, VersionedRef]
    retrieval_score: float


@dataclass(frozen=True)
class SceneSmithAssetSelection:
    """Multi-candidate selection plus its immutable retrieval evidence."""

    candidates: tuple[SceneSmithAssetCandidate, ...]
    retrieval_report: ArtifactRef


class AssetsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: AssetSpec) -> VersionedRef:
        return self._platform._register(
            spec, spec.reference(), "application/vnd.physrealx.asset+json"
        )

    def get(self, reference: VersionedRef) -> AssetSpec:
        return self._platform._load_registered(reference, AssetSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.ASSET)

    def inspect_source(
        self,
        source: str | Path,
        *,
        importer: str = "artmodeling",
        config: Mapping[str, JSONValue] | None = None,
    ) -> SourceInspection:
        source = Path(source)
        adapter = self._platform.plugins.importer(importer)
        try:
            inspection = adapter.inspect(source, config=config)
        except PhysRealXError:
            raise
        except Exception as error:
            raise SourcePackageError(
                f"observation importer {adapter.descriptor.name!r} failed during inspection"
            ) from error
        return inspection.model_copy(update={"source_uri": self._source_locator(source)})

    def ingest(
        self,
        source: str | Path,
        *,
        resource_id: str,
        version: str,
        display_name: str,
        importer: str = "artmodeling",
        config: Mapping[str, JSONValue] | None = None,
    ) -> VersionedRef:
        source = Path(source)
        config = config or {}
        source_locator = self._source_locator(source)
        try:
            existing = self._platform.repository.get_resource(
                ResourceType.OBSERVATION_SET.value, resource_id, version
            )
        except ResourceNotFoundError:
            pass
        else:
            manifest = self._platform._load_registered(existing, ObservationSetManifest)
            if (
                not self._same_source(manifest.source_uri, source)
                or manifest.importer != importer
                or manifest.metadata.get("import_config") != dict(config)
            ):
                raise ResourceConflictError(
                    "observation version already exists with a different or unaudited import "
                    "configuration; use a new explicit version"
                )
            return existing
        adapter = self._platform.plugins.importer(importer)
        try:
            manifest = adapter.ingest(
                source,
                resource_id=resource_id,
                version=version,
                display_name=display_name,
                config=config,
                artifacts=self._platform.artifacts,
            )
        except PhysRealXError:
            raise
        except Exception as error:
            raise SourcePackageError(
                f"observation importer {adapter.descriptor.name!r} failed during ingestion"
            ) from error
        if (
            manifest.resource_id != resource_id
            or manifest.version != version
            or manifest.importer != adapter.descriptor.name
            or manifest.importer_version != adapter.descriptor.version
        ):
            raise DomainValidationError("observation importer returned mismatched identity")
        manifest = manifest.model_copy(update={"source_uri": source_locator})
        referenced = tuple(
            artifact for stream in manifest.streams for artifact in stream.artifacts
        ) + tuple(item.artifact for item in manifest.attachments)
        if any(not self._platform.artifacts.verify(item) for item in referenced):
            raise DomainValidationError("observation importer returned an unpublished artifact")
        return self._platform._register(
            manifest,
            manifest.reference(),
            "application/vnd.physrealx.observation-set+json",
        )

    def import_partnet_mobility(
        self,
        catalog: str | Path,
        *,
        model_id: str,
        resource_id: str,
        version: str = "1.1.0",
        display_name: str | None = None,
        target_extent_m: float,
    ) -> VersionedRef:
        """Register one PartNet-Mobility URDF as a metric canonical asset."""
        from physrealx.integrations.partnet_mobility import (
            PARTNET_IMPORT_REVISION,
            import_partnet_mobility_asset,
        )

        for reference in self.list():
            if (reference.resource_id, reference.version) != (resource_id, version):
                continue
            asset = self.get(reference)
            if (
                asset.metadata.get("source") != "PartNet-Mobility"
                or asset.metadata.get("import_revision") != PARTNET_IMPORT_REVISION
                or asset.metadata.get("model_id") != model_id
                or asset.metadata.get("target_extent_m") != target_extent_m
            ):
                raise DomainValidationError(
                    "registered PartNet-Mobility asset identity has different import settings"
                )
            return reference
        return self.register(
            import_partnet_mobility_asset(
                self._platform.artifacts,
                catalog,
                model_id=model_id,
                resource_id=resource_id,
                version=version,
                display_name=display_name,
                target_extent_m=target_extent_m,
            )
        )

    def retrieve_partnet_mobility(
        self,
        catalog: str | Path,
        *,
        scene_type: str,
        instruction: str,
        method: str,
        seed: int = 0,
        candidate_count: int = 3,
        version: str = "1.2.0",
    ) -> SceneSmithAssetSelection:
        """Run SceneSmith full-library retrieval and register viable PM candidates."""
        from typing import cast

        from physrealx.integrations.partnet_mobility import (
            PartNetSceneType,
            retrieve_partnet_scene_assets,
        )

        canonical_method = method.strip().casefold()
        if canonical_method != "scenesmith":
            raise DomainValidationError(
                "PartNet-Mobility scene retrieval currently requires method='SceneSmith'"
            )
        if type(candidate_count) is not int or not 1 <= candidate_count <= 8:
            raise DomainValidationError("candidate_count must be between 1 and 8")
        retrieval = retrieve_partnet_scene_assets(
            catalog,
            scene_type=cast(PartNetSceneType, scene_type),
            instruction=instruction,
            seed=seed,
            candidate_count=min(16, candidate_count * 3),
        )
        registered: list[SceneSmithAssetCandidate] = []
        rejected: list[dict[str, JSONValue]] = []
        for candidate in retrieval.candidates:
            assets: dict[str, VersionedRef] = {}
            try:
                for item in candidate.assets:
                    assets[item.name] = self.import_partnet_mobility(
                        catalog,
                        model_id=item.model_id,
                        resource_id=(
                            f"partnet.{scene_type}.{item.name}.{item.model_id}"
                        ),
                        version=version,
                        display_name=item.category,
                        target_extent_m=item.target_extent_m,
                    )
            except (DomainValidationError, OSError, ValueError) as error:
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "stage": "canonical_import",
                        "reason": str(error),
                    }
                )
                continue
            registered.append(
                SceneSmithAssetCandidate(
                    candidate_id=candidate.candidate_id,
                    assets=assets,
                    retrieval_score=candidate.score,
                )
            )
            if len(registered) == candidate_count:
                break
        if len(registered) < candidate_count:
            raise DomainValidationError(
                "SceneSmith PM retrieval did not yield enough importable candidates: "
                f"required {candidate_count}, found {len(registered)}"
            )
        selected_ids = {item.candidate_id for item in registered}
        report = self._platform.artifacts.put_json(
            {
                "schema": "physrealx.scenesmith-retrieval-report/1",
                "method": "scenesmith",
                "upstream": {
                    "source": "https://github.com/nepfaff/scenesmith",
                    "revision": "67cc408fd38334b4a926efef45e284302ed5055b",
                    "license": "MIT",
                },
                "scene_type": scene_type,
                "instruction": instruction.strip(),
                "library": {
                    "evaluated_models": retrieval.evaluated_models,
                    "valid_models": retrieval.valid_models,
                    "rejected_models": retrieval.rejected_models,
                },
                "selected": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "score": candidate.score,
                        "assets": [
                            {
                                "name": item.name,
                                "category": item.category,
                                "model_id": item.model_id,
                                "score": item.score,
                                "score_components": item.score_components,
                            }
                            for item in candidate.assets
                        ],
                    }
                    for candidate in retrieval.candidates
                    if candidate.candidate_id in selected_ids
                ],
                "rejected_candidates": rejected,
            }
        )
        return SceneSmithAssetSelection(
            candidates=tuple(registered),
            retrieval_report=report,
        )

    def import_mujoco_model(
        self,
        source: str | Path,
        *,
        resource_id: str,
        version: str = "1.0.0",
        display_name: str,
        source_name: str,
        source_revision: str,
        license_name: str,
    ) -> VersionedRef:
        """Register a local open-source MJCF robot as a canonical asset."""
        identity = {
            "source": source_name,
            "source_revision": source_revision,
            "source_model": Path(source).name,
            "license": license_name,
        }
        for reference in self.list():
            if (reference.resource_id, reference.version) != (resource_id, version):
                continue
            asset = self.get(reference)
            if any(asset.metadata.get(key) != value for key, value in identity.items()):
                raise DomainValidationError(
                    "registered MuJoCo asset identity has different source provenance"
                )
            return reference
        from physrealx.integrations.mujoco_model import import_mujoco_model_asset

        return self.register(
            import_mujoco_model_asset(
                self._platform.artifacts,
                source,
                resource_id=resource_id,
                version=version,
                display_name=display_name,
                source_name=source_name,
                source_revision=source_revision,
                license_name=license_name,
            )
        )

    def create_room_shell(
        self,
        *,
        resource_id: str,
        version: str = "1.0.0",
        width_m: float,
        depth_m: float,
        height_m: float,
    ) -> VersionedRef:
        """Register a metric floor and two-wall room environment."""
        from physrealx.integrations.scene_assets import room_shell_asset

        return self.register(
            room_shell_asset(
                resource_id=resource_id,
                version=version,
                width_m=width_m,
                depth_m=depth_m,
                height_m=height_m,
            )
        )

    def get_observations(self, reference: VersionedRef) -> ObservationSetManifest:
        return self._platform._load_registered(reference, ObservationSetManifest)

    def resolve_observation_source(self, reference: VersionedRef) -> Path:
        """Resolve a registered capture's existing local source directory."""
        if reference.resource_type is not ResourceType.OBSERVATION_SET:
            raise DomainValidationError(
                "resolve_observation_source requires an observation-set reference"
            )
        observations = self.get_observations(reference)
        source = self._source_path(observations.source_uri)
        if source is None:
            raise SourcePackageError(
                f"observation source is missing or cannot be resolved locally: "
                f"{observations.source_uri!r}"
            )
        return source

    def list_observations(self) -> builtins.list[VersionedRef]:
        return self._platform._list_resources(ResourceType.OBSERVATION_SET)

    def import_reference(
        self,
        observations: VersionedRef,
        *,
        usd_path: str,
        root_link: str | None = None,
        moving_links: tuple[str, ...] | None = None,
    ) -> ArtifactRef:
        """Import supplied USD geometry and kinematics with explicit model provenance."""
        from physrealx.integrations.usd_reference import import_usd_reference

        return import_usd_reference(
            self.get_observations(observations),
            self._platform.artifacts,
            usd_path=usd_path,
            root_link=root_link,
            moving_links=moving_links,
        )

    def prepare_rgbd(
        self,
        observations: VersionedRef,
        camera_rig: ArtifactRef,
        *,
        stride: int = 4,
        max_frames_per_stream: int | None = 8,
    ) -> ArtifactRef:
        """Decode registered RGB-D views; None selects all within the 512-view safety limit."""
        from physrealx.integrations.capture_rgbd import prepare_rgbd_frames

        return prepare_rgbd_frames(
            self.get_observations(observations),
            camera_rig,
            self._platform.artifacts,
            stride=stride,
            max_frames_per_stream=max_frames_per_stream,
        )

    def recover_cameras(
        self,
        observations: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Recover camera intrinsics, poses and an unscaled sparse reconstruction."""

        if observations.resource_type is not ResourceType.OBSERVATION_SET:
            raise DomainValidationError(
                "assets.recover_cameras requires an observation-set reference"
            )
        return self._platform.methods.execute(
            MethodCapability.CAMERA_RECOVER,
            method,
            inputs={"observations": observations},
            config=config,
        )

    def align_cameras_metric(
        self,
        camera_rig: ArtifactRef,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Resolve a recovered camera rig into metres from held-out metrology evidence."""

        return self._platform.methods.execute(
            MethodCapability.CAMERA_ALIGN_METRIC,
            method,
            inputs={"camera_rig": camera_rig, "evidence": evidence},
            config=config,
        )

    def enhance_depth(
        self,
        rgbd: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Complete calibrated metric RGB-D while retaining held-out quality evidence."""

        return self._platform.methods.execute(
            MethodCapability.DEPTH_ENHANCE,
            method,
            inputs={"rgbd": rgbd},
            config=config,
        )

    def reconstruct_geometry(
        self,
        source: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Reconstruct metric per-link visual geometry from a validated source bundle."""

        return self._platform.methods.execute(
            MethodCapability.GEOMETRY_RECONSTRUCT,
            method,
            inputs={"source": source},
            config=config,
        )

    def evaluate_reconstruction(self, source: ArtifactRef, *, seed: int = 2026) -> ArtifactRef:
        """Evaluate any normalized reconstruction with the fixed DTArt metric protocol."""
        from physrealx.integrations._artifacts import read_json
        from physrealx.integrations.reconstruction_evaluation import (
            ReconstructionEvaluationConfig,
            ReconstructionEvaluationInput,
            evaluate_reconstruction,
        )

        request = ReconstructionEvaluationInput.model_validate(
            read_json(
                self._platform.artifacts, source, "physrealx.reconstruction-evaluation-input/1"
            )
        )
        report = evaluate_reconstruction(
            self._platform.artifacts, request, ReconstructionEvaluationConfig(seed=seed)
        )
        return self._platform.artifacts.put_json(report)

    def preview_reconstruction(
        self, evaluation: ArtifactRef, *, frames: int = 24, fps: int = 12
    ) -> ArtifactRef:
        """Animate appearance, reconstructed geometry and reference geometry together."""
        from physrealx.integrations.reconstruction_preview import preview_reconstruction

        return preview_reconstruction(self._platform.artifacts, evaluation, frames=frames, fps=fps)

    def preview_physical_asset(
        self,
        assembly: ArtifactRef,
        *,
        frames: int = 24,
        fps: int = 12,
        rejected_refinement: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Animate assembled visual and collision geometry through declared joint limits."""
        from physrealx.integrations.physical_preview import preview_physical_asset

        return preview_physical_asset(
            self._platform.artifacts,
            assembly,
            frames=frames,
            fps=fps,
            rejected_refinement=rejected_refinement,
        )

    def register_reconstruction_geometry(
        self,
        reconstruction: ArtifactRef,
        registration: ArtifactRef,
    ) -> ArtifactRef:
        """Preserve reconstructed surfaces while binding the input registered frames."""
        from physrealx.integrations.reconstructed_asset import register_reconstruction_geometry

        return register_reconstruction_geometry(
            self._platform.artifacts, reconstruction, registration
        )

    def validate(
        self,
        asset: AssetSpec | VersionedRef,
        *,
        profile: AssetValidationProfile | str = AssetValidationProfile.DIGITAL_TWIN,
    ) -> AssetValidationReport:
        target = asset if isinstance(asset, AssetSpec) else self.get(asset)
        try:
            selected_profile = AssetValidationProfile(profile)
        except ValueError as error:
            raise DomainValidationError(f"unknown asset validation profile: {profile!r}") from error
        return validate_asset(target, selected_profile, self._platform.artifacts)

    def preview_physics_report(
        self,
        prior: ArtifactRef,
        feasibility: ArtifactRef,
        readiness: ArtifactRef,
        initial_feasibility: ArtifactRef | None = None,
    ) -> ArtifactRef:
        from physrealx.integrations.physical_preview import preview_physics_report

        return preview_physics_report(
            self._platform.artifacts,
            prior,
            feasibility,
            readiness,
            initial_feasibility,
        )

    def preview_physics_video(
        self, directory: Path, *, frames: int = 24, fps: int = 8
    ) -> ArtifactRef:
        """Animate synchronized RGB, depth and manually reviewed object/link masks."""
        from physrealx.integrations.physical_preview import preview_physics_video

        return preview_physics_video(self._platform.artifacts, directory, frames=frames, fps=fps)

    def articulate(
        self,
        pair: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.ARTICULATION_RECOVER,
            method,
            inputs={"pair": pair},
            config=config,
        )

    def build_collision(
        self,
        asset: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.COLLISION_BUILD,
            method,
            inputs={"asset": asset},
            config=config,
        )

    def calibrate_physics(
        self,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.PHYSICS_CALIBRATE,
            method,
            inputs={"evidence": evidence},
            config=config,
        )

    def predict_physics(
        self,
        reference: ArtifactRef,
        registration: ArtifactRef,
        geometry: ArtifactRef,
        images: tuple[ArtifactRef, ...],
        *,
        python: str,
        model: str,
        revision: str,
        timeout_seconds: int = 600,
        cached_prior: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Initialize per-body and joint priors with a pinned local VLM implementation."""
        from physrealx.integrations.physical_estimation import predict_physics

        return predict_physics(
            self._platform.artifacts,
            reference,
            registration,
            geometry,
            images,
            python=python,
            model=model,
            revision=revision,
            timeout_seconds=timeout_seconds,
            cached_prior=cached_prior,
        )

    def refine_video_geometry(
        self,
        evidence: ArtifactRef,
        *,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Refine observed surfaces without changing the start/end joint range."""
        from physrealx.integrations.video_geometry import (
            GeometryRefinementConfig,
            refine_video_geometry,
        )

        return refine_video_geometry(
            self._platform.artifacts,
            evidence,
            config=GeometryRefinementConfig.model_validate(config or {}),
        )

    def check_physics_feasibility(
        self,
        assembly: ArtifactRef,
        *,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Audit collision clearance and dynamics; does not certify measured physical truth."""
        from physrealx.integrations.physical_feasibility import FeasibilityConfig, check_feasibility

        return check_feasibility(
            self._platform.artifacts,
            assembly,
            config=FeasibilityConfig.model_validate(config or {}),
        )

    def refine_articulated_collision(
        self,
        assembly: ArtifactRef,
        feasibility: ArtifactRef,
        *,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Remove collision-proxy ghosts without changing visuals or joint limits."""
        from physrealx.integrations.articulated_geometry import (
            ArticulatedCollisionRefinementConfig,
            refine_articulated_collision,
        )

        return refine_articulated_collision(
            self._platform.artifacts,
            assembly,
            feasibility,
            config=ArticulatedCollisionRefinementConfig.model_validate(config or {}),
        )

    def validate_physics(
        self,
        package: ArtifactRef,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Validate packaged physics against an independent measured trajectory."""
        return self._platform.methods.execute(
            MethodCapability.ASSET_VALIDATE,
            method,
            inputs={"package": package, "evidence": evidence},
            config=config,
        )

    def assemble(
        self,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Assemble all links and joints from explicit immutable measured evidence."""
        return self._platform.methods.execute(
            MethodCapability.ASSET_ASSEMBLE,
            method,
            inputs={"evidence": evidence},
            config=config,
        )

    def package(
        self,
        asset: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.ASSET_PACKAGE,
            method,
            inputs={"asset": asset},
            config=config,
        )

    def _source_locator(self, source: Path) -> str:
        """Return a stable project-relative URI when the source belongs to this checkout."""

        resolved = source.expanduser().resolve(strict=True)
        if self._platform.workspace.name == ".physrealx":
            try:
                relative = resolved.relative_to(self._platform.workspace.parent)
            except ValueError:
                pass
            else:
                return f"project:///{quote(relative.as_posix(), safe='/')}"
        return resolved.as_uri()

    def _source_path(self, locator: str) -> Path | None:
        """Resolve supported source locators without making their spelling an identity."""

        try:
            parsed = urlsplit(locator)
            if parsed.query or parsed.fragment:
                return None
            project_root = None
            if (
                parsed.scheme == "project"
                and not parsed.netloc
                and self._platform.workspace.name == ".physrealx"
            ):
                relative = PurePosixPath(unquote(parsed.path).lstrip("/"))
                if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                    return None
                project_root = self._platform.workspace.parent.resolve(strict=True)
                candidate = project_root.joinpath(*relative.parts)
            elif parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
                candidate = Path(unquote(parsed.path))
                if not candidate.is_absolute():
                    return None
            else:
                return None
            resolved = candidate.resolve(strict=True)
            if project_root is not None:
                resolved.relative_to(project_root)
            return resolved
        except (OSError, RuntimeError, ValueError):
            return None

    def _same_source(self, locator: str, source: Path) -> bool:
        stored = self._source_path(locator)
        try:
            requested = source.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return stored == requested
