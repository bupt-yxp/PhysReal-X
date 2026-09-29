# SceneSmith integration

PhysReal-X pins SceneSmith commit `67cc408fd38334b4a926efef45e284302ed5055b` and adapts
its stage contracts rather than vendoring its complete Blender, Drake, SAM3D and agent runtime.

The `scenesmith` method scans the full local PartNet-Mobility library, filters invalid source
packages, ranks assets using task semantics, joint function, geometry completeness and requested
scale, and returns diversified candidates. The scene adapter applies support, collision, room and
interaction-corridor constraints. PhysReal-X then runs its canonical robot placement and
MuJoCo/Isaac Sim execution gates, preserving both the selected candidate and rejected evidence.

This adaptation does not claim SceneSmith benchmark scores and does not run its optional open-set
3D generation backends. The immutable upstream source, revision and license are recorded in every
retrieval and layout report.
