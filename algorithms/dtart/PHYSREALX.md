# PhysReal-X integration notes

Upstream: NVlabs/DigitalTwinArt, commit `1a48b402e4bf4bb7731296e8e230f0db3d86fe4f`.
The retained source is subject to the accompanying NVIDIA research license.
Large demo media and the upstream environment creation script are not vendored.

Local changes:
- Compile the two CUDA extensions with C++17 and the current PhysReal-X Torch ABI.
- Accept arbitrary run directories.
- Reject malformed correspondence files instead of silently discarding them.
- Import CUDA extensions exclusively as `mycuda.common` / `mycuda.gridencoder`.
  The upstream mixed imports loaded the same binary twice and caused an invalid
  free at interpreter shutdown on the current environment. Native import/exit
  now succeeds without suppressing errors or bypassing shutdown.
- Use `shutil.copyfile` for checkpoints, including paths containing spaces.
- Replace the upstream author-specific debug output path with a relative default.
- RGB-D rectification is shared with ArtGS in the platform RGB-D module.
- LoFTR uses Kornia with verified outdoor weights and object-mask filtering.

Native checkpoint and both-state mesh reload paths are covered by the DTArt runtime
acceptance test. The default oven4 Notebook currently selects ArtGS; selecting DTArt uses
the same reconstruction, registration, evaluation and preview contracts. See
`docs/acceptance.md` for the current executed method and evidence boundary.
