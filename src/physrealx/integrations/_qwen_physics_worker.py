"""Local optional Transformers worker; keeps torch/CUDA outside the platform process."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path


def model_file_fingerprints(model: str) -> dict[str, str]:
    """Fingerprint local inference inputs, including tokenizer and chat templates."""
    root = Path(model)
    if not root.is_dir():
        return {}
    fingerprints = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[relative.as_posix()] = digest.hexdigest()
    return fingerprints


def _inference_memory_budget(available: dict[int | str, int]) -> dict[int | str, int]:
    """Leave runtime space beyond the static weights placed by Accelerate."""
    gib = 1024**3
    budget: dict[int | str, int] = {}
    primary_assigned = False
    for device in sorted(key for key in available if isinstance(key, int)):
        free = available[device]
        # KV cache and temporary activations need space on every participating
        # GPU. Inputs, vision prefill and CPU-offloaded layers also use the first.
        reserve = max(2 * gib, free // 5) + (0 if primary_assigned else 2 * gib)
        if free > reserve:
            budget[device] = free - reserve
            primary_assigned = True
    if "cpu" in available:
        free = available["cpu"]
        remaining = free - max(2 * gib, free // 10)
        if remaining > 0:
            budget["cpu"] = remaining
    if not budget:
        raise MemoryError("insufficient available memory after reserving inference workspace")
    return budget


def main() -> None:
    import torch
    from PIL import Image

    transformers = import_module("transformers")

    request = json.loads(Path(sys.argv[1]).read_text())
    model_files = model_file_fingerprints(request["model"])
    processor = transformers.AutoProcessor.from_pretrained(
        request["model"],
        revision=request["revision"],
        local_files_only=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    # get_max_memory initializes CUDA before querying free memory; use currently
    # available capacity, so other processes' allocations are never counted as ours.
    memory_budget = _inference_memory_budget(import_module("accelerate.utils").get_max_memory())
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        request["model"],
        revision=request["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=memory_budget,
        attn_implementation="sdpa",
    )
    images = [Image.open(path).convert("RGB") for path in request["images"]]
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in images),
                {"type": "text", "text": request["prompt"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    answer = processor.batch_decode(
        tokens[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    Path(sys.argv[2]).write_text(
        json.dumps(
            {
                "text": answer,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "model_class": type(model).__name__,
                "local_model_sha256": model_files,
                "weight_memory_budget_bytes": memory_budget,
                "device_map": model.hf_device_map,
            }
        )
    )


if __name__ == "__main__":
    main()
