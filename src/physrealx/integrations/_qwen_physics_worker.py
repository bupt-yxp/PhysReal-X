"""Local optional Transformers worker; keeps torch/CUDA outside the platform process."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import import_module
from pathlib import Path


def main() -> None:
    import torch
    from PIL import Image

    transformers = import_module("transformers")

    request = json.loads(Path(sys.argv[1]).read_text())
    model_files = {}
    model_root = Path(request["model"])
    if model_root.is_dir():
        for path in sorted(model_root.iterdir()):
            if path.suffix not in {".json", ".safetensors"}:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            model_files[path.name] = digest.hexdigest()
    processor = transformers.AutoProcessor.from_pretrained(
        request["model"],
        revision=request["revision"],
        local_files_only=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        request["model"],
        revision=request["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
            }
        )
    )


if __name__ == "__main__":
    main()
