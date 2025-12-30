from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"


def _default_image_path() -> Path:
    return Path(__file__).resolve().parent / "images" / "1.png"


def run_ocr(*, model_id: str, image_path: Path, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = "<image>\nFree OCR."

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    inputs = None

    # Common multimodal processors support `processor(text=..., images=...)`
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    except TypeError:
        # Some models (e.g., chat-template based) require building an input string first.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Free OCR."},
                ],
            }
        ]
        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=text, images=image, return_tensors="pt")

    if inputs is None:
        raise RuntimeError(
            "Unable to build model inputs for this processor/model combo. "
            "Try updating transformers or using a different model revision."
        )

    # Ensure tensors are on CPU
    inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )

    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    else:
        text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)

    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek OCR locally (CPU-only).")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=_default_image_path(),
        help="Path to image (default: src/images/1.png)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens (CPU: keep modest; default: 1024)",
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    text = run_ocr(model_id=args.model, image_path=image_path, max_new_tokens=args.max_new_tokens)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())