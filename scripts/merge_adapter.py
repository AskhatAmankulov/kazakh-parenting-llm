"""Merge the LoRA adapter into full-precision base weights (needed for GGUF / vLLM without LoRA support).

    python scripts/merge_adapter.py                                   # -> artifacts/merged
    python scripts/merge_adapter.py --push yourname/qwen3-4b-kk-ru-parenting   # optional HF Hub upload

Note: merge into fp16 weights, NOT into the 4-bit model — merging into quantised weights loses precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kpl.config import base_parser, load_config, resolve  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--out", default="artifacts/merged")
    ap.add_argument("--push", default=None, help="HF Hub repo id (needs HF_TOKEN)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg.model.base, torch_dtype=torch.float16, device_map="cpu",
                                                low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, str(resolve(cfg.paths.adapter_dir)))
    model = model.merge_and_unload()
    out = resolve(args.out)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(str(resolve(cfg.paths.adapter_dir))).save_pretrained(out)
    print(f"merged model -> {out}")
    if args.push:
        model.push_to_hub(args.push, private=True)
        AutoTokenizer.from_pretrained(str(out)).push_to_hub(args.push, private=True)
        print(f"pushed to https://huggingface.co/{args.push} (private)")


if __name__ == "__main__":
    main()
