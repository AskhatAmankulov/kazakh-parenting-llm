"""Model loading + generation behind one interface.

  HFChat     — local open-weights model (optionally 4-bit, optionally with a LoRA adapter)
  OpenAIChat — any OpenAI-compatible endpoint (OpenAI, Gemini, Groq, vLLM server, llama.cpp server)

Both expose .chat(list_of_conversations, **gen_kwargs) -> list[str].
"""
from __future__ import annotations

import os
import time
from typing import Any

Messages = list[dict[str, str]]


def bnb_config(compute_dtype: str = "float16", quant_type: str = "nf4", double_quant: bool = True):
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
    )


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, load_in_4bit: bool = True, compute_dtype: str = "float16",
               adapter: str | None = None, quant_type: str = "nf4", double_quant: bool = True,
               device_map: Any = "auto"):
    import torch
    from transformers import AutoModelForCausalLM

    kw: dict[str, Any] = {"device_map": device_map}
    if load_in_4bit:
        kw["quantization_config"] = bnb_config(compute_dtype, quant_type, double_quant)
    dtype = getattr(torch, compute_dtype)
    try:  # transformers >= 4.56 / 5.x: `dtype`; older versions: `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def apply_template(tok, messages: Messages, add_generation_prompt: bool) -> str:
    # enable_thinking=False is ignored by templates that don't know it (Qwen2.5, Llama),
    # and switches off <think> blocks for hybrid Qwen3 checkpoints.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


class HFChat:
    def __init__(self, base: str, adapter: str | None = None, load_in_4bit: bool = True,
                 compute_dtype: str = "float16", batch_size: int = 8, model=None, tokenizer=None):
        self.tok = tokenizer or load_tokenizer(base)
        self.tok.padding_side = "left"
        self.model = model or load_model(base, load_in_4bit, compute_dtype, adapter)
        self.batch_size = batch_size

    def set_adapter_enabled(self, enabled: bool):
        """Toggle a loaded LoRA on/off so base and fine-tuned runs share one copy of the weights."""
        if hasattr(self.model, "disable_adapter_layers"):
            (self.model.enable_adapter_layers if enabled else self.model.disable_adapter_layers)()

    def chat(self, convs: list[Messages], max_new_tokens: int = 512, temperature: float = 0.3,
             top_p: float = 0.9, repetition_penalty: float = 1.05, **_) -> list[str]:
        import torch

        outs: list[str] = []
        for i in range(0, len(convs), self.batch_size):
            batch = [apply_template(self.tok, m, True) for m in convs[i : i + self.batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True))
        return outs


class OpenAIChat:
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "TEACHER_API_KEY",
                 max_retries: int = 6, rpm: int = 0):
        from openai import OpenAI

        key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"set ${api_key_env} (or $OPENAI_API_KEY) for the teacher/judge endpoint")
        self.client = OpenAI(api_key=key, base_url=base_url or None)
        self.model = model
        self.max_retries = max_retries
        self.min_interval = 60.0 / rpm if rpm else 0.0   # client-side throttle for free-tier rate limits
        self._last = 0.0

    def chat(self, convs: list[Messages], max_new_tokens: int = 1024, temperature: float = 0.7,
             top_p: float = 0.95, json_mode: bool = False, **_) -> list[str]:
        outs = []
        for m in convs:
            kw: dict[str, Any] = dict(model=self.model, messages=m, max_tokens=max_new_tokens,
                                      temperature=temperature, top_p=top_p)
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            for attempt in range(self.max_retries):
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
                try:
                    r = self.client.chat.completions.create(**kw)
                    outs.append((r.choices[0].message.content or "").strip())
                    break
                except Exception as e:  # rate limits on free tiers are routine
                    if attempt == self.max_retries - 1:
                        raise
                    wait = 2 ** attempt * 3
                    print(f"[teacher] {type(e).__name__}: {e} — retry in {wait}s")
                    time.sleep(wait)
        return outs


def teacher_from_cfg(cfg) -> HFChat | OpenAIChat:
    t = cfg["teacher"]
    if t["backend"] == "openai":
        return OpenAIChat(t["model"], t.get("base_url"), t.get("api_key_env", "TEACHER_API_KEY"), rpm=t.get("rpm", 0))
    if t["backend"] == "hf":
        return HFChat(t["hf_model"], load_in_4bit=True, compute_dtype=cfg["model"]["compute_dtype"], batch_size=4)
    raise ValueError(f"unknown teacher backend {t['backend']!r}")
