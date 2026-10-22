"""SFT tokenisation with completion-only loss.

Loss is computed ONLY on the assistant answer. System prompt, child profile, retrieved context
and the question are masked with -100 — otherwise the model spends capacity learning to
reproduce the RAG passages instead of learning to answer.

Implemented by hand (prompt ids + completion ids) rather than via a TRL collator, so it works
identically with any chat template and any library version.
"""
from __future__ import annotations

from typing import Any

from .llm import apply_template

IGNORE = -100


def tokenize_example(tok, messages: list[dict[str, str]], max_len: int) -> dict[str, Any] | None:
    assert messages[-1]["role"] == "assistant", "last message must be the target answer"
    prompt_text = apply_template(tok, messages[:-1], add_generation_prompt=True)
    full_text = apply_template(tok, messages, add_generation_prompt=False)
    if full_text.startswith(prompt_text):
        completion_text = full_text[len(prompt_text):]
    else:
        # Some templates (e.g. hybrid-thinking Qwen3) add an empty <think></think> block only to the
        # generation prompt. Train exactly what inference will see: prompt + answer + end-of-turn.
        completion_text = messages[-1]["content"].strip() + (tok.eos_token or "")

    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion_text, add_special_tokens=False)["input_ids"]
    if tok.eos_token_id is not None and (not c_ids or c_ids[-1] != tok.eos_token_id):
        # most templates end the turn with their own end token; make sure generation can stop
        if tok.eos_token not in completion_text:
            c_ids = c_ids + [tok.eos_token_id]

    if len(p_ids) + len(c_ids) > max_len:
        return None  # dropped and counted by the caller; truncating answers teaches the model to stop mid-sentence
    return {
        "input_ids": p_ids + c_ids,
        "attention_mask": [1] * (len(p_ids) + len(c_ids)),
        "labels": [IGNORE] * len(p_ids) + c_ids,
    }


class PadCollator:
    """Right-pads input_ids / attention_mask / labels to the longest item in the batch."""

    def __init__(self, pad_id: int, multiple_of: int = 8):
        self.pad_id, self.mult = pad_id, multiple_of

    def __call__(self, batch: list[dict[str, list[int]]]):
        import torch

        L = max(len(b["input_ids"]) for b in batch)
        L = (L + self.mult - 1) // self.mult * self.mult
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            pad = L - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["labels"].append(b["labels"] + [IGNORE] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}
