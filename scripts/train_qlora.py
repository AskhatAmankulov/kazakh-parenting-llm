"""QLoRA fine-tuning (4-bit NF4 base + LoRA adapters on all linear projections).

    python scripts/train_qlora.py
    python scripts/train_qlora.py --set model.base=Qwen/Qwen2.5-7B-Instruct train.epochs=2
    python scripts/train_qlora.py --set train.max_steps=10      # smoke test

Fits on a single 16 GB T4 for 4B-8B models at max_seq_len 1536.
Outputs: adapter + tokenizer, train_log.json (loss curves), loss.png, run_config.yaml.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from kpl.config import base_parser, load_config, resolve, set_seed  # noqa: E402
from kpl.data import read_jsonl  # noqa: E402
from kpl.llm import load_model, load_tokenizer  # noqa: E402
from kpl.sft import PadCollator, tokenize_example  # noqa: E402


def tokenize_split(tok, rows, max_len):
    out, dropped = [], 0
    for r in rows:
        ex = tokenize_example(tok, r["messages"], max_len)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


def main():
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    t, m, l = cfg.train, cfg.model, cfg.lora
    out_dir = resolve(cfg.paths.adapter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(m.base)
    tok.padding_side = "right"
    pdir = resolve(cfg.paths.processed_dir)
    train_rows, val_rows = read_jsonl(pdir / "train.jsonl"), read_jsonl(pdir / "val.jsonl")
    if not train_rows:
        sys.exit("no training data — run scripts/prepare_dataset.py first")
    train_ds, d1 = tokenize_split(tok, train_rows, t.max_seq_len)
    val_ds, d2 = tokenize_split(tok, val_rows, t.max_seq_len)
    lens = sorted(len(x["input_ids"]) for x in train_ds)
    sup = sum(sum(1 for y in x["labels"] if y != -100) for x in train_ds)
    print(f"train={len(train_ds)} (dropped {d1} > {t.max_seq_len} tok) val={len(val_ds)} (dropped {d2})")
    print(f"seq len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]} | supervised tokens={sup}")

    # single GPU for training: sharding a 4B 4-bit model over 2xT4 only adds cross-device traffic
    model = load_model(m.base, m.load_in_4bit, m.compute_dtype, quant_type=m.bnb_quant_type,
                       double_quant=m.double_quant, device_map={"": 0} if torch.cuda.is_available() else None)
    model.config.use_cache = False
    if m.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=t.gradient_checkpointing)
    elif t.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(r=l.r, lora_alpha=l.alpha, lora_dropout=l.dropout, target_modules=list(l.target_modules),
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    fp16 = m.compute_dtype == "float16"
    ta = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=t.epochs,
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t.per_device_batch,
        per_device_eval_batch_size=t.per_device_batch,
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.lr,
        lr_scheduler_type=t.scheduler,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        logging_steps=t.logging_steps,
        eval_steps=t.eval_steps,
        save_steps=t.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        bf16=not fp16,
        optim="paged_adamw_8bit" if m.load_in_4bit else "adamw_torch",
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
    )
    try:
        args_tr = TrainingArguments(eval_strategy="steps", save_strategy="steps", **ta)
    except TypeError:  # transformers < 4.41
        args_tr = TrainingArguments(evaluation_strategy="steps", save_strategy="steps", **ta)

    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=PadCollator(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    metrics = trainer.evaluate()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "run_config.yaml").write_text(yaml.safe_dump(dict(cfg), allow_unicode=True), encoding="utf-8")
    log = {"history": trainer.state.log_history, "final_eval": metrics, "train_minutes": round(minutes, 1),
           "n_train": len(train_ds), "n_val": len(val_ds),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0}
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    plot_loss(trainer.state.log_history, out_dir / "loss.png")
    print(f"done in {minutes:.1f} min | eval_loss={metrics.get('eval_loss'):.4f} | adapter -> {out_dir}")


def plot_loss(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train", color="#4C78A8", lw=1.6)
    if ev:
        ax.plot(*zip(*ev), label="validation", color="#F58518", lw=2, marker="o", ms=4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (answer tokens only)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
