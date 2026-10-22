"""Answer the held-out eval set with up to four system variants:

    base       — base model, no retrieval
    base_rag   — base model + hybrid RAG context
    ft         — QLoRA fine-tuned model, no retrieval
    ft_rag     — fine-tuned + RAG   (the production configuration)

The base weights are loaded once; the LoRA adapter is switched on/off between variants.
Decoding is greedy so runs are reproducible.

    python scripts/generate_answers.py
    python scripts/generate_answers.py --variants base base_rag      # before training
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kpl.config import base_parser, load_config, resolve, set_seed  # noqa: E402
from kpl.data import load_eval, write_jsonl  # noqa: E402
from kpl.llm import HFChat  # noqa: E402
from kpl.prompts import build_messages  # noqa: E402
from kpl.rag import HybridRetriever  # noqa: E402

ALL = ["base", "base_rag", "ft", "ft_rag"]


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--variants", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)

    items = load_eval(resolve(cfg.paths.eval_set))
    if args.limit:
        items = items[: args.limit]
    need_ft = any(v.startswith("ft") for v in args.variants)
    adapter = str(resolve(cfg.paths.adapter_dir)) if need_ft else None
    if need_ft and not Path(adapter, "adapter_config.json").exists():
        sys.exit(f"no adapter at {adapter} — train first or drop the ft variants")

    idx = resolve(cfg.paths.index_dir)
    dense = (idx / "embeddings.npy").exists()
    retriever = HybridRetriever.load(idx, cfg.rag.embed_model if dense else None, rrf_k=cfg.rag.rrf_k,
                                     bm25_weight=cfg.rag.bm25_weight, dense_weight=cfg.rag.dense_weight)
    hits = {
        it["id"]: retriever.search(it["question"], lang=it["lang"], k=cfg.rag.top_k,
                                   age_months=it["profile"].get("age_months"))
        for it in items
    }

    m = cfg.model
    chat = HFChat(m.base, adapter=adapter, load_in_4bit=m.load_in_4bit, compute_dtype=m.compute_dtype,
                  batch_size=args.batch_size)
    g = dict(cfg.generation)
    g["temperature"] = 0.0

    out_dir = resolve(cfg.paths.generations_dir)
    for v in args.variants:
        use_rag = v.endswith("_rag")
        if need_ft:
            chat.set_adapter_enabled(v.startswith("ft"))
        convs = [
            build_messages(it["question"], it["lang"], it["profile"],
                           [h.chunk for h in hits[it["id"]]] if use_rag else None)
            for it in items
        ]
        t0 = time.time()
        answers = chat.chat(convs, **g)
        sec = time.time() - t0
        rows = []
        for it, ans in zip(items, answers):
            rows.append({
                "id": it["id"], "variant": v, "lang": it["lang"], "type": it["type"],
                "question": it["question"], "answer": ans,
                "retrieved": [h.chunk["id"] for h in hits[it["id"]]] if use_rag else [],
                "retrieved_docs": [h.chunk["doc_id"] for h in hits[it["id"]]] if use_rag else [],
            })
        write_jsonl(out_dir / f"{v}.jsonl", rows)
        print(f"{v:9s} {len(rows)} answers in {sec:.0f}s ({sec/len(rows):.1f}s/q) -> {out_dir / (v + '.jsonl')}")


if __name__ == "__main__":
    main()
