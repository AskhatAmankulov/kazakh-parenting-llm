"""Merge seed + synthetic data -> clean -> attach RAG context to a share of samples -> chat-format train/val.

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --no-synthetic            # seed data only
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kpl.config import base_parser, load_config, resolve, set_seed  # noqa: E402
from kpl.data import clean_samples, load_eval, read_jsonl, read_yaml_list, write_jsonl  # noqa: E402
from kpl.prompts import build_messages  # noqa: E402
from kpl.rag import HybridRetriever  # noqa: E402


def get_retriever(cfg) -> HybridRetriever:
    idx = resolve(cfg.paths.index_dir)
    if (idx / "chunks.jsonl").exists():
        dense = (idx / "embeddings.npy").exists()
        try:
            return HybridRetriever.load(idx, cfg.rag.embed_model if dense else None, rrf_k=cfg.rag.rrf_k)
        except ImportError:
            print("[info] sentence-transformers not installed -> BM25-only context")
            return HybridRetriever.load(idx, None, rrf_k=cfg.rag.rrf_k)
    print("[info] no index found -> building BM25-only index on the fly")
    return HybridRetriever.build(resolve(cfg.paths.kb_dir), None, cfg.rag.chunk_chars, cfg.rag.chunk_overlap)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--no-synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    seed = [dict(r, origin="seed") for r in read_yaml_list(resolve(cfg.paths.seed_glob))]
    synth = [] if args.no_synthetic else [r for r in read_jsonl(resolve(cfg.paths.synthetic)) if not r.get("failed")]
    eval_qs = [e["question"] for e in load_eval(resolve(cfg.paths.eval_set))]
    print(f"loaded seed={len(seed)} synthetic={len(synth)} | eval questions used as leak blocklist: {len(eval_qs)}")

    d = cfg.data
    rows, stats = clean_samples(
        seed + synth,
        min_answer_chars=d.min_answer_chars,
        max_answer_chars=d.max_answer_chars,
        near_dup_threshold=d.near_dup_threshold,
        require_lang_match=d.require_lang_match,
        blocklist_questions=eval_qs,
    )
    print("cleaning:", json.dumps(stats, ensure_ascii=False))

    retriever = get_retriever(cfg)
    ratio = cfg.train.train_on_rag_context_ratio
    out = []
    for r in rows:
        use_ctx = rng.random() < ratio
        ctx = []
        if use_ctx:
            hits = retriever.search(r["question"], lang=r["lang"], k=cfg.rag.top_k,
                                    age_months=(r.get("profile") or {}).get("age_months"))
            ctx = [h.chunk for h in hits]
        out.append({
            "id": r["id"],
            "lang": r["lang"],
            "origin": r.get("origin", "synthetic"),
            "with_context": bool(ctx),
            "messages": build_messages(r["question"], r["lang"], r.get("profile"), ctx, r["answer"]),
        })

    # stratified split by language so val has both
    train, val = [], []
    for lang in sorted({o["lang"] for o in out}):
        group = [o for o in out if o["lang"] == lang]
        rng.shuffle(group)
        n_val = max(2, round(len(group) * d.val_ratio))
        val += group[:n_val]
        train += group[n_val:]
    rng.shuffle(train)

    pdir = resolve(cfg.paths.processed_dir)
    write_jsonl(pdir / "train.jsonl", train)
    write_jsonl(pdir / "val.jsonl", val)
    summary = {
        "cleaning": stats,
        "train": len(train),
        "val": len(val),
        "train_by_lang": dict(Counter(o["lang"] for o in train)),
        "train_by_origin": dict(Counter(o["origin"] for o in train)),
        "train_with_context": sum(o["with_context"] for o in train),
    }
    (pdir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
