"""Chunk the knowledge base and build the hybrid (BM25 + bge-m3) index.

    python scripts/build_index.py                 # BM25 + dense (needs sentence-transformers, GPU recommended)
    python scripts/build_index.py --no-dense      # BM25 only (CPU, seconds)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kpl.config import base_parser, load_config, resolve  # noqa: E402
from kpl.rag import HybridRetriever  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    r = cfg.rag

    retriever = HybridRetriever.build(
        resolve(cfg.paths.kb_dir),
        embed_model=None if args.no_dense else r.embed_model,
        size=r.chunk_chars,
        overlap=r.chunk_overlap,
    )
    out = resolve(cfg.paths.index_dir)
    retriever.save(out)
    by_lang = Counter(c["lang"] for c in retriever.chunks)
    lens = [len(c["text"]) for c in retriever.chunks]
    print(f"chunks: {len(retriever.chunks)} {dict(by_lang)} | chars avg={sum(lens)//len(lens)} max={max(lens)}")
    print(f"dense: {'yes ' + str(retriever.emb.shape) if retriever.emb is not None else 'no'} -> {out}")


if __name__ == "__main__":
    main()
