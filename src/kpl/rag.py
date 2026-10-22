"""Hybrid retrieval over the bilingual knowledge base.

  * chunking: markdown section-aware (## headings) with character windows + overlap
  * sparse:   BM25 over lightly stemmed tokens (prefix truncation — crude, but it helps a lot
              for agglutinative Kazakh: "баланың", "балаға", "балалар" -> "бала"; prefix length 4 chosen on training questions)
  * dense:    BAAI/bge-m3 (multilingual, covers Kazakh), cosine similarity
  * fusion:   Reciprocal Rank Fusion + a small boost for chunks whose age range covers the child
  * filter:   same-language chunks first (KB is parallel kk/ru), so the context never pulls
              the model into the wrong language
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .textutils import normalize

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_TOKEN = re.compile(r"[\wәғқңөұүһіё]+", re.UNICODE)
STEM_LEN = 4  # tuned on the seed TRAIN questions (recall@1 47/68 vs 40/68 without stemming), not on eval


# ---------------------------------------------------------------- knowledge base parsing
def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v
    return meta, text[m.end():]


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries, packing paragraphs into windows of ~size chars."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            out.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def chunk_document(path: Path, size: int = 900, overlap: int = 150) -> list[dict[str, Any]]:
    raw = normalize(path.read_text(encoding="utf-8"))
    meta, body = parse_front_matter(raw + "\n")
    doc_id = meta.get("id", path.stem)
    lang = meta.get("lang", path.parent.name)
    title = meta.get("title", doc_id)
    age_from, age_to = (int(x) for x in str(meta.get("age_months", "0-36")).split("-"))

    body = re.sub(r"^# .*\n", "", body.lstrip(), count=1)        # drop the H1, title is in meta
    sections = re.split(r"^## +", body, flags=re.M)
    chunks = []
    for s_i, sec in enumerate(sections):
        if s_i == 0:                                              # intro before first ##
            heading, text = title, sec.strip()
        else:
            h, _, text = sec.partition("\n")
            heading, text = f"{title} — {h.strip()}", text.strip()
        if not text:
            continue
        for w_i, win in enumerate(_windows(text, size, overlap)):
            chunks.append(
                {
                    "id": f"{doc_id}#{s_i}.{w_i}",
                    "doc_id": doc_id,
                    "lang": lang,
                    "title": heading,
                    "text": win,
                    "age_from": age_from,
                    "age_to": age_to,
                    "sources": meta.get("sources", []),
                }
            )
    return chunks


def load_kb(kb_dir: Path, size: int = 900, overlap: int = 150) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(Path(kb_dir).glob("*/*.md")):
        chunks.extend(chunk_document(path, size, overlap))
    return chunks


# ---------------------------------------------------------------- retrieval
def tokenize(text: str) -> list[str]:
    return [t[:STEM_LEN] for t in _TOKEN.findall(text.lower()) if len(t) > 1]


@dataclass
class Hit:
    chunk: dict[str, Any]
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    def __init__(self, chunks: list[dict[str, Any]], embeddings: np.ndarray | None = None,
                 embedder: Any = None, rrf_k: int = 60, bm25_weight: float = 1.0,
                 dense_weight: float = 1.0, age_boost: float = 0.15):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in chunks])
        self.emb = embeddings
        self.embedder = embedder
        self.rrf_k, self.wb, self.wd, self.age_boost = rrf_k, bm25_weight, dense_weight, age_boost

    # -- persistence -------------------------------------------------------------
    @classmethod
    def build(cls, kb_dir: Path, embed_model: str | None, size: int = 900, overlap: int = 150, **kw):
        chunks = load_kb(kb_dir, size, overlap)
        embedder = emb = None
        if embed_model:
            embedder = load_embedder(embed_model)
            emb = embed(embedder, [c["title"] + "\n" + c["text"] for c in chunks])
        return cls(chunks, emb, embedder, **kw)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        if self.emb is not None:
            np.save(out_dir / "embeddings.npy", self.emb)

    @classmethod
    def load(cls, out_dir: Path, embed_model: str | None = None, **kw):
        chunks = [json.loads(l) for l in open(out_dir / "chunks.jsonl", encoding="utf-8")]
        emb_path = out_dir / "embeddings.npy"
        emb = np.load(emb_path) if emb_path.exists() else None
        embedder = load_embedder(embed_model) if (embed_model and emb is not None) else None
        return cls(chunks, emb, embedder, **kw)

    # -- search ------------------------------------------------------------------
    def search(self, query: str, lang: str | None = None, k: int = 4,
               age_months: int | None = None, pool: int = 30) -> list[Hit]:
        idx = np.arange(len(self.chunks))
        if lang:
            same = [i for i, c in enumerate(self.chunks) if c["lang"] == lang]
            if same:
                idx = np.array(same)
        scores: dict[int, float] = {}
        bm_rank: dict[int, int] = {}
        de_rank: dict[int, int] = {}

        bm = self.bm25.get_scores(tokenize(query))[idx]
        for r, j in enumerate(np.argsort(-bm)[:pool]):
            if bm[j] <= 0:
                break
            i = int(idx[j])
            bm_rank[i] = r
            scores[i] = scores.get(i, 0.0) + self.wb / (self.rrf_k + r + 1)

        if self.emb is not None and self.embedder is not None:
            q = embed(self.embedder, [query])[0]
            sims = self.emb[idx] @ q
            for r, j in enumerate(np.argsort(-sims)[:pool]):
                i = int(idx[j])
                de_rank[i] = r
                scores[i] = scores.get(i, 0.0) + self.wd / (self.rrf_k + r + 1)

        if age_months is not None and scores:
            top = max(scores.values())
            for i in scores:
                c = self.chunks[i]
                if c["age_from"] <= age_months <= c["age_to"]:
                    scores[i] += self.age_boost * top

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [Hit(self.chunks[i], s, bm_rank.get(i), de_rank.get(i)) for i, s in ranked]


# ---------------------------------------------------------------- embeddings
def load_embedder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(model, texts: list[str], batch_size: int = 16) -> np.ndarray:
    v = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 64)
    return np.asarray(v, dtype=np.float32)
