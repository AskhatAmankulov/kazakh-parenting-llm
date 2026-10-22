"""FastAPI service: language detection -> safety guardrail -> hybrid RAG -> fine-tuned model.

Backends (env KPL_BACKEND):
  llamacpp  — OpenAI-compatible llama.cpp server running the Q4_K_M GGUF (CPU-friendly, default)
                llama-server -m artifacts/gguf/model-Q4_K_M.gguf --port 8080 -c 4096
  hf        — transformers + 4-bit base + LoRA adapter on GPU
  mock      — echoes retrieved context; for tests / UI work without a model

    uvicorn app.server:app --port 8000
    curl -s localhost:8000/chat -H 'content-type: application/json' \
         -d '{"question": "Балам 10 айлық, түнде жиі оянады", "profile": {"age_months": 10}}'
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from kpl.config import load_config, resolve  # noqa: E402
from kpl.guardrails import emergency_banner  # noqa: E402
from kpl.prompts import build_messages  # noqa: E402
from kpl.rag import HybridRetriever  # noqa: E402
from kpl.textutils import detect_lang  # noqa: E402

cfg = load_config(os.environ.get("KPL_CONFIG"))
BACKEND = os.environ.get("KPL_BACKEND", "llamacpp")


class Profile(BaseModel):
    age_months: int | None = Field(None, ge=0, le=72)
    name: str | None = None
    premature_weeks: int = Field(0, ge=0, le=20)
    notes: list[str] = []


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    lang: str | None = Field(None, pattern="^(kk|ru)$", description="auto-detected if omitted")
    profile: Profile = Profile()
    use_rag: bool = True


class Source(BaseModel):
    doc_id: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    lang: str
    emergency: bool
    sources: list[Source]
    latency_ms: int


def _make_llm():
    if BACKEND == "mock":
        return None
    if BACKEND == "llamacpp":
        from kpl.llm import OpenAIChat

        os.environ.setdefault("LLAMACPP_API_KEY", "none")
        return OpenAIChat(model=os.environ.get("KPL_MODEL", "local"),
                          base_url=os.environ.get("LLAMACPP_URL", "http://localhost:8080/v1"),
                          api_key_env="LLAMACPP_API_KEY")
    if BACKEND == "hf":
        from kpl.llm import HFChat

        m = cfg.model
        return HFChat(m.base, adapter=str(resolve(cfg.paths.adapter_dir)), load_in_4bit=m.load_in_4bit,
                      compute_dtype=m.compute_dtype, batch_size=1)
    raise RuntimeError(f"unknown KPL_BACKEND={BACKEND}")


def _make_retriever() -> HybridRetriever:
    idx = resolve(cfg.paths.index_dir)
    if (idx / "chunks.jsonl").exists():
        dense = (idx / "embeddings.npy").exists() and os.environ.get("KPL_DENSE", "1") == "1"
        try:
            return HybridRetriever.load(idx, cfg.rag.embed_model if dense else None, rrf_k=cfg.rag.rrf_k)
        except ImportError:
            return HybridRetriever.load(idx, None, rrf_k=cfg.rag.rrf_k)
    return HybridRetriever.build(resolve(cfg.paths.kb_dir), None, cfg.rag.chunk_chars, cfg.rag.chunk_overlap)


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app):
    state["retriever"] = _make_retriever()
    state["llm"] = _make_llm()
    yield
    state.clear()


app = FastAPI(title="Kazakh/Russian parenting assistant", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "chunks": len(state["retriever"].chunks)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()
    lang = req.lang or detect_lang(req.question)
    if lang not in ("kk", "ru"):
        raise HTTPException(422, "only Kazakh and Russian are supported")
    profile = req.profile.model_dump()

    hits = state["retriever"].search(req.question, lang=lang, k=cfg.rag.top_k,
                                     age_months=profile.get("age_months")) if req.use_rag else []
    chunks = [h.chunk for h in hits]
    messages = build_messages(req.question, lang, profile, chunks)

    llm = state["llm"]
    if llm is None:
        answer = "[mock] " + (chunks[0]["text"][:400] if chunks else "no context")
    else:
        g = dict(cfg.generation)
        answer = llm.chat([messages], **g)[0]

    banner = emergency_banner(req.question, lang)
    if banner:
        answer = f"{banner}\n\n{answer}"

    seen, sources = set(), []
    for c in chunks:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            sources.append(Source(doc_id=c["doc_id"], title=c["title"].split(" — ")[0]))
    return ChatResponse(answer=answer, lang=lang, emergency=bool(banner), sources=sources,
                        latency_ms=int((time.time() - t0) * 1000))
