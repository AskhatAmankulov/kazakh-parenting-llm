from pathlib import Path

import pytest

from kpl.config import ROOT
from kpl.prompts import ChildProfile, build_messages
from kpl.rag import HybridRetriever, load_kb, tokenize

KB = ROOT / "data" / "kb"


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever.build(KB, embed_model=None)


def test_kb_is_parallel():
    ru = {p.stem for p in (KB / "ru").glob("*.md")}
    kk = {p.stem for p in (KB / "kk").glob("*.md")}
    assert ru == kk and len(ru) >= 12


def test_chunks_have_metadata():
    chunks = load_kb(KB)
    assert chunks
    for c in chunks:
        assert c["lang"] in ("ru", "kk")
        assert 0 <= c["age_from"] <= c["age_to"] <= 36
        assert len(c["text"]) > 40
        assert not c["text"].startswith("---")


def test_stemming_merges_kazakh_forms():
    assert tokenize("баланың балаға балалар") == ["бала"] * 3


@pytest.mark.parametrize(
    "q,lang,doc",
    [
        ("Можно ли давать мёд ребёнку 8 месяцев?", "ru", "feeding"),
        ("Как приучить к горшку?", "ru", "toilet"),
        ("Баланы қалай қауіпсіз ұйықтату керек? Шалқасынан ба?", "kk", "sleep"),
        ("Балам батарейка жұтып қойды", "kk", "safety"),
    ],
)
def test_retrieval_top4(retriever, q, lang, doc):
    hits = retriever.search(q, lang=lang, k=4)
    assert all(h.chunk["lang"] == lang for h in hits)
    assert doc in {h.chunk["doc_id"] for h in hits}


def test_corrected_age():
    p = ChildProfile(age_months=7, premature_weeks=8)
    assert p.corrected_age_months == 5
    assert ChildProfile(age_months=30, premature_weeks=8).corrected_age_months == 30  # no correction after 2y


def test_build_messages_layout():
    ctx = [{"title": "Сон", "text": "Спать на спине."}]
    msgs = build_messages("Как спать?", "ru", {"age_months": 14}, ctx, answer="На спине.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "1 г. 2 мес." in msgs[0]["content"]
    assert msgs[1]["content"].startswith("Справочные материалы")
    assert msgs[1]["content"].endswith("Как спать?")
    kk = build_messages("Қалай?", "kk", {"age_months": 5})
    assert "5 ай" in kk[0]["content"] and "қазақ тілінде" in kk[0]["content"]
