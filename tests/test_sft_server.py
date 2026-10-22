"""Loss-masking and API tests that run without torch weights or a GPU."""
import os

import pytest

from kpl.sft import IGNORE, tokenize_example


class CharTokenizer:
    """Minimal stand-in for a HF tokenizer with a ChatML template (Qwen style)."""

    eos_token = "<|im_end|>"
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _msgs(answer="Ответ."):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": answer},
    ]


def test_only_answer_is_supervised():
    tok = CharTokenizer()
    ex = tokenize_example(tok, _msgs("Ответ."), max_len=10_000)
    supervised = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE)
    assert supervised == "Ответ.<|im_end|>\n"
    prompt = "".join(chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l == IGNORE)
    assert prompt.endswith("<|im_start|>assistant\n") and "вопрос" in prompt
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_too_long_is_dropped_not_truncated():
    assert tokenize_example(CharTokenizer(), _msgs("x" * 500), max_len=100) is None


@pytest.fixture(scope="module")
def client():
    os.environ["KPL_BACKEND"] = "mock"
    from fastapi.testclient import TestClient

    import app.server as server

    server.BACKEND = "mock"
    with TestClient(server.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["chunks"] > 0


def test_chat_detects_kazakh_and_cites_sources(client):
    r = client.post("/chat", json={"question": "Балам түнде жиі оянады, ұйқысын қалай жақсартамын?",
                                   "profile": {"age_months": 10}})
    body = r.json()
    assert r.status_code == 200
    assert body["lang"] == "kk" and not body["emergency"]
    assert "sleep" in {s["doc_id"] for s in body["sources"]}


def test_chat_emergency_banner(client):
    body = client.post("/chat", json={"question": "Сын проглотил батарейку от пульта"}).json()
    assert body["emergency"] and body["answer"].startswith("⚠️")


def test_chat_rejects_other_languages(client):
    assert client.post("/chat", json={"question": "my baby does not sleep"}).status_code == 422
