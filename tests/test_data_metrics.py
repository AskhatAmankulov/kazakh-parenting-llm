import pytest

from kpl.config import ROOT, load_config
from kpl.data import clean_samples, load_eval, read_yaml_list
from kpl.guardrails import emergency_banner
from kpl.metrics import fact_coverage, item_metrics, no_dosing, referral

CFG = load_config()
SEED = read_yaml_list(ROOT / CFG["paths"]["seed_glob"])
EVAL = load_eval(ROOT / CFG["paths"]["eval_set"])


def test_seed_passes_quality_filters():
    d = CFG["data"]
    kept, stats = clean_samples(
        SEED, min_answer_chars=d["min_answer_chars"], max_answer_chars=d["max_answer_chars"],
        near_dup_threshold=d["near_dup_threshold"], require_lang_match=True,
        blocklist_questions=[e["question"] for e in EVAL],
    )
    assert stats["kept"] == len(SEED), stats


def test_ids_unique_and_balanced():
    ids = [s["id"] for s in SEED] + [e["id"] for e in EVAL]
    assert len(ids) == len(set(ids))
    for rows in (SEED, EVAL):
        langs = [r["lang"] for r in rows]
        assert langs.count("ru") == langs.count("kk")


def test_eval_schema():
    kb_ids = {p.stem for p in (ROOT / "data/kb/ru").glob("*.md")}
    for e in EVAL:
        assert e["type"] in {"guidance", "red_flag", "out_of_scope"}
        assert e["key_facts"] and all(isinstance(f, list) and f for f in e["key_facts"])
        assert set(e["sources"]) <= kb_ids
        assert e["reference"]


def test_reference_answers_score_well():
    """Sanity check of the metric itself: gold answers should cover most of their own key facts."""
    cov = [fact_coverage(e["reference"], e["key_facts"]) for e in EVAL]
    assert sum(cov) / len(cov) > 0.8


def test_metrics_basic():
    assert fact_coverage("Спать на спине, на твёрдом матрасе", [["на спин"], ["тверд"], ["свдс"]]) == pytest.approx(2 / 3)
    assert referral("Срочно покажите ребёнка педиатру", "ru")
    assert referral("Дереу 103 нөміріне қоңырау шалыңыз", "kk")
    assert not no_dosing("Дайте 5 мл сиропа")
    assert no_dosing("Дозу подберёт врач")


def test_item_metrics_red_flag():
    gold = next(e for e in EVAL if e["type"] == "red_flag")
    gen = {"variant": "x", "answer": "Это нормально, подождите.", "retrieved_docs": gold["sources"]}
    m = item_metrics(gen, gold, k=4)
    assert m["referral"] is False and m["recall_at_k"] == 1.0


@pytest.mark.parametrize(
    "q,lang,expected",
    [
        ("Ребёнок проглотил батарейку", "ru", True),
        ("У сына судороги и температура", "ru", True),
        ("Сколько спит ребёнок в 5 месяцев?", "ru", False),
        ("Балам дем алмай жатыр", "kk", True),
        ("Балам түбекке отырғысы келмейді", "kk", False),
    ],
)
def test_guardrail(q, lang, expected):
    assert (emergency_banner(q, lang) is not None) is expected


def test_parse_teacher_output():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gs", ROOT / "scripts" / "generate_synthetic.py")
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    raw = '```json\n{"items": [{"age_months": 6, "question": "Q?", "answer": "A."}, {"question": ""}]}\n```'
    assert gs.parse_items(raw) == [{"age_months": 6, "question": "Q?", "answer": "A."}]
    assert gs.parse_items("not json") == []
