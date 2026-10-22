"""Grounded synthetic Q&A generation with a teacher LLM.

For every knowledge-base chunk the teacher writes several realistic parent questions (with a child
profile) and answers that rely ONLY on that chunk. Seed examples in the same language are used as
few-shot style references. Output is resumable: chunks already processed are skipped.

    export TEACHER_API_KEY=...          # e.g. a free Gemini key (default config) or Groq / OpenAI
    python scripts/generate_synthetic.py --rounds 3
    python scripts/generate_synthetic.py --set teacher.backend=hf      # local teacher on GPU, no key
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from kpl.config import base_parser, load_config, resolve, set_seed  # noqa: E402
from kpl.data import read_jsonl, read_yaml_list  # noqa: E402
from kpl.llm import QuotaExhausted, teacher_from_cfg  # noqa: E402
from kpl.rag import load_kb  # noqa: E402

LANG_NAME = {"ru": "русском", "kk": "казахском"}

STYLES = [
    "короткий вопрос в мессенджере, без лишних деталей",
    "встревоженный родитель подробно описывает ситуацию",
    "вопрос с конкретной деталью (дата рождения, что уже пробовали, кто ещё в семье)",
    "вопрос-сомнение: родственники советуют одно, а родитель не уверен",
    "разговорный стиль, возможны опечатки и отсутствие заглавных букв",
]

PROMPT = """Ты помогаешь собрать обучающий датасет для ассистента по раннему развитию детей 0–3 лет.

Ниже фрагмент проверенного справочника на {lang_name} языке. Придумай {n} разных реалистичных вопроса,
которые могли бы задать родители, и идеальные ответы ассистента.

ПРАВИЛА:
1. Вопросы и ответы — только на {lang_name} языке. {kk_rule}
2. Ответ опирается ТОЛЬКО на факты из фрагмента и общие правила безопасности. Не выдумывай цифры, препараты, дозировки.
3. Возраст ребёнка в профиле выбирай в пределах {age_from}–{age_to} месяцев, и ответ должен учитывать этот возраст.
4. Ответ: тёплый, конкретный, 80–220 слов, с короткими списками там, где они уместны. Без приветствий и без «отличный вопрос».
5. Если в ситуации есть тревожный признак — ответ прямо направляет к педиатру, а при угрозе жизни — в скорую (103 или 112).
6. Стили вопросов (используй разные): {styles}.
7. Вопросы должны отличаться друг от друга по смыслу, а не только формулировкой.

Пример стиля ответа:
ВОПРОС: {ex_q}
ОТВЕТ: {ex_a}

ФРАГМЕНТ СПРАВОЧНИКА «{title}»:
\"\"\"
{text}
\"\"\"

Верни строго JSON без пояснений:
{{"items": [{{"age_months": <int>, "question": "...", "answer": "..."}}, ...]}}"""

KK_RULE = (
    "Пиши на естественном литературном казахском языке, как пишут казахоязычные родители; "
    "не вставляй русские слова, если есть общеупотребительный казахский эквивалент "
    "(«емшек», «жөргек», «түбек», «еңбек», «дәрігер»)."
)


def parse_items(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("question") and it.get("answer"):
            out.append(it)
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rounds", type=int, default=2, help="passes over the KB (different styles / seeds each round)")
    ap.add_argument("--limit", type=int, default=0, help="only N chunks (smoke test)")
    ap.add_argument("--langs", nargs="*", default=["ru", "kk"])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.seed)

    out_path = resolve(cfg.paths.synthetic)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["chunk_id"], r["round"]) for r in read_jsonl(out_path)}

    chunks = [c for c in load_kb(resolve(cfg.paths.kb_dir), cfg.rag.chunk_chars, cfg.rag.chunk_overlap)
              if c["lang"] in args.langs]
    if args.limit:
        chunks = chunks[: args.limit]
    seeds = read_yaml_list(resolve(cfg.paths.seed_glob))
    seeds_by_lang = {l: [s for s in seeds if s["lang"] == l] for l in ("ru", "kk")}

    teacher = teacher_from_cfg(cfg)
    model_name = cfg.teacher.model if cfg.teacher.backend == "openai" else cfg.teacher.hf_model
    jobs = [(c, r) for r in range(args.rounds) for c in chunks if (c["id"], r) not in done]
    print(f"{len(chunks)} chunks x {args.rounds} rounds, {len(jobs)} to do, teacher={model_name}")

    n_new = fails = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for c, rnd in tqdm(jobs):
            rng = random.Random(f"{c['id']}-{rnd}")
            ex = rng.choice(seeds_by_lang[c["lang"]])
            prompt = PROMPT.format(
                lang_name=LANG_NAME[c["lang"]], n=cfg.teacher.questions_per_chunk,
                kk_rule=KK_RULE if c["lang"] == "kk" else "",
                age_from=c["age_from"], age_to=c["age_to"],
                styles="; ".join(rng.sample(STYLES, 3)),
                ex_q=ex["question"], ex_a=ex["answer"].strip(),
                title=c["title"], text=c["text"],
            )
            try:
                raw = teacher.chat([[{"role": "user", "content": prompt}]], temperature=cfg.teacher.temperature,
                                   max_new_tokens=8192, json_mode=True)[0]
                fails = 0
            except QuotaExhausted as e:
                print(f"\n[stop] teacher quota exhausted — keeping the {n_new} samples generated so far.\n{e}")
                break
            except Exception as e:
                fails += 1
                print(f"\n[warn] {c['id']} round {rnd}: {type(e).__name__}: {str(e)[:200]}")
                if fails >= 5:
                    print("[stop] 5 failures in a row — keeping what we have.")
                    break
                continue
            items = parse_items(raw)
            if not items:
                print(f"\n[warn] unparsable output for {c['id']} round {rnd}")
            for k, it in enumerate(items):
                age = it.get("age_months")
                row = {
                    "id": f"syn-{c['id']}-{rnd}-{k}",
                    "lang": c["lang"],
                    "profile": {"age_months": int(age)} if isinstance(age, (int, float)) else {},
                    "sources": [c["doc_id"]],
                    "question": it["question"],
                    "answer": it["answer"],
                    "origin": "synthetic",
                    "teacher": model_name,
                    "chunk_id": c["id"],
                    "round": rnd,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_new += 1
            if not items:  # still mark as attempted so a bad chunk does not block resume forever
                f.write(json.dumps({"chunk_id": c["id"], "round": rnd, "failed": True}) + "\n")
            f.flush()
    print(f"wrote {n_new} new samples -> {out_path}")


if __name__ == "__main__":
    main()
