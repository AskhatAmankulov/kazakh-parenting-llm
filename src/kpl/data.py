"""Dataset I/O, validation and cleaning."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .textutils import detect_lang, dedup_near, lang_share, normalize

REQUIRED = ("id", "lang", "question", "answer")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_yaml_list(pattern: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in sorted(glob.glob(str(pattern))):
        with open(p, encoding="utf-8") as f:
            items.extend(yaml.safe_load(f) or [])
    return items


def load_eval(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    items = read_yaml_list(path) if path.suffix in {".yaml", ".yml"} else read_jsonl(path)
    for it in items:
        it.setdefault("profile", {})
        it.setdefault("sources", [])
        it.setdefault("key_facts", [])
        it.setdefault("type", "guidance")
    return items


# ------------------------------------------------------------------- cleaning
def clean_samples(rows: list[dict[str, Any]], *, min_answer_chars: int, max_answer_chars: int,
                  near_dup_threshold: float, require_lang_match: bool,
                  blocklist_questions: list[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Returns (kept rows, drop statistics). Order of filters is deliberate — cheap ones first."""
    stats = {"input": len(rows)}
    out = []
    for r in rows:
        if any(not r.get(k) for k in REQUIRED):
            stats["missing_field"] = stats.get("missing_field", 0) + 1
            continue
        r = dict(r)
        r["question"], r["answer"] = normalize(r["question"]), normalize(r["answer"])
        n = len(r["answer"])
        if n < min_answer_chars or n > max_answer_chars:
            stats["bad_length"] = stats.get("bad_length", 0) + 1
            continue
        if require_lang_match:
            if detect_lang(r["question"]) != r["lang"] or lang_share(r["answer"], r["lang"]) < 0.9:
                stats["lang_mismatch"] = stats.get("lang_mismatch", 0) + 1
                continue
        out.append(r)

    # exact dup by (lang, question)
    seen, uniq = set(), []
    for r in out:
        key = (r["lang"], r["question"].lower())
        if key in seen:
            stats["exact_dup"] = stats.get("exact_dup", 0) + 1
            continue
        seen.add(key)
        uniq.append(r)

    # near dup within each language (seed rows come first so they win)
    uniq.sort(key=lambda r: (r["lang"], r.get("origin") != "seed"))
    kept: list[dict[str, Any]] = []
    for lang in sorted({r["lang"] for r in uniq}):
        group = [r for r in uniq if r["lang"] == lang]
        idx = dedup_near([r["question"] for r in group], near_dup_threshold)
        stats["near_dup"] = stats.get("near_dup", 0) + len(group) - len(idx)
        kept.extend(group[i] for i in idx)

    # leakage guard: nothing that looks like an eval question may enter training
    if blocklist_questions:
        from .textutils import char_ngrams, jaccard

        block = [char_ngrams(q) for q in blocklist_questions]
        safe = []
        for r in kept:
            s = char_ngrams(r["question"])
            if any(jaccard(s, b) >= 0.6 for b in block):
                stats["eval_leak"] = stats.get("eval_leak", 0) + 1
                continue
            safe.append(r)
        kept = safe

    stats["kept"] = len(kept)
    return kept, stats
