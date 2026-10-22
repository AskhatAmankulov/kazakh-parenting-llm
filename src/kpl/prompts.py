"""Prompt construction: system prompt per language, child-profile personalisation, RAG context block.

The SAME builder is used for training, evaluation and serving, so the model never sees a prompt
format at inference time that it was not trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SYSTEM = {
    "ru": (
        "Ты — заботливый консультант по раннему развитию детей от 0 до 3 лет. "
        "Отвечай на русском языке, тепло и по делу, опираясь на доказательные рекомендации (ВОЗ, ЮНИСЕФ, педиатрические ассоциации). "
        "Учитывай возраст ребёнка. Давай конкретные, выполнимые советы. "
        "Не ставь диагнозы и не назначай лекарства. Если есть тревожные признаки, прямо скажи, что нужно обратиться к педиатру, "
        "а при угрозе жизни — вызвать скорую (103 или 112)."
    ),
    "kk": (
        "Сен 0 мен 3 жас аралығындағы балалардың ерте дамуы бойынша қамқор кеңесшісің. "
        "Тек қазақ тілінде, жылы әрі нақты жауап бер, дәлелді ұсынымдарға (ДДСҰ, ЮНИСЕФ, педиатриялық қауымдастықтар) сүйен. "
        "Баланың жасын ескер. Нақты, орындауға болатын кеңестер бер. "
        "Диагноз қойма, дәрі тағайындама. Алаңдататын белгілер болса, педиатрға қаралу керектігін тура айт, "
        "ал өмірге қауіп төнсе — жедел жәрдем шақыруды (103 немесе 112) айт."
    ),
}

CONTEXT_HEADER = {
    "ru": "Справочные материалы (используй, если они относятся к вопросу; не выдумывай фактов сверх них):",
    "kk": "Анықтамалық материалдар (сұраққа қатысты болса пайдалан; олардан тыс дерек ойдан шығарма):",
}


@dataclass
class ChildProfile:
    age_months: int | None = None
    name: str | None = None
    premature_weeks: int = 0          # born N weeks early -> use corrected age for milestones
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ChildProfile":
        d = d or {}
        return cls(
            age_months=d.get("age_months"),
            name=d.get("name"),
            premature_weeks=int(d.get("premature_weeks") or 0),
            notes=list(d.get("notes") or []),
        )

    @property
    def corrected_age_months(self) -> int | None:
        if self.age_months is None:
            return None
        if self.premature_weeks >= 3 and self.age_months < 24:   # correction is used until ~2 years
            return max(0, round(self.age_months - self.premature_weeks / 4.345))
        return self.age_months


def _fmt_age(months: int, lang: str) -> str:
    y, m = divmod(months, 12)
    if lang == "kk":
        if y == 0:
            return f"{m} ай"
        return f"{y} жас" + (f" {m} ай" if m else "")
    if y == 0:
        return f"{m} мес."
    return (f"{y} г. " + (f"{m} мес." if m else "")).strip()


def profile_block(p: ChildProfile, lang: str) -> str:
    if p.age_months is None and not p.notes and not p.name:
        return ""
    lines = []
    if lang == "kk":
        lines.append("Бала туралы мәлімет:")
        if p.name:
            lines.append(f"- аты: {p.name}")
        if p.age_months is not None:
            lines.append(f"- жасы: {_fmt_age(p.age_months, lang)}")
        if p.premature_weeks >= 3 and p.corrected_age_months is not None:
            lines.append(
                f"- {p.premature_weeks} апта ерте туған; даму кезеңдерін түзетілген жаспен бағала: {_fmt_age(p.corrected_age_months, lang)}"
            )
        lines += [f"- {n}" for n in p.notes]
    else:
        lines.append("Профиль ребёнка:")
        if p.name:
            lines.append(f"- имя: {p.name}")
        if p.age_months is not None:
            lines.append(f"- возраст: {_fmt_age(p.age_months, lang)}")
        if p.premature_weeks >= 3 and p.corrected_age_months is not None:
            lines.append(
                f"- родился на {p.premature_weeks} нед. раньше срока; вехи развития оценивай по скорректированному возрасту: {_fmt_age(p.corrected_age_months, lang)}"
            )
        lines += [f"- {n}" for n in p.notes]
    return "\n".join(lines)


def context_block(chunks: list[dict[str, Any]], lang: str) -> str:
    if not chunks:
        return ""
    parts = [CONTEXT_HEADER[lang]]
    for i, ch in enumerate(chunks, 1):
        parts.append(f"[{i}] {ch['title']}\n{ch['text']}")
    return "\n\n".join(parts)


def build_messages(
    question: str,
    lang: str,
    profile: dict[str, Any] | ChildProfile | None = None,
    context: list[dict[str, Any]] | None = None,
    answer: str | None = None,
) -> list[dict[str, str]]:
    if lang not in SYSTEM:
        raise ValueError(f"unsupported lang {lang!r}")
    p = profile if isinstance(profile, ChildProfile) else ChildProfile.from_dict(profile)
    system = SYSTEM[lang]
    pb = profile_block(p, lang)
    if pb:
        system += "\n\n" + pb
    user = question.strip()
    cb = context_block(context or [], lang)
    if cb:
        user = f"{cb}\n\n---\n{user}"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer.strip()})
    return msgs
