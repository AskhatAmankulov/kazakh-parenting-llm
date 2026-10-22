"""Language identification (kk / ru), normalisation and near-duplicate detection.

No external model: Kazakh Cyrillic has 9 letters Russian does not (ә ғ қ ң ө ұ ү һ і),
which makes a cheap and very accurate kk/ru classifier for anything longer than a phrase.
This matters for the product: a parent who writes in Kazakh must get the answer in Kazakh,
and "language drift" into Russian is one of the main failure modes of open models.
"""
from __future__ import annotations

import re
import unicodedata

KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
KK_WORDS = {
    "және", "бұл", "үшін", "мен", "сен", "біз", "бала", "баланың", "балаға", "керек", "болады",
    "емес", "жоқ", "бар", "қалай", "неше", "ай", "айлық", "жаста", "жасында", "тамақ", "ұйқы",
    "дәрігер", "ата-ана", "сәби", "нәресте", "егер", "кейін", "дейін", "өте", "қажет",
    "су", "болсын", "шамамен", "немесе", "сіз", "сізге", "оны", "оның", "ол", "тек", "бала", "балаңыз",
    "күніне", "сағат", "кемінде", "жоқ", "иә", "сондай", "сол", "осы", "бір", "екі", "үш", "мысалы",
    "болса", "береді", "беріңіз", "жасаңыз", "тексеріңіз", "салыңыз", "ойын", "ойыншық", "ана", "әке",
}
# Kazakh imperative / plural / case endings that Russian words practically never have
KK_SUFFIX = re.compile(r"(ңыз|ңіз|сын|сін|лар|лер|дар|дер|тар|тер|ның|нің|дың|дің|ға|ге|қа|ке|мен|бен|пен)$")
RU_WORDS = {
    "и", "в", "не", "на", "что", "как", "это", "ребёнок", "ребенок", "ребёнка", "ребенка", "месяцев",
    "лет", "нужно", "можно", "если", "врачу", "сон", "когда", "почему", "он", "она", "мой", "моя",
    "с", "по", "для", "у", "от", "за", "к", "до", "при", "или", "а", "но", "то", "так", "уже", "ещё", "еще",
    "его", "её", "ее", "все", "всё", "только", "очень", "около", "это", "этот", "чтобы", "вы", "ему", "ей",
}
_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_lang(text: str) -> str:
    """Return 'kk', 'ru', 'en' or 'other'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "other"
    cyr = [c for c in letters if "Ѐ" <= c <= "ӿ"]
    latin = [c for c in letters if c.isascii()]
    if len(cyr) < 0.5 * len(letters):
        return "en" if len(latin) >= 0.5 * len(letters) else "other"

    kk_letter_ratio = sum(c in KK_LETTERS for c in cyr) / len(cyr)
    words = [w.lower() for w in _WORD_RE.findall(text)]
    kk_hits = sum(w in KK_WORDS for w in words)
    ru_hits = sum(w in RU_WORDS for w in words)

    # Typical Kazakh prose: 5-10 % Kazakh-specific letters. Russian: 0 %.
    if kk_letter_ratio >= 0.02:
        return "kk"
    kk_suffix_hits = sum(bool(KK_SUFFIX.search(w)) for w in words if w not in RU_WORDS)
    if kk_hits + 0.5 * kk_suffix_hits > ru_hits:
        return "kk"
    return "ru"


def lang_share(text: str, lang: str) -> float:
    """Share of sentences written in `lang` — catches answers that start in Kazakh and drift to Russian."""
    sents = [s for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if len(s.strip()) >= 15]
    if not sents:
        return 1.0 if detect_lang(text) == lang else 0.0
    return sum(detect_lang(s) == lang for s in sents) / len(sents)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = re.sub(r"\W+", " ", text.lower().replace("ё", "е")).strip()
    t = f" {t} "
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dedup_near(items: list[str], threshold: float = 0.82, n: int = 5) -> list[int]:
    """Indices of items to keep. O(N^2) over shingles — fine for a few thousand questions.
    Uses a length-bucket prefilter so the common case stays fast."""
    keep: list[int] = []
    sigs: list[set[str]] = []
    for i, text in enumerate(items):
        s = char_ngrams(text, n)
        dup = False
        for kept_sig in sigs:
            # Jaccard can't exceed min/max size ratio — skip hopeless pairs
            small, big = sorted((len(s), len(kept_sig)))
            if big and small / big < threshold:
                continue
            if jaccard(s, kept_sig) >= threshold:
                dup = True
                break
        if not dup:
            keep.append(i)
            sigs.append(s)
    return keep
