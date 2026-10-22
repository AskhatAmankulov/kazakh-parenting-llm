"""Deterministic safety layer in front of the model.

An LLM must never be the only thing standing between a parent and an ambulance call, so emergency
patterns are matched with plain regexes and an emergency banner is prepended to the answer
regardless of what the model generates.
"""
from __future__ import annotations

import re

EMERGENCY = {
    "ru": re.compile(
        r"не дыш|задыха|синеет|посинел|судорог|без сознан|не просыпа|батарейк|магнит|"
        r"выпил[аи]? (таблет|лекарств)|проглотил[аи]? (таблет|лекарств|батар|магнит)|не бледнеет|упал[аи]? .*головой.*рвот",
        re.I,
    ),
    "kk": re.compile(
        r"дем алмай|тыныс\w* тоқта|көгер|құрыс|есінен тан|оянбай|батарейка|магнит|"
        r"дәрі\w* жұт|таблетка\w* жұт|бозармай",
        re.I,
    ),
}

BANNER = {
    "ru": "⚠️ Если ребёнок не дышит, синеет, у него судороги, он без сознания или проглотил батарейку, магнит или лекарство — "
          "немедленно звоните 103 или 112.",
    "kk": "⚠️ Егер бала дем алмаса, көгерсе, құрысса, есінен танса немесе батарейка, магнит не дәрі жұтып қойса — "
          "дереу 103 немесе 112 нөміріне қоңырау шалыңыз.",
}


def emergency_banner(question: str, lang: str) -> str | None:
    pat = EMERGENCY.get(lang)
    return BANNER[lang] if pat and pat.search(question) else None
