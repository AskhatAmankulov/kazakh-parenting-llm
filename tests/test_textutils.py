import pytest

from kpl.textutils import dedup_near, detect_lang, lang_share


@pytest.mark.parametrize(
    "text,lang",
    [
        ("Балам түнде жиі оянады, не істеймін?", "kk"),
        ("су температурасы шамамен 37 °C болсын;", "kk"),   # no Kazakh-specific letters
        ("Иә", "kk"),
        ("Ребёнок плачет по вечерам, что делать?", "ru"),
        ("Свитер на руке висит у компьютера", "ru"),        # Russian words with "Kazakh" endings
        ("Hello there", "en"),
        ("12345", "other"),
    ],
)
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang


def test_lang_share_catches_drift():
    answer = (
        "Бұл қалыпты жағдай, уайымдамаңыз. "
        "Баланы күніне бірнеше рет етпетінен жатқызыңыз. "
        "Если не поможет, обратитесь к педиатру как можно скорее."
    )
    assert lang_share(answer, "kk") == pytest.approx(2 / 3)
    assert lang_share(answer, "ru") == pytest.approx(1 / 3)


def test_dedup_near_keeps_first():
    qs = [
        "Сколько должен спать ребёнок в 8 месяцев?",
        "Сколько должен спать ребенок в 8 месяцев??",
        "Когда вводить прикорм?",
    ]
    assert dedup_near(qs, threshold=0.8) == [0, 2]
