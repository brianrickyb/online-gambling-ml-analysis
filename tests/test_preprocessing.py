import pandas as pd
import pytest

from src.preprocessing import (
    clean_data, non_latin_letter_ratio, normalize_text, styled_char_ratio, text_stats,
    TEXT_STAT_NAMES,
)


@pytest.mark.parametrize("raw, expected", [
    ("𝗪𝗘𝗧𝗢𝗡𝟴𝟴", "weton88"),                 # math bold
    ("Ｐｕｌａｕ７７７", "pulau777"),          # full-width
    ("AHMаDtoTO", "ahmadtoto"),              # Cyrillic "а"
    ("gacu​r ha​ri", "gacur hari"),  # zero-width space
    ("k o i s l o t", "koislot"),            # spaced letters
    ("ᑭ ᑌ ᒪ ᗩ ᑌ ᗯ I ᑎ", "pulauwin"),         # Canadian syllabics
    ("В𝒪Ꮤ𝒪𝟩𝟩", "bowo77"),                   # Cherokee (has case)
    ("PulΛᴜ𝟽𝟽𝟽", "pulau777"),                # Greek + small caps
    ("cek https://x.co/a @budi", "cek url user"),
])
def test_normalize_text_undoes_obfuscation(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_keeps_plain_indonesian():
    assert normalize_text("Mobilnya  keren,  Pak!") == "mobilnya keren, pak!"


def test_styled_and_non_latin_ratios():
    assert styled_char_ratio("biasa saja") == 0
    assert styled_char_ratio("𝐒𝐆𝐈") == 1
    assert non_latin_letter_ratio("biasa saja") == 0
    assert non_latin_letter_ratio("ᗩᗷ") == 1


def test_text_stats_shape():
    assert len(text_stats("main di alexis17")) == len(TEXT_STAT_NAMES)


def test_clean_data_drops_leaky_columns_and_duplicates():
    df = pd.DataFrame({
        "komentar": ["a", "a", "b", "c", None],
        "label": [1, 1, 0, 1, 0],
        "predicted_label": [1, 1, 0, 1, 0],
        "komentar_clean": ["a", "a", "b", "c", "d"],
    })
    out = clean_data(df)
    assert "predicted_label" not in out.columns
    assert "komentar_clean" not in out.columns
    assert sorted(out["komentar"]) == ["a", "b", "c"]
