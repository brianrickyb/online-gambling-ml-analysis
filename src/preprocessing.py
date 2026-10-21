import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import LABEL_COL, LEAKY_COLS, TEXT_COL

# Invisible characters spammers insert to break up words, e.g. "gacu​r".
ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")
# Letters from other scripts that look like Latin ones and that NFKC does not
# fix, e.g. Cyrillic "AHMаDtoTO", Greek "PulΛᴜ777", Canadian syllabics "ᑭᑌᒪᗩᑌᗯIᑎ".
# Cased scripts only need lowercase keys because lower() runs in between.
HOMOGLYPHS = str.maketrans({
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "я": "r",
    # Greek
    "α": "a", "λ": "a", "ο": "o", "ρ": "p", "ν": "v", "τ": "t", "κ": "k", "ι": "i",
    # Small capitals and Latin extended
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ɢ": "g", "ʜ": "h", "ɪ": "i",
    "ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ʀ": "r",
    "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "ʏ": "y", "ᴢ": "z", "ⱥ": "a",
    # Canadian syllabics and Cherokee (no case, so not touched by lower())
    "ᗩ": "a", "ᗷ": "b", "ᑕ": "c", "ᗪ": "d", "ᗴ": "e", "ᖴ": "f", "ᕼ": "h", "ᒍ": "j",
    "ᒪ": "l", "ᗰ": "m", "ᑎ": "n", "ᑭ": "p", "ᖇ": "r", "ᔕ": "s", "ᑌ": "u", "ᐯ": "v",
    "ᗯ": "w", "᙭": "x", "Ꮤ": "w", "Ꭺ": "a", "Ᏼ": "b", "Ꮯ": "c", "Ꭼ": "e", "Ꮋ": "h",
    "Ꮶ": "k", "Ꮇ": "m", "Ꮲ": "p", "Ꮪ": "s", "Ꭲ": "t",
})
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[\w.\-]+")
# Three or more single characters separated by spaces: "k o i s l o t".
SPACED_LETTERS = re.compile(r"\b(?:\w ){2,}\w\b")
# A word glued to a number, typical for gambling sites: "alexis17", "sgi88".
BRAND_NUMBER = re.compile(r"[a-z]{3,}\d{2,}")


def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)


def clean_data(df):
    """Drop leaky columns, empty rows, and duplicate comments.

    Duplicates must be removed *before* the train/test split, otherwise the
    same comment can land in both sets and inflate the test score.
    """
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].str.strip() != ""]
    # When the same comment appears with different labels, keep the majority label.
    df = (
        df.groupby(TEXT_COL, as_index=False, sort=False)
        .agg({LABEL_COL: lambda s: int(s.mean() >= 0.5), **{
            c: "first" for c in df.columns if c not in (TEXT_COL, LABEL_COL)
        }})
    )
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df.reset_index(drop=True)


def normalize_text(text):
    """Undo the tricks spammers use to dodge keyword filters.

    "𝗪𝗘𝗧𝗢𝗡𝟴𝟴" -> "weton88", "Ｐｕｌａｕ７７７" -> "pulau777",
    "k o i s l o t" -> "koislot", Cyrillic "а" -> Latin "a".
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = ZERO_WIDTH.sub("", text)
    # Translate before and after lower(): Cherokee has case, so "Ꮤ" would become "ꮤ".
    text = text.translate(HOMOGLYPHS).lower().translate(HOMOGLYPHS)
    text = URL.sub(" url ", text)
    text = MENTION.sub(" user ", text)
    text = SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), text)
    return re.sub(r"\s+", " ", text).strip()


def styled_char_ratio(text):
    """Share of characters that NFKC changes (fancy fonts, full-width, etc.)"""
    text = str(text)
    if not text:
        return 0.0
    changed = sum(unicodedata.normalize("NFKC", ch) != ch for ch in text)
    return changed / len(text)


def non_latin_letter_ratio(text):
    """Share of letters outside plain a-z after NFKC.

    Normal Indonesian comments are almost pure ASCII, so exotic letters
    (Greek, Cherokee, syllabics...) are a strong spam signal on their own.
    """
    letters = [ch for ch in unicodedata.normalize("NFKC", str(text)) if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(not ch.isascii() for ch in letters) / len(letters)


def text_stats(text):
    """Hand-crafted numeric features computed from the raw comment"""
    raw = str(text)
    norm = normalize_text(raw)
    n = max(len(norm), 1)
    return [
        styled_char_ratio(raw),
        non_latin_letter_ratio(raw),
        sum(ch.isdigit() for ch in norm) / n,
        len(BRAND_NUMBER.findall(norm)),
        np.log1p(len(norm)),
        int(bool(ZERO_WIDTH.search(raw))),
    ]


TEXT_STAT_NAMES = [
    "styled_char_ratio", "non_latin_letter_ratio", "digit_ratio", "brand_number_count", "log_length", "has_zero_width",
]


class TextStats(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper so text_stats() can live inside a Pipeline"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([text_stats(t) for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(TEXT_STAT_NAMES)


def feature_engineering(df):
    """Add normalized text and numeric stats columns (used for EDA)"""
    df = df.copy()
    df["text_norm"] = df[TEXT_COL].map(normalize_text)
    stats = pd.DataFrame(df[TEXT_COL].map(text_stats).tolist(), columns=TEXT_STAT_NAMES, index=df.index)
    return pd.concat([df, stats], axis=1)
