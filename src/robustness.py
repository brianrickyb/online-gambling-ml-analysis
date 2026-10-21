"""Does the model catch gambling sites it has never seen?

The normal random split puts comments about the same site (e.g. "alexis17")
in both train and test, so the test score partly measures memorizing site
names. Here, for each of the most common sites, we train on every comment
that does NOT mention it and test on the comments that do. Recall on that
held-out site is a more honest estimate of how the model handles a brand-new
gambling site.

Usage:
    python -m src.robustness
"""
import json
import re
import sys
from collections import Counter

import pandas as pd

from src.config import DATA_PATH, LABEL_COL, REPORT_DIR, TEXT_COL
from src.model import build_pipeline, get_candidate_models
from src.preprocessing import BRAND_NUMBER, clean_data, load_data, normalize_text


def find_brands(texts, labels, top_n=8, min_len=4):
    """Most common site names in judol comments: the letters of "alexis17" -> "alexis"."""
    counts = Counter()
    for text in texts[labels == 1]:
        for match in set(BRAND_NUMBER.findall(text)):
            brand = re.sub(r"\d+", "", match)
            if len(brand) >= min_len:
                counts[brand] += 1
    return [brand for brand, _ in counts.most_common(top_n)]


def brand_holdout(df, brands):
    """Train without each brand, then measure recall on that brand's comments"""
    norm = df[TEXT_COL].map(normalize_text)
    rows = []
    for brand in brands:
        mentions = norm.str.contains(brand, regex=False)
        train, test = df[~mentions], df[mentions & (df[LABEL_COL] == 1)]
        model = build_pipeline(get_candidate_models()["logistic_regression"])
        model.fit(train[TEXT_COL], train[LABEL_COL])
        recall = float(model.predict(test[TEXT_COL]).mean())
        rows.append({"held_out_brand": brand, "n_test": len(test), "recall_unseen": recall})
        print(f"  {brand:<12} n={len(test):<4} recall on unseen brand = {recall:.3f}")
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = clean_data(load_data(DATA_PATH))
    norm = df[TEXT_COL].map(normalize_text)
    brands = find_brands(norm, df[LABEL_COL])
    print(f"Top brands: {brands}")
    result = brand_holdout(df, brands)
    weighted = (result["recall_unseen"] * result["n_test"]).sum() / result["n_test"].sum()
    print(f"Weighted recall on unseen brands: {weighted:.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"brands": result.to_dict(orient="records"), "weighted_recall": float(weighted)}
    (REPORT_DIR / "robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
