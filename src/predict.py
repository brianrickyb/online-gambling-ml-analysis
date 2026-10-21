"""Classify new comments with the trained model.

Usage:
    python -m src.predict "mantap videonya bang" "gacor parah di 𝐒𝐆𝐈𝟖𝟖"
"""
import argparse
import sys

import joblib

from src.config import LABEL_NAMES, MODEL_PATH


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    return joblib.load(path)


def predict(texts, model=None):
    """Return a list of (text, label_name, judol_score) tuples"""
    model = model or load_model()
    labels = model.predict(texts)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1]
    else:
        scores = model.decision_function(texts)
    return [(t, LABEL_NAMES[int(l)], float(s)) for t, l, s in zip(texts, labels, scores)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="+")
    for text, label, score in predict(parser.parse_args().texts):
        print(f"[{label:>6}] score={score:.3f}  {text}")
