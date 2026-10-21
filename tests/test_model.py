import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model, top_features
from src.model import build_pipeline, select_best_model

SPAM = ["gacor di alexis17", "maxwin 𝐒𝐆𝐈𝟖𝟖 mantap", "jp terus di pulau777", "depo 50 wd 2jt weton88"]
HAM = ["mobilnya keren bang", "kapan review motor listrik", "suaranya kurang jelas", "mantap videonya"]


def _data(repeat=5):
    X = pd.Series((SPAM + HAM) * repeat)
    y = pd.Series(([1] * len(SPAM) + [0] * len(HAM)) * repeat)
    return X, y


def test_pipeline_learns_toy_data():
    X, y = _data()
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert metrics["f1"] == 1.0
    assert model.predict(pd.Series(["gacor parah di toto88"]))[0] == 1
    assert set(top_features(model, n=3)) == {"judol", "normal"}


def test_select_best_model_prefers_simpler_within_noise():
    results = {
        "simple": {"f1": 0.980, "f1_std": 0.01},
        "complex": {"f1": 0.985, "f1_std": 0.01},
    }
    assert select_best_model(results) == "simple"
    results["simple"]["f1"] = 0.90
    assert select_best_model(results) == "complex"
