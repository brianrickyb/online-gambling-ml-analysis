from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE, TEST_SIZE
from src.preprocessing import TextStats, normalize_text


def build_features():
    """Text -> numeric matrix.

    - word TF-IDF: catches words like "gacor", "maxwin", "sultan"
    - char TF-IDF: catches brand fragments like "88", "toto", "slot" even when misspelled
    - TextStats: fancy-font ratio, digit ratio, etc. (scaled to 0..1 so NB can use it)
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(
            preprocessor=normalize_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            preprocessor=normalize_text, analyzer="char_wb", ngram_range=(2, 5),
            min_df=2, sublinear_tf=True,
        )),
        ("stats", Pipeline([("extract", TextStats()), ("scale", MinMaxScaler())])),
    ])


def get_candidate_models():
    """Classifiers to compare. Linear models are usually strongest on sparse text."""
    return {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(
            C=10, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "linear_svm": LinearSVC(C=1, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(classifier):
    """Full pipeline: raw comment text in, label out"""
    return Pipeline([("features", build_features()), ("clf", classifier)])


def split_data(X, y):
    """Stratified split keeps the judol/normal ratio equal in train and test"""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )


def compare_models(X_train, y_train, models=None, cv=5):
    """Cross-validate every candidate on the training set only"""
    models = models or get_candidate_models()
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, clf in models.items():
        scores = cross_validate(
            build_pipeline(clf), X_train, y_train, cv=folds,
            scoring=["accuracy", "precision", "recall", "f1"], n_jobs=1,
        )
        results[name] = {
            metric: float(scores[f"test_{metric}"].mean())
            for metric in ["accuracy", "precision", "recall", "f1"]
        }
        results[name]["f1_std"] = float(scores["test_f1"].std())
        results[name]["fit_seconds"] = float(scores["fit_time"].mean())
        print(f"  {name:<20} F1={results[name]['f1']:.4f} ± {results[name]['f1_std']:.4f}")
    return results


def select_best_model(cv_results):
    """One-standard-error rule: among models whose F1 is within one std of the
    best, pick the simplest (earliest in get_candidate_models order). A 0.001
    F1 gap that is smaller than the fold-to-fold noise is not a real win.
    """
    best = max(cv_results.values(), key=lambda r: r["f1"])
    threshold = best["f1"] - best["f1_std"]
    return next(name for name, r in cv_results.items() if r["f1"] >= threshold)


# Searched only for logistic regression, the model chosen by select_best_model().
PARAM_GRID = {
    "clf__C": [1, 10, 30],
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "features__char__ngram_range": [(2, 5), (3, 6)],
}


def tune_model(X_train, y_train, cv=5):
    """Grid search over PARAM_GRID with stratified CV on the training set only"""
    search = GridSearchCV(
        build_pipeline(get_candidate_models()["logistic_regression"]),
        PARAM_GRID,
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def train_model(X, y, classifier=None):
    """Split, train one pipeline, and return it with the held-out test set"""
    X_train, X_test, y_train, y_test = split_data(X, y)
    classifier = classifier or get_candidate_models()["logistic_regression"]
    model = build_pipeline(classifier)
    model.fit(X_train, y_train)
    return model, X_test, y_test
