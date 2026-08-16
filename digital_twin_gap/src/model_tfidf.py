"""
model_tfidf.py
==============
ARCHITECTURE 1 — TF-IDF + Logistic Regression.

The simplest of the three twins and the one that always runs (pure CPU,
scikit-learn, trains in seconds). It turns each shopping prompt into a sparse
bag-of-n-grams vector and fits a multinomial logistic-regression classifier
over the four choices A/B/C/D.

Why include it: it is the honest floor. If a heavy transformer can't beat a
TF-IDF baseline, that's a finding. It also transfers surprisingly poorly across
the human/agent boundary, which makes the "gap" visible with almost no compute.

Every model file exposes the SAME two functions so run_all.py can treat them
interchangeably:

    fit(train_rows, cfg)                     -> a fitted model object
    predict(model, test_rows)                -> list[str] predicted labels
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def fit(train_rows, cfg):
    p = cfg["models"]["tfidf_logreg"]
    X = [r["prompt"] for r in train_rows]
    y = [r["label"] for r in train_rows]

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=p["max_features"],
                    ngram_range=(1, p["ngram_max"]),
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=p["C"],
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def predict(model, test_rows):
    X = [r["prompt"] for r in test_rows]
    return list(model.predict(X))
