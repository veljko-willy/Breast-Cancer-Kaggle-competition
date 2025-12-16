from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path("dataset")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TRUE_LABELS_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = DATA_DIR / "best_model_submission.csv"

LABEL_MAP = {"M": 1, "B": 0}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
RANDOM_STATE = 42


def load_training_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(TRAIN_PATH)
    y = df["label"].map(LABEL_MAP).astype(int)
    X = df.drop(columns=["label", "id"])
    return X, y, df["id"]


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(TEST_PATH)
    return df.drop(columns=["id"]), df["id"]


def load_true_test_labels() -> pd.Series:
    """Returns true labels for the provided test ids (if available)."""
    df_true = pd.read_csv(TRUE_LABELS_PATH)
    return df_true.set_index("id")["label"].map(LABEL_MAP)


def build_model_spaces() -> Dict[str, Dict[str, object]]:
    """Defines the estimators and grids we want to score."""
    return {
        "LogisticRegression": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear", max_iter=500, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 5.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        },
        "SupportVectorClassifier": {
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__C": [0.5, 1.0, 5.0],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_jobs=1, random_state=RANDOM_STATE, oob_score=False
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 6, 12],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        GradientBoostingClassifier(random_state=RANDOM_STATE),
                    )
                ]
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> Tuple[GridSearchCV, pd.DataFrame]:
    model_spaces = build_model_spaces()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard_rows = []
    best_search: GridSearchCV | None = None

    for model_name, config in model_spaces.items():
        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
            refit="roc_auc",
            n_jobs=1,
        )
        search.fit(X, y)

        leaderboard_rows.append(
            {
                "model": model_name,
                "best_roc_auc": search.best_score_,
                "best_accuracy": search.cv_results_["mean_test_accuracy"][
                    search.best_index_
                ],
                "best_params": search.best_params_,
            }
        )

        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by="best_roc_auc", ascending=False
    )
    return best_search, leaderboard


def get_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # bring scores to [0,1] to behave like probabilities for ROC AUC
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - scores.min()) / score_range
    raise AttributeError("Model does not expose predict_proba or decision_function.")


def main() -> None:
    X_train, y_train, _ = load_training_data()
    X_test, test_ids = load_test_data()
    true_test_labels = load_true_test_labels()

    best_search, leaderboard = evaluate_models(X_train, y_train)

    print("\nLeaderboard (sorted by ROC AUC):")
    print(leaderboard.to_string(index=False))

    best_model = best_search.best_estimator_
    print(f"\nBest model: {best_search.best_params_}")

    y_test_pred = best_model.predict(X_test)
    y_test_proba = get_probabilities(best_model, X_test)

    # Align predictions with any true labels we might have (if Kaggle provided them).
    y_true_aligned = true_test_labels.reindex(test_ids).dropna()
    evaluated_ids = y_true_aligned.index
    if not y_true_aligned.empty:
        mask = test_ids.isin(evaluated_ids).to_numpy()
        accuracy = accuracy_score(y_true_aligned, y_test_pred[mask])
        roc_auc = roc_auc_score(y_true_aligned, y_test_proba[mask])
        print(f"\nAccuracy on provided true labels: {accuracy:.4f}")
        print(f"ROC AUC on provided true labels: {roc_auc:.4f}")
        print(
            "\nClassification report:\n",
            classification_report(y_true_aligned, y_test_pred[mask]),
        )
    else:
        print("\nNo ground-truth labels provided for the test set.")

    submission_df = pd.DataFrame(
        {
            "id": test_ids,
            "label": pd.Series(y_test_pred).map(INV_LABEL_MAP),
            "probability_malignant": y_test_proba,
        }
    )
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved submission file to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
