from pathlib import Path
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve


warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

RANDOM_STATE = 42
TARGET = "disengaged_late"
ID_AND_OUTCOME_COLUMNS = {
    "user_id",
    "obs_first_event_time",
    "obs_last_event_time",
    "future_events",
    "future_active_days",
    "returned_late",
    "disengaged_late",
}


def load_modeling_table() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA / "phase3_user_behavior_features.csv")


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [col for col in df.columns if col not in ID_AND_OUTCOME_COLUMNS]
    return df[feature_columns], df[TARGET].astype(int)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        min_samples_leaf=30,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def threshold_for_top_share(scores: np.ndarray, share: float) -> float:
    return float(np.quantile(scores, 1 - share))


def evaluate_scores(
    model_name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "brier_score": brier_score_loss(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def evaluate_top_k(model_name: str, y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for share in [0.01, 0.05, 0.10, 0.20]:
        threshold = threshold_for_top_share(scores, share)
        metrics = evaluate_scores(
            f"{model_name}_top_{int(share * 100)}pct",
            y_true,
            scores,
            threshold,
        )
        metrics["flagged_share"] = share
        rows.append(metrics)
    return pd.DataFrame(rows)


def save_curves(y_test: pd.Series, model_scores: dict[str, np.ndarray]) -> None:
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, label=f"{model_name} AUC={roc_auc_score(y_test, scores):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_title("ROC Curves for Disengagement Models")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    prevalence = y_test.mean()
    for model_name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, label=f"{model_name} AP={average_precision_score(y_test, scores):.3f}")
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"prevalence={prevalence:.3f}")
    ax.set_title("Precision-Recall Curves for Disengagement Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_precision_recall_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, scores in model_scores.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_test, scores, n_bins=10, strategy="quantile"
        )
        ax.plot(mean_predicted, fraction_positive, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_title("Calibration Curves for Disengagement Models")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed disengagement rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase4_calibration_curves.png", dpi=150)
    plt.close(fig)


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    )
    return result.sort_values("abs_coefficient", ascending=False)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df = load_modeling_table()
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    baseline_scores = np.zeros(len(y_test))
    baseline_metrics = evaluate_scores(
        "all_users_return_baseline",
        y_test,
        baseline_scores,
        threshold=0.5,
    )

    models = build_models(X)
    model_scores = {}
    all_metrics = [baseline_metrics]
    top_share_metrics = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        model_scores[model_name] = scores

        default_threshold_metrics = evaluate_scores(
            model_name,
            y_test,
            scores,
            threshold=0.5,
        )
        all_metrics.append(default_threshold_metrics)
        top_share_metrics.append(evaluate_top_k(model_name, y_test, scores))

        if model_name == "logistic_regression":
            logistic_coefficients(model).to_csv(
                REPORTS / "phase4_logistic_coefficients.csv", index=False
            )

    metrics = pd.DataFrame(all_metrics)
    top_share = pd.concat(top_share_metrics, ignore_index=True)
    metrics.to_csv(REPORTS / "phase4_model_metrics.csv", index=False)
    top_share.to_csv(REPORTS / "phase4_top_share_threshold_metrics.csv", index=False)
    save_curves(y_test, model_scores)

    print("\n=== Train/test class balance ===")
    print(
        pd.DataFrame(
            {
                "split": ["train", "test"],
                "users": [len(y_train), len(y_test)],
                "disengaged_users": [int(y_train.sum()), int(y_test.sum())],
                "disengagement_rate": [y_train.mean(), y_test.mean()],
            }
        ).to_string(index=False)
    )

    print("\n=== Model metrics at 0.5 threshold ===")
    print(metrics.to_string(index=False))

    print("\n=== Top-share intervention metrics ===")
    print(top_share.to_string(index=False))

    print("\nSaved Phase 4 model metrics, coefficient table, and curves.")


if __name__ == "__main__":
    main()
