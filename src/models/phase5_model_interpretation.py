from pathlib import Path

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def load_modeling_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DATA / "phase3_user_behavior_features.csv")
    feature_columns = [col for col in df.columns if col not in ID_AND_OUTCOME_COLUMNS]
    return df[feature_columns], df[TARGET].astype(int)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_logistic_model(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
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
    )


def build_random_forest_model(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
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
    )


def clean_feature_name(feature: str) -> str:
    return feature.replace("numeric__", "").replace("categorical__", "")


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    result = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in feature_names],
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "direction": np.where(coefs > 0, "higher_risk", "lower_risk"),
            "abs_coefficient": np.abs(coefs),
        }
    )
    return result.sort_values("abs_coefficient", ascending=False)


def model_based_feature_importance(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    result = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in feature_names],
            "gini_importance": importances,
        }
    )
    return result.sort_values("gini_importance", ascending=False)


def permutation_importance_table(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean_pr_auc_drop": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean_pr_auc_drop", ascending=False)


def risk_by_feature_bins(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    risk_scores: np.ndarray,
    feature: str,
    bins: int = 5,
) -> pd.DataFrame:
    temp = X_test[[feature]].copy()
    temp["actual_disengaged"] = y_test.values
    temp["predicted_risk"] = risk_scores
    temp["bin"] = pd.qcut(temp[feature].rank(method="first"), q=bins, labels=False)
    grouped = (
        temp.groupby("bin")
        .agg(
            users=(feature, "size"),
            feature_min=(feature, "min"),
            feature_max=(feature, "max"),
            feature_median=(feature, "median"),
            observed_disengagement_rate=("actual_disengaged", "mean"),
            mean_predicted_risk=("predicted_risk", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "feature", feature)
    return grouped


def product_hypotheses() -> pd.DataFrame:
    rows = [
        {
            "model_signal": "High recency: many days since last event",
            "possible_product_hypothesis": "Users who stop appearing before the observation window ends may already be disengaging.",
            "possible_intervention": "Test a re-entry feed that emphasizes familiar high-affinity content when the user returns.",
            "evidence_type": "Predictive association",
            "causal_warning": "We do not know whether changing the feed would cause the user to return.",
        },
        {
            "model_signal": "Low or declining activity",
            "possible_product_hypothesis": "A drop in activity may be an early warning sign that the feed is losing relevance.",
            "possible_intervention": "Test a lightweight exploration module or refreshed recommendation mix for declining users.",
            "evidence_type": "Predictive association",
            "causal_warning": "Decline may be caused by outside factors, not only recommendation quality.",
        },
        {
            "model_signal": "Content diversity and repeat behavior",
            "possible_product_hypothesis": "Users may differ in whether they need novelty or familiar content to stay engaged.",
            "possible_intervention": "Segment recommendation strategies by narrow-interest vs broad-interest users.",
            "evidence_type": "Predictive association",
            "causal_warning": "Observed diversity is partly created by the recommender's exposure policy.",
        },
        {
            "model_signal": "Weak deep-engagement signals",
            "possible_product_hypothesis": "Users who click but do not long-view, like, or otherwise deeply engage may be less satisfied.",
            "possible_intervention": "Test ranking that optimizes for long-view quality, not only click probability.",
            "evidence_type": "Predictive association",
            "causal_warning": "Offline associations must be validated with an experiment before launch.",
        },
    ]
    return pd.DataFrame(rows)


def save_figures(
    logistic: pd.DataFrame,
    permutation: pd.DataFrame,
    risk_bins: pd.DataFrame,
) -> None:
    FIGURES.mkdir(exist_ok=True)

    top_logistic = logistic.head(12).sort_values("coefficient")
    colors = np.where(top_logistic["coefficient"] > 0, "#b44b4b", "#3f6f9f")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_logistic["feature"], top_logistic["coefficient"], color=colors)
    ax.set_title("Largest Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient on standardized / encoded feature")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase5_logistic_top_coefficients.png", dpi=150)
    plt.close(fig)

    top_perm = permutation.head(12).sort_values("importance_mean_pr_auc_drop")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_perm["feature"], top_perm["importance_mean_pr_auc_drop"], color="#4f7f6f")
    ax.set_title("Permutation Importance for Random Forest")
    ax.set_xlabel("Mean PR-AUC drop when feature is shuffled")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase5_random_forest_permutation_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for feature, group in risk_bins.groupby("feature"):
        ax.plot(
            group["feature_median"],
            group["mean_predicted_risk"],
            marker="o",
            label=feature,
        )
    ax.set_title("Predicted Risk Across Feature Bins")
    ax.set_xlabel("Feature bin median")
    ax.set_ylabel("Mean predicted disengagement risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase5_risk_by_feature_bins.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    X, y = load_modeling_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logistic_model = build_logistic_model(X)
    random_forest_model = build_random_forest_model(X)
    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    rf_scores = random_forest_model.predict_proba(X_test)[:, 1]
    logistic = logistic_coefficients(logistic_model)
    gini = model_based_feature_importance(random_forest_model)
    permutation = permutation_importance_table(random_forest_model, X_test, y_test)

    risk_features = [
        "obs_days_since_last_event",
        "obs_events",
        "obs_event_count_ratio",
        "obs_tag_entropy",
    ]
    risk_bins = pd.concat(
        [risk_by_feature_bins(X_test, y_test, rf_scores, feature) for feature in risk_features],
        ignore_index=True,
    )
    hypotheses = product_hypotheses()

    logistic.to_csv(REPORTS / "phase5_logistic_interpretation.csv", index=False)
    gini.to_csv(REPORTS / "phase5_random_forest_gini_importance.csv", index=False)
    permutation.to_csv(REPORTS / "phase5_random_forest_permutation_importance.csv", index=False)
    risk_bins.to_csv(REPORTS / "phase5_risk_by_feature_bins.csv", index=False)
    hypotheses.to_csv(REPORTS / "phase5_product_hypotheses.csv", index=False)
    save_figures(logistic, permutation, risk_bins)

    print("\n=== Random forest validation ranking metrics ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_scores):.3f}")
    print(f"PR-AUC: {average_precision_score(y_test, rf_scores):.3f}")

    print("\n=== Top logistic regression coefficients ===")
    print(
        logistic[
            ["feature", "coefficient", "odds_ratio", "direction"]
        ].head(12).to_string(index=False)
    )

    print("\n=== Top random forest permutation importances ===")
    print(permutation.head(12).to_string(index=False))

    print("\n=== Product hypotheses ===")
    print(hypotheses.to_string(index=False))

    print("\nSaved Phase 5 interpretation tables and figures.")


if __name__ == "__main__":
    main()
