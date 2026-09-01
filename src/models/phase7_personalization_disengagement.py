from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.phase6_recommendation_system import (  # noqa: E402
    TOP_K_VALUES,
    build_id_maps,
    build_interaction_matrix,
    build_test_relevance,
    load_log,
    popularity_recommendations,
    popularity_scores,
    positive_interactions,
    svd_recommendations,
)


RAW_DATA = PROJECT_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

TRAIN_LOG = RAW_DATA / "log_standard_4_08_to_4_21_pure.csv"
TEST_LOG = RAW_DATA / "log_standard_4_22_to_5_08_pure.csv"
FEATURE_TABLE = PROCESSED_DATA / "phase3_user_behavior_features.csv"

K = 10


def dcg_at_k(recommended: list[int], relevant: set[int]) -> float:
    return sum(
        1 / np.log2(rank + 2)
        for rank, item_idx in enumerate(recommended[:K])
        if item_idx in relevant
    )


def user_level_metrics(
    model_name: str,
    recommendations: dict[int, list[int]],
    test_relevance: dict[int, set[int]],
    idx_to_user: dict[int, int],
) -> pd.DataFrame:
    rows = []
    for user_idx, relevant in test_relevance.items():
        recs = recommendations[user_idx][:K]
        hits = len(set(recs) & relevant)
        ideal_hits = min(len(relevant), K)
        ideal_dcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
        rows.append(
            {
                "user_id": idx_to_user[user_idx],
                "model": model_name,
                "k": K,
                "relevant_future_items": len(relevant),
                "precision_at_k": hits / K,
                "recall_at_k": hits / len(relevant),
                "hit_at_k": int(hits > 0),
                "ndcg_at_k": dcg_at_k(recs, relevant) / ideal_dcg if ideal_dcg else 0,
            }
        )
    return pd.DataFrame(rows)


def assign_activity_segment(df: pd.DataFrame) -> pd.Series:
    labels = ["low", "medium", "high", "very_high"]
    return pd.qcut(df["obs_events"].rank(method="first"), q=4, labels=labels)


def assign_decline_segment(df: pd.DataFrame) -> pd.Series:
    bins = [-np.inf, 0.25, 0.75, 1.25, np.inf]
    labels = ["sharp_decline", "moderate_decline", "stable", "increased"]
    return pd.cut(df["obs_event_count_ratio"], bins=bins, labels=labels)


def build_recommendation_metrics() -> pd.DataFrame:
    train_events = load_log(TRAIN_LOG)
    test_events = load_log(TEST_LOG)
    train_positive = positive_interactions(train_events)
    test_positive = positive_interactions(test_events)

    user_to_idx, item_to_idx, _ = build_id_maps(train_positive)
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    train_matrix = build_interaction_matrix(train_positive, user_to_idx, item_to_idx)
    test_relevance = build_test_relevance(test_positive, user_to_idx, item_to_idx)
    evaluated_users = list(test_relevance.keys())

    pop_scores = popularity_scores(train_positive, item_to_idx)
    pop_recs = popularity_recommendations(train_matrix, pop_scores, evaluated_users, max(TOP_K_VALUES))
    svd_recs = svd_recommendations(train_matrix, evaluated_users, max(TOP_K_VALUES))

    return pd.concat(
        [
            user_level_metrics("popularity", pop_recs, test_relevance, idx_to_user),
            user_level_metrics("matrix_factorization_svd", svd_recs, test_relevance, idx_to_user),
        ],
        ignore_index=True,
    )


def summarize_segments(joined: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    return (
        joined.groupby([segment_col, "model"], observed=True)
        .agg(
            users=("user_id", "nunique"),
            mean_precision_at_10=("precision_at_k", "mean"),
            mean_recall_at_10=("recall_at_k", "mean"),
            mean_hit_rate_at_10=("hit_at_k", "mean"),
            mean_ndcg_at_10=("ndcg_at_k", "mean"),
        )
        .reset_index()
    )


def summarize_svd_lift(joined: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    wide = joined.pivot_table(
        index=["user_id", segment_col],
        columns="model",
        values=["precision_at_k", "recall_at_k", "hit_at_k", "ndcg_at_k"],
        observed=True,
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for metric in ["precision_at_k", "recall_at_k", "hit_at_k", "ndcg_at_k"]:
        wide[f"{metric}_svd_minus_popularity"] = (
            wide[f"{metric}_matrix_factorization_svd"] - wide[f"{metric}_popularity"]
        )

    return (
        wide.groupby(segment_col, observed=True)
        .agg(
            users=("user_id", "nunique"),
            mean_precision_lift=("precision_at_k_svd_minus_popularity", "mean"),
            mean_recall_lift=("recall_at_k_svd_minus_popularity", "mean"),
            mean_hit_rate_lift=("hit_at_k_svd_minus_popularity", "mean"),
            mean_ndcg_lift=("ndcg_at_k_svd_minus_popularity", "mean"),
        )
        .reset_index()
    )


def coverage_by_disengagement(features: pd.DataFrame, rec_metrics: pd.DataFrame) -> pd.DataFrame:
    evaluable_users = set(rec_metrics["user_id"])
    temp = features[["user_id", "disengaged_late", "returned_late"]].copy()
    temp["recommendation_evaluable"] = temp["user_id"].isin(evaluable_users)
    return (
        temp.groupby(["disengaged_late", "returned_late"])
        .agg(
            users=("user_id", "nunique"),
            recommendation_evaluable_users=("recommendation_evaluable", "sum"),
            recommendation_evaluable_share=("recommendation_evaluable", "mean"),
        )
        .reset_index()
    )


def save_figures(activity_lift: pd.DataFrame, decline_lift: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(activity_lift["activity_segment"].astype(str), activity_lift["mean_ndcg_lift"])
    ax.set_title("SVD NDCG@10 Lift by Early Activity Segment")
    ax.set_xlabel("Early activity segment")
    ax.set_ylabel("Mean NDCG@10 lift over popularity")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase7_svd_ndcg_lift_by_activity_segment.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(decline_lift["decline_segment"].astype(str), decline_lift["mean_ndcg_lift"])
    ax.set_title("SVD NDCG@10 Lift by Activity Change Segment")
    ax.set_xlabel("Early activity change segment")
    ax.set_ylabel("Mean NDCG@10 lift over popularity")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase7_svd_ndcg_lift_by_decline_segment.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    features = pd.read_csv(FEATURE_TABLE)
    features["activity_segment"] = assign_activity_segment(features)
    features["decline_segment"] = assign_decline_segment(features)

    rec_metrics = build_recommendation_metrics()
    joined = rec_metrics.merge(
        features[
            [
                "user_id",
                "disengaged_late",
                "returned_late",
                "activity_segment",
                "decline_segment",
                "obs_events",
                "obs_event_count_ratio",
                "obs_days_since_last_event",
            ]
        ],
        on="user_id",
        how="left",
    )

    coverage = coverage_by_disengagement(features, rec_metrics)
    activity_summary = summarize_segments(joined, "activity_segment")
    decline_summary = summarize_segments(joined, "decline_segment")
    activity_lift = summarize_svd_lift(joined, "activity_segment")
    decline_lift = summarize_svd_lift(joined, "decline_segment")

    hypotheses = pd.DataFrame(
        [
            {
                "finding": "SVD improves offline ranking more clearly for high-activity users than low-activity users.",
                "interpretation": "Collaborative filtering benefits from richer user histories.",
                "product_implication": "Use matrix-factorization personalization for users with enough history; pair low-history users with exploration, popularity, or content-based approaches.",
                "claim_type": "Offline predictive evidence",
            },
            {
                "finding": "Users who fully disengage have no late positive interactions, so recommendation metrics are undefined for them.",
                "interpretation": "Offline recommender evaluation is conditional on returning and interacting.",
                "product_implication": "To prove retention impact, test personalization in an A/B experiment before users disengage.",
                "claim_type": "Measurement limitation",
            },
            {
                "finding": "Activity-change segments can show where SVD improves or struggles relative to popularity.",
                "interpretation": "Personalization value may vary by whether a user's recent activity is declining, stable, or increasing.",
                "product_implication": "Consider segment-specific personalization tests instead of one global intervention.",
                "claim_type": "Segment-level descriptive evidence",
            },
        ]
    )

    rec_metrics.to_csv(REPORTS / "phase7_user_level_recommendation_metrics.csv", index=False)
    joined.to_csv(REPORTS / "phase7_recommendation_disengagement_joined.csv", index=False)
    coverage.to_csv(REPORTS / "phase7_recommendation_evaluation_coverage_by_disengagement.csv", index=False)
    activity_summary.to_csv(REPORTS / "phase7_recommendation_metrics_by_activity_segment.csv", index=False)
    decline_summary.to_csv(REPORTS / "phase7_recommendation_metrics_by_decline_segment.csv", index=False)
    activity_lift.to_csv(REPORTS / "phase7_svd_lift_by_activity_segment.csv", index=False)
    decline_lift.to_csv(REPORTS / "phase7_svd_lift_by_decline_segment.csv", index=False)
    hypotheses.to_csv(REPORTS / "phase7_personalization_disengagement_hypotheses.csv", index=False)
    save_figures(activity_lift, decline_lift)

    print("\n=== Recommendation evaluability by disengagement label ===")
    print(coverage.to_string(index=False))

    print("\n=== SVD lift over popularity by activity segment ===")
    print(activity_lift.to_string(index=False))

    print("\n=== SVD lift over popularity by activity-change segment ===")
    print(decline_lift.to_string(index=False))

    print("\n=== Phase 7 hypotheses ===")
    print(hypotheses.to_string(index=False))

    print("\nSaved Phase 7 joined metrics, segment summaries, and figures.")


if __name__ == "__main__":
    main()
