from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

TRAIN_LOG = RAW_DATA / "log_standard_4_08_to_4_21_pure.csv"
TEST_LOG = RAW_DATA / "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FEATURES = RAW_DATA / "video_features_basic_pure.csv"

RANDOM_STATE = 42
TOP_K_VALUES = [5, 10, 20]
LATENT_FACTORS = 50


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_time"] = pd.to_datetime(df["time_ms"], unit="ms")
    return df


def positive_interactions(events: pd.DataFrame) -> pd.DataFrame:
    positive = events[
        (events["is_click"] == 1)
        | (events["long_view"] == 1)
        | (events["is_like"] == 1)
        | (events["is_follow"] == 1)
        | (events["is_comment"] == 1)
        | (events["is_forward"] == 1)
    ].copy()
    return positive[["user_id", "video_id"]].drop_duplicates()


def build_id_maps(train_positive: pd.DataFrame) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    users = np.sort(train_positive["user_id"].unique())
    items = np.sort(train_positive["video_id"].unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    item_to_idx = {video_id: idx for idx, video_id in enumerate(items)}
    return user_to_idx, item_to_idx, items


def build_interaction_matrix(
    train_positive: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
) -> csr_matrix:
    rows = train_positive["user_id"].map(user_to_idx)
    cols = train_positive["video_id"].map(item_to_idx)
    data = np.ones(len(train_positive), dtype=np.float32)
    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_to_idx), len(item_to_idx)),
    )


def build_test_relevance(
    test_positive: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
) -> dict[int, set[int]]:
    eligible = test_positive[
        test_positive["user_id"].isin(user_to_idx)
        & test_positive["video_id"].isin(item_to_idx)
    ].copy()
    eligible["user_idx"] = eligible["user_id"].map(user_to_idx)
    eligible["item_idx"] = eligible["video_id"].map(item_to_idx)
    return eligible.groupby("user_idx")["item_idx"].apply(set).to_dict()


def popularity_scores(train_positive: pd.DataFrame, item_to_idx: dict[int, int]) -> np.ndarray:
    counts = train_positive["video_id"].value_counts()
    scores = np.zeros(len(item_to_idx), dtype=np.float32)
    for video_id, count in counts.items():
        if video_id in item_to_idx:
            scores[item_to_idx[video_id]] = count
    return scores


def recommend_from_scores(
    scores: np.ndarray,
    seen_item_indices: set[int],
    k: int,
) -> list[int]:
    adjusted = scores.copy()
    if seen_item_indices:
        adjusted[list(seen_item_indices)] = -np.inf
    top_items = np.argpartition(adjusted, -k)[-k:]
    top_items = top_items[np.argsort(adjusted[top_items])[::-1]]
    return top_items.tolist()


def dcg_at_k(recommended: list[int], relevant: set[int]) -> float:
    return sum(
        1 / np.log2(rank + 2)
        for rank, item_idx in enumerate(recommended)
        if item_idx in relevant
    )


def evaluate_recommendations(
    model_name: str,
    recommendations: dict[int, list[int]],
    test_relevance: dict[int, set[int]],
    k: int,
) -> dict:
    precisions = []
    recalls = []
    hits = []
    ndcgs = []

    for user_idx, relevant in test_relevance.items():
        recs = recommendations[user_idx][:k]
        true_positives = len(set(recs) & relevant)
        precisions.append(true_positives / k)
        recalls.append(true_positives / len(relevant))
        hits.append(1 if true_positives > 0 else 0)
        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg_at_k(recs, relevant) / ideal_dcg if ideal_dcg > 0 else 0)

    return {
        "model": model_name,
        "k": k,
        "evaluated_users": len(test_relevance),
        "precision_at_k": np.mean(precisions),
        "recall_at_k": np.mean(recalls),
        "hit_rate_at_k": np.mean(hits),
        "ndcg_at_k": np.mean(ndcgs),
    }


def popularity_recommendations(
    train_matrix: csr_matrix,
    base_scores: np.ndarray,
    users: list[int],
    max_k: int,
) -> dict[int, list[int]]:
    recommendations = {}
    for user_idx in users:
        seen_items = set(train_matrix[user_idx].indices)
        recommendations[user_idx] = recommend_from_scores(base_scores, seen_items, max_k)
    return recommendations


def svd_recommendations(
    train_matrix: csr_matrix,
    users: list[int],
    max_k: int,
) -> dict[int, list[int]]:
    svd = TruncatedSVD(
        n_components=min(LATENT_FACTORS, train_matrix.shape[1] - 1),
        random_state=RANDOM_STATE,
    )
    user_factors = svd.fit_transform(train_matrix)
    item_factors = svd.components_.T

    recommendations = {}
    for user_idx in users:
        scores = user_factors[user_idx] @ item_factors.T
        seen_items = set(train_matrix[user_idx].indices)
        recommendations[user_idx] = recommend_from_scores(scores, seen_items, max_k)
    return recommendations


def segment_users(train_positive: pd.DataFrame, user_to_idx: dict[int, int]) -> pd.DataFrame:
    counts = train_positive.groupby("user_id").size().rename("train_positive_items")
    segments = counts.reset_index()
    segments["user_idx"] = segments["user_id"].map(user_to_idx)
    segments["activity_segment"] = pd.qcut(
        segments["train_positive_items"].rank(method="first"),
        q=4,
        labels=["low", "medium", "high", "very_high"],
    )
    return segments[["user_idx", "activity_segment", "train_positive_items"]]


def evaluate_by_segment(
    model_name: str,
    recommendations: dict[int, list[int]],
    test_relevance: dict[int, set[int]],
    segments: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    segment_lookup = segments.set_index("user_idx")["activity_segment"].to_dict()
    rows = []
    for segment in ["low", "medium", "high", "very_high"]:
        users = [
            user_idx
            for user_idx in test_relevance
            if segment_lookup.get(user_idx) == segment
        ]
        if not users:
            continue
        subset_relevance = {user_idx: test_relevance[user_idx] for user_idx in users}
        subset_recommendations = {user_idx: recommendations[user_idx] for user_idx in users}
        metrics = evaluate_recommendations(model_name, subset_recommendations, subset_relevance, k)
        metrics["activity_segment"] = segment
        rows.append(metrics)
    return pd.DataFrame(rows)


def save_figures(metrics: pd.DataFrame, segment_metrics: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]:
        pivot = metrics.pivot(index="k", columns="model", values=metric)
        pivot.plot(marker="o", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by K")
        ax.set_xlabel("K")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(title="Model")
        fig.tight_layout()
        fig.savefig(FIGURES / f"phase6_{metric}.png", dpi=150)
        ax.clear()
    plt.close(fig)

    k10 = segment_metrics[segment_metrics["k"] == 10]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in k10.groupby("model"):
        ax.plot(
            group["activity_segment"].astype(str),
            group["ndcg_at_k"],
            marker="o",
            label=model_name,
        )
    ax.set_title("NDCG@10 by User Activity Segment")
    ax.set_xlabel("Training-period activity segment")
    ax.set_ylabel("NDCG@10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase6_ndcg_at_10_by_activity_segment.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    train_events = load_log(TRAIN_LOG)
    test_events = load_log(TEST_LOG)
    train_positive = positive_interactions(train_events)
    test_positive = positive_interactions(test_events)

    user_to_idx, item_to_idx, items = build_id_maps(train_positive)
    train_matrix = build_interaction_matrix(train_positive, user_to_idx, item_to_idx)
    test_relevance = build_test_relevance(test_positive, user_to_idx, item_to_idx)
    evaluated_users = list(test_relevance.keys())
    max_k = max(TOP_K_VALUES)

    pop_scores = popularity_scores(train_positive, item_to_idx)
    pop_recs = popularity_recommendations(train_matrix, pop_scores, evaluated_users, max_k)
    svd_recs = svd_recommendations(train_matrix, evaluated_users, max_k)

    metrics = []
    for k in TOP_K_VALUES:
        metrics.append(evaluate_recommendations("popularity", pop_recs, test_relevance, k))
        metrics.append(evaluate_recommendations("matrix_factorization_svd", svd_recs, test_relevance, k))
    metrics = pd.DataFrame(metrics)

    segments = segment_users(train_positive, user_to_idx)
    segment_metrics = pd.concat(
        [
            evaluate_by_segment("popularity", pop_recs, test_relevance, segments, 10),
            evaluate_by_segment("matrix_factorization_svd", svd_recs, test_relevance, segments, 10),
        ],
        ignore_index=True,
    )

    coverage = pd.DataFrame(
        [
            {
                "train_positive_interactions": len(train_positive),
                "test_positive_interactions": len(test_positive),
                "train_users": len(user_to_idx),
                "train_items": len(item_to_idx),
                "test_users_with_known_train_history_and_known_test_items": len(test_relevance),
                "test_positive_users_total": test_positive["user_id"].nunique(),
                "test_positive_items_total": test_positive["video_id"].nunique(),
                "test_positive_items_seen_in_train_share": test_positive["video_id"].isin(items).mean(),
            }
        ]
    )

    metrics.to_csv(REPORTS / "phase6_recommendation_metrics.csv", index=False)
    segment_metrics.to_csv(REPORTS / "phase6_recommendation_metrics_by_segment.csv", index=False)
    coverage.to_csv(REPORTS / "phase6_recommendation_evaluation_coverage.csv", index=False)
    save_figures(metrics, segment_metrics)

    print("\n=== Recommendation evaluation coverage ===")
    print(coverage.to_string(index=False))

    print("\n=== Overall recommendation metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== NDCG@10 by activity segment ===")
    print(
        segment_metrics[
            ["model", "activity_segment", "evaluated_users", "precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
        ].to_string(index=False)
    )

    print("\nSaved Phase 6 recommendation metrics and figures.")


if __name__ == "__main__":
    main()
