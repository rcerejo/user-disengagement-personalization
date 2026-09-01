from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

EARLY_STANDARD_LOG = RAW_DATA / "log_standard_4_08_to_4_21_pure.csv"
LATE_STANDARD_LOG = RAW_DATA / "log_standard_4_22_to_5_08_pure.csv"
USER_FEATURES = RAW_DATA / "user_features_pure.csv"
VIDEO_FEATURES_BASIC = RAW_DATA / "video_features_basic_pure.csv"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["event_time"] = pd.to_datetime(df["time_ms"], unit="ms")
    df["event_date"] = pd.to_datetime(df["event_time"].dt.date)
    return df


def safe_entropy(values: pd.Series) -> float:
    counts = values.dropna().value_counts()
    if counts.empty:
        return 0.0
    shares = counts / counts.sum()
    return float(-(shares * shares.map(lambda x: __import__("math").log2(x))).sum())


def build_behavioral_features(early: pd.DataFrame) -> pd.DataFrame:
    video_features = pd.read_csv(
        VIDEO_FEATURES_BASIC,
        usecols=["video_id", "author_id", "video_type", "upload_type", "tag"],
    )
    events = early.merge(video_features, on="video_id", how="left")
    observation_end = events["event_time"].max()

    valid_duration = events["duration_ms"] > 0
    events["watch_ratio"] = np.nan
    events.loc[valid_duration, "watch_ratio"] = (
        events.loc[valid_duration, "play_time_ms"]
        / events.loc[valid_duration, "duration_ms"]
    )
    events["deep_engagement"] = (
        (events["long_view"] == 1)
        | (events["is_like"] == 1)
        | (events["is_follow"] == 1)
        | (events["is_comment"] == 1)
        | (events["is_forward"] == 1)
    ).astype(int)

    midpoint = events["event_time"].min() + (
        events["event_time"].max() - events["event_time"].min()
    ) / 2
    events["half"] = pd.Series("first_half", index=events.index)
    events.loc[events["event_time"] > midpoint, "half"] = "second_half"

    base = (
        events.groupby("user_id")
        .agg(
            obs_events=("video_id", "size"),
            obs_active_days=("event_date", "nunique"),
            obs_unique_videos=("video_id", "nunique"),
            obs_unique_authors=("author_id", "nunique"),
            obs_unique_tags=("tag", "nunique"),
            obs_click_rate=("is_click", "mean"),
            obs_long_view_rate=("long_view", "mean"),
            obs_like_rate=("is_like", "mean"),
            obs_follow_rate=("is_follow", "mean"),
            obs_comment_rate=("is_comment", "mean"),
            obs_forward_rate=("is_forward", "mean"),
            obs_hate_rate=("is_hate", "mean"),
            obs_deep_engagement_rate=("deep_engagement", "mean"),
            obs_avg_play_time_ms=("play_time_ms", "mean"),
            obs_median_play_time_ms=("play_time_ms", "median"),
            obs_avg_watch_ratio=("watch_ratio", "mean"),
            obs_median_watch_ratio=("watch_ratio", "median"),
            obs_total_profile_stay_time=("profile_stay_time", "sum"),
            obs_total_comment_stay_time=("comment_stay_time", "sum"),
            obs_first_event_time=("event_time", "min"),
            obs_last_event_time=("event_time", "max"),
        )
        .reset_index()
    )
    base["obs_days_since_last_event"] = (
        observation_end - base["obs_last_event_time"]
    ).dt.total_seconds() / 86400
    base["obs_events_per_active_day"] = base["obs_events"] / base["obs_active_days"]
    base["obs_video_repeat_rate"] = 1 - (base["obs_unique_videos"] / base["obs_events"])
    base["obs_author_repeat_rate"] = 1 - (base["obs_unique_authors"] / base["obs_events"])

    entropy = (
        events.groupby("user_id")
        .agg(
            obs_tag_entropy=("tag", safe_entropy),
            obs_upload_type_entropy=("upload_type", safe_entropy),
        )
        .reset_index()
    )

    half_counts = (
        events.groupby(["user_id", "half"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["first_half", "second_half"]:
        if col not in half_counts.columns:
            half_counts[col] = 0
    half_counts["obs_event_count_change"] = (
        half_counts["second_half"] - half_counts["first_half"]
    )
    half_counts["obs_event_count_ratio"] = (
        (half_counts["second_half"] + 1) / (half_counts["first_half"] + 1)
    )
    half_counts = half_counts[
        ["user_id", "first_half", "second_half", "obs_event_count_change", "obs_event_count_ratio"]
    ].rename(
        columns={
            "first_half": "obs_events_first_half",
            "second_half": "obs_events_second_half",
        }
    )

    user_features = pd.read_csv(
        USER_FEATURES,
        usecols=[
            "user_id",
            "user_active_degree",
            "is_lowactive_period",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num",
            "fans_user_num",
            "friend_user_num",
            "register_days",
        ],
    )

    features = (
        base.merge(entropy, on="user_id", how="left")
        .merge(half_counts, on="user_id", how="left")
        .merge(user_features, on="user_id", how="left")
    )
    return features


def add_future_label(features: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    late_user_activity = (
        late.groupby("user_id")
        .agg(
            future_events=("video_id", "size"),
            future_active_days=("event_date", "nunique"),
        )
        .reset_index()
    )
    modeling_table = features.merge(late_user_activity, on="user_id", how="left")
    modeling_table["future_events"] = modeling_table["future_events"].fillna(0).astype(int)
    modeling_table["future_active_days"] = (
        modeling_table["future_active_days"].fillna(0).astype(int)
    )
    modeling_table["returned_late"] = modeling_table["future_events"] > 0
    modeling_table["disengaged_late"] = ~modeling_table["returned_late"]
    return modeling_table


def summarize_features(modeling_table: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "obs_events",
        "obs_active_days",
        "obs_unique_videos",
        "obs_unique_authors",
        "obs_unique_tags",
        "obs_click_rate",
        "obs_long_view_rate",
        "obs_like_rate",
        "obs_hate_rate",
        "obs_deep_engagement_rate",
        "obs_avg_watch_ratio",
        "obs_days_since_last_event",
        "obs_events_per_active_day",
        "obs_video_repeat_rate",
        "obs_author_repeat_rate",
        "obs_tag_entropy",
        "obs_upload_type_entropy",
        "obs_event_count_change",
        "obs_event_count_ratio",
        "future_events",
        "future_active_days",
    ]
    return (
        modeling_table[feature_columns]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )


def save_figures(modeling_table: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(modeling_table["obs_events"], bins=50, log=True)
    ax.set_title("Observation-Window Events Per User")
    ax.set_xlabel("Early-period events")
    ax.set_ylabel("Users, log scale")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase3_obs_events_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        modeling_table.loc[~modeling_table["disengaged_late"], "obs_days_since_last_event"],
        modeling_table.loc[modeling_table["disengaged_late"], "obs_days_since_last_event"],
    ]
    ax.boxplot(groups, tick_labels=["False", "True"], showfliers=False)
    ax.set_title("Recency by Future Disengagement")
    ax.set_xlabel("Disengaged in late period")
    ax.set_ylabel("Days since last early-period event")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase3_recency_by_disengagement.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        modeling_table.loc[~modeling_table["disengaged_late"], "obs_event_count_ratio"],
        modeling_table.loc[modeling_table["disengaged_late"], "obs_event_count_ratio"],
    ]
    ax.boxplot(groups, tick_labels=["False", "True"], showfliers=False)
    ax.set_title("Activity Change by Future Disengagement")
    ax.set_xlabel("Disengaged in late period")
    ax.set_ylabel("Second-half events / first-half events")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase3_activity_change_by_disengagement.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PROCESSED_DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    early = load_events(EARLY_STANDARD_LOG)
    late = load_events(LATE_STANDARD_LOG)
    features = build_behavioral_features(early)
    modeling_table = add_future_label(features, late)
    feature_summary = summarize_features(modeling_table)

    modeling_table.to_csv(PROCESSED_DATA / "phase3_user_behavior_features.csv", index=False)
    feature_summary.to_csv(REPORTS / "phase3_feature_summary.csv", index=False)
    save_figures(modeling_table)

    label_counts = modeling_table["disengaged_late"].value_counts().rename_axis(
        "disengaged_late"
    ).reset_index(name="users")
    label_counts["share"] = label_counts["users"] / len(modeling_table)
    label_counts.to_csv(REPORTS / "phase3_disengagement_label_distribution.csv", index=False)

    print("\n=== Modeling table shape ===")
    print(modeling_table.shape)

    print("\n=== Future disengagement label distribution ===")
    print(label_counts.to_string(index=False))

    print("\n=== Selected feature summary ===")
    selected = feature_summary[
        feature_summary["feature"].isin(
            [
                "obs_events",
                "obs_active_days",
                "obs_click_rate",
                "obs_long_view_rate",
                "obs_days_since_last_event",
                "obs_tag_entropy",
                "obs_event_count_ratio",
            ]
        )
    ]
    print(selected.to_string(index=False))

    print("\nSaved Phase 3 modeling table, summaries, and figures.")


if __name__ == "__main__":
    main()
