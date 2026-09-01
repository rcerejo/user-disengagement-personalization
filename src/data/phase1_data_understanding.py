from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

LOG_FILES = {
    "standard_early": RAW_DATA / "log_standard_4_08_to_4_21_pure.csv",
    "standard_late": RAW_DATA / "log_standard_4_22_to_5_08_pure.csv",
    "random_late": RAW_DATA / "log_random_4_22_to_5_08_pure.csv",
}

FEEDBACK_COLUMNS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
]

METADATA_FILES = {
    "user_features": RAW_DATA / "user_features_pure.csv",
    "video_features_basic": RAW_DATA / "video_features_basic_pure.csv",
    "video_features_statistic": RAW_DATA / "video_features_statistic_pure.csv",
}


def load_log(path: Path) -> pd.DataFrame:
    """Load one interaction log and add readable timestamp fields."""
    df = pd.read_csv(path)
    df["event_time"] = pd.to_datetime(df["time_ms"], unit="ms")
    df["event_date"] = df["event_time"].dt.date
    return df


def summarize_log(name: str, df: pd.DataFrame) -> dict:
    summary = {
        "log": name,
        "rows": len(df),
        "users": df["user_id"].nunique(),
        "videos": df["video_id"].nunique(),
        "first_event_time": df["event_time"].min(),
        "last_event_time": df["event_time"].max(),
        "active_days": df["event_date"].nunique(),
        "mean_events_per_user": len(df) / df["user_id"].nunique(),
        "median_events_per_user": df.groupby("user_id").size().median(),
        "mean_events_per_video": len(df) / df["video_id"].nunique(),
        "median_events_per_video": df.groupby("video_id").size().median(),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in FEEDBACK_COLUMNS:
        if col in df.columns:
            summary[f"{col}_rate"] = df[col].mean()

    if {"play_time_ms", "duration_ms"}.issubset(df.columns):
        valid_duration = df["duration_ms"] > 0
        watch_ratio = df.loc[valid_duration, "play_time_ms"] / df.loc[valid_duration, "duration_ms"]
        summary["median_watch_ratio"] = watch_ratio.median()
        summary["p90_watch_ratio"] = watch_ratio.quantile(0.90)
        summary["zero_duration_rows"] = int((~valid_duration).sum())

    return summary


def user_activity_distribution(df: pd.DataFrame, log_name: str) -> pd.DataFrame:
    user_counts = df.groupby("user_id").size()
    return pd.DataFrame(
        {
            "log": [log_name],
            "min": [user_counts.min()],
            "p25": [user_counts.quantile(0.25)],
            "median": [user_counts.median()],
            "mean": [user_counts.mean()],
            "p75": [user_counts.quantile(0.75)],
            "p90": [user_counts.quantile(0.90)],
            "p99": [user_counts.quantile(0.99)],
            "max": [user_counts.max()],
        }
    )


def daily_product_metrics(df: pd.DataFrame, log_name: str) -> pd.DataFrame:
    daily = (
        df.groupby("event_date")
        .agg(
            active_users=("user_id", "nunique"),
            events=("user_id", "size"),
            videos_seen=("video_id", "nunique"),
            click_rate=("is_click", "mean"),
            long_view_rate=("long_view", "mean"),
            like_rate=("is_like", "mean"),
            hate_rate=("is_hate", "mean"),
            avg_play_time_ms=("play_time_ms", "mean"),
        )
        .reset_index()
    )
    daily.insert(0, "log", log_name)
    return daily


def infer_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Infer sessions from inactivity gaps inside each user timeline."""
    ordered = df.sort_values(["user_id", "event_time"]).copy()
    gap = ordered.groupby("user_id")["event_time"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    ordered["session_id_inferred"] = new_session.groupby(ordered["user_id"]).cumsum()
    return ordered


def session_summary(df: pd.DataFrame, log_name: str) -> dict:
    sessionized = infer_sessions(df)
    sessions = (
        sessionized.groupby(["user_id", "session_id_inferred"])
        .agg(
            session_start=("event_time", "min"),
            session_end=("event_time", "max"),
            events=("video_id", "size"),
            videos_seen=("video_id", "nunique"),
            long_views=("long_view", "sum"),
            likes=("is_like", "sum"),
        )
        .reset_index()
    )
    duration_seconds = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds()
    sessions["duration_seconds"] = duration_seconds

    return {
        "log": log_name,
        "sessions": len(sessions),
        "mean_sessions_per_user": len(sessions) / df["user_id"].nunique(),
        "median_events_per_session": sessions["events"].median(),
        "p90_events_per_session": sessions["events"].quantile(0.90),
        "median_session_duration_seconds": sessions["duration_seconds"].median(),
        "p90_session_duration_seconds": sessions["duration_seconds"].quantile(0.90),
    }


def metadata_summary() -> pd.DataFrame:
    rows = []
    for name, path in METADATA_FILES.items():
        df = pd.read_csv(path, nrows=5)
        row_count = sum(1 for _ in open(path, encoding="utf-8")) - 1
        rows.append(
            {
                "file": name,
                "rows": row_count,
                "columns": len(df.columns),
                "column_names": ", ".join(df.columns),
            }
        )
    return pd.DataFrame(rows)


def early_to_late_return_rate(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    early_users = pd.Index(early["user_id"].unique())
    late_users = pd.Index(late["user_id"].unique())
    returned = early_users.intersection(late_users)
    return pd.DataFrame(
        [
            {
                "eligible_early_users": len(early_users),
                "active_again_late_users": len(returned),
                "observed_return_rate": len(returned) / len(early_users),
                "not_seen_late_users": len(early_users) - len(returned),
            }
        ]
    )


def save_phase1_figures(
    logs: dict[str, pd.DataFrame],
    summaries: pd.DataFrame,
    daily_metrics: pd.DataFrame,
) -> None:
    FIGURES.mkdir(exist_ok=True)

    standard_daily = daily_metrics[daily_metrics["log"].str.startswith("standard")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for log_name, group in standard_daily.groupby("log"):
        ax.plot(group["event_date"], group["active_users"], marker="o", label=log_name)
    ax.set_title("Daily Active Users in Standard Recommendation Logs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Active users")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase1_daily_active_users.png", dpi=150)
    plt.close(fig)

    feedback_rates = summaries.set_index("log")[
        ["is_click_rate", "long_view_rate", "is_like_rate", "is_hate_rate"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    feedback_rates.plot(kind="bar", ax=ax)
    ax.set_title("Feedback Rates by Log Type")
    ax.set_xlabel("Log")
    ax.set_ylabel("Event-level rate")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(FIGURES / "phase1_feedback_rates.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for log_name, df in logs.items():
        user_counts = df.groupby("user_id").size()
        ax.hist(user_counts, bins=50, alpha=0.45, label=log_name, log=True)
    ax.set_title("User Activity Distribution")
    ax.set_xlabel("Events per user")
    ax.set_ylabel("Number of users, log scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase1_user_activity_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    logs = {name: load_log(path) for name, path in LOG_FILES.items()}

    summaries = pd.DataFrame(
        [summarize_log(name, df) for name, df in logs.items()]
    )
    user_distributions = pd.concat(
        [user_activity_distribution(df, name) for name, df in logs.items()],
        ignore_index=True,
    )
    daily_metrics = pd.concat(
        [daily_product_metrics(df, name) for name, df in logs.items()],
        ignore_index=True,
    )
    session_metrics = pd.DataFrame(
        [
            session_summary(logs["standard_early"], "standard_early"),
            session_summary(logs["standard_late"], "standard_late"),
        ]
    )
    metadata = metadata_summary()
    return_rate = early_to_late_return_rate(
        logs["standard_early"], logs["standard_late"]
    )

    summaries.to_csv(REPORTS / "phase1_log_summaries.csv", index=False)
    user_distributions.to_csv(REPORTS / "phase1_user_activity_distribution.csv", index=False)
    daily_metrics.to_csv(REPORTS / "phase1_daily_product_metrics.csv", index=False)
    session_metrics.to_csv(REPORTS / "phase1_session_metrics.csv", index=False)
    metadata.to_csv(REPORTS / "phase1_metadata_summary.csv", index=False)
    return_rate.to_csv(REPORTS / "phase1_observed_return_rate.csv", index=False)
    save_phase1_figures(logs, summaries, daily_metrics)

    print("\n=== Log summaries ===")
    print(summaries.to_string(index=False))

    print("\n=== User activity distribution ===")
    print(user_distributions.to_string(index=False))

    print("\n=== First 10 daily metric rows ===")
    print(daily_metrics.head(10).to_string(index=False))

    print("\n=== Inferred session metrics, standard logs only ===")
    print(session_metrics.to_string(index=False))

    print("\n=== Metadata files ===")
    print(metadata.to_string(index=False))

    print("\n=== Early-to-late observed return rate, standard logs ===")
    print(return_rate.to_string(index=False))

    print("\nSaved report tables to reports/ and figures to reports/figures/.")


if __name__ == "__main__":
    main()
