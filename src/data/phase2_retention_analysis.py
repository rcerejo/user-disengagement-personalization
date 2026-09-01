from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

STANDARD_LOG_FILES = {
    "standard_early": RAW_DATA / "log_standard_4_08_to_4_21_pure.csv",
    "standard_late": RAW_DATA / "log_standard_4_22_to_5_08_pure.csv",
}


def load_standard_logs() -> pd.DataFrame:
    frames = []
    for log_name, path in STANDARD_LOG_FILES.items():
        df = pd.read_csv(path)
        df["log"] = log_name
        df["event_time"] = pd.to_datetime(df["time_ms"], unit="ms")
        df["event_date"] = pd.to_datetime(df["event_time"].dt.date)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def assign_activity_cohort(event_counts: pd.Series) -> pd.Series:
    labels = ["low", "medium", "high", "very_high"]
    ranked = event_counts.rank(method="first")
    return pd.qcut(ranked, q=4, labels=labels)


def build_user_cohorts(events: pd.DataFrame) -> pd.DataFrame:
    early = events[events["log"] == "standard_early"]
    late = events[events["log"] == "standard_late"]
    user_features = pd.read_csv(RAW_DATA / "user_features_pure.csv")

    early_user = (
        early.groupby("user_id")
        .agg(
            first_seen_date=("event_date", "min"),
            early_events=("video_id", "size"),
            early_active_days=("event_date", "nunique"),
            early_videos_seen=("video_id", "nunique"),
            early_click_rate=("is_click", "mean"),
            early_long_view_rate=("long_view", "mean"),
            early_like_rate=("is_like", "mean"),
            early_hate_rate=("is_hate", "mean"),
        )
        .reset_index()
    )
    early_user["activity_cohort"] = assign_activity_cohort(early_user["early_events"])

    late_user = (
        late.groupby("user_id")
        .agg(
            late_events=("video_id", "size"),
            late_active_days=("event_date", "nunique"),
            late_first_seen=("event_date", "min"),
        )
        .reset_index()
    )

    cohorts = early_user.merge(late_user, on="user_id", how="left")
    cohorts = cohorts.merge(user_features, on="user_id", how="left")
    cohorts["returned_late"] = cohorts["late_events"].notna()
    cohorts["late_events"] = cohorts["late_events"].fillna(0).astype(int)
    cohorts["late_active_days"] = cohorts["late_active_days"].fillna(0).astype(int)
    cohorts["days_until_late_return"] = (
        cohorts["late_first_seen"] - cohorts["first_seen_date"]
    ).dt.days
    return cohorts


def retention_by_activity(cohorts: pd.DataFrame) -> pd.DataFrame:
    return (
        cohorts.groupby("activity_cohort", observed=True)
        .agg(
            users=("user_id", "nunique"),
            observed_return_rate=("returned_late", "mean"),
            median_early_events=("early_events", "median"),
            median_late_events=("late_events", "median"),
            median_early_active_days=("early_active_days", "median"),
            median_late_active_days=("late_active_days", "median"),
        )
        .reset_index()
    )


def retention_by_user_active_degree(cohorts: pd.DataFrame) -> pd.DataFrame:
    return (
        cohorts.groupby("user_active_degree", dropna=False)
        .agg(
            users=("user_id", "nunique"),
            observed_return_rate=("returned_late", "mean"),
            median_early_events=("early_events", "median"),
            median_late_events=("late_events", "median"),
        )
        .reset_index()
        .sort_values("users", ascending=False)
    )


def retention_by_first_seen_date(cohorts: pd.DataFrame) -> pd.DataFrame:
    return (
        cohorts.groupby("first_seen_date")
        .agg(
            users=("user_id", "nunique"),
            observed_return_rate=("returned_late", "mean"),
            median_early_events=("early_events", "median"),
        )
        .reset_index()
    )


def daily_retention_curve(events: pd.DataFrame) -> pd.DataFrame:
    first_seen = events.groupby("user_id")["event_date"].min().rename("first_seen_date")
    user_days = events[["user_id", "event_date"]].drop_duplicates().merge(
        first_seen, on="user_id"
    )
    user_days["day_since_first_seen"] = (
        user_days["event_date"] - user_days["first_seen_date"]
    ).dt.days

    cohort_sizes = first_seen.groupby(first_seen).size().rename("cohort_users")
    active_by_day = (
        user_days.groupby(["first_seen_date", "day_since_first_seen"])
        .agg(active_users=("user_id", "nunique"))
        .reset_index()
        .merge(cohort_sizes, on="first_seen_date")
    )
    active_by_day["retention_rate"] = (
        active_by_day["active_users"] / active_by_day["cohort_users"]
    )
    return active_by_day


def save_figures(
    by_activity: pd.DataFrame,
    by_active_degree: pd.DataFrame,
    by_first_seen: pd.DataFrame,
    daily_curve: pd.DataFrame,
) -> None:
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_activity["activity_cohort"].astype(str), by_activity["observed_return_rate"])
    ax.set_title("Observed Late-Period Return Rate by Early Activity Cohort")
    ax.set_xlabel("Early activity cohort")
    ax.set_ylabel("Observed return rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase2_return_rate_by_activity_cohort.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    top_degrees = by_active_degree.sort_values("users", ascending=False).head(6)
    ax.bar(top_degrees["user_active_degree"].astype(str), top_degrees["observed_return_rate"])
    ax.set_title("Observed Late-Period Return Rate by User Active Degree")
    ax.set_xlabel("User active degree")
    ax.set_ylabel("Observed return rate")
    ax.set_ylim(0, 1)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase2_return_rate_by_user_active_degree.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        by_first_seen["first_seen_date"],
        by_first_seen["observed_return_rate"],
        marker="o",
    )
    ax.set_title("Observed Late-Period Return Rate by First Seen Date")
    ax.set_xlabel("First seen date")
    ax.set_ylabel("Observed return rate")
    ax.set_ylim(0, 1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase2_return_rate_by_first_seen_date.png", dpi=150)
    plt.close(fig)

    large_cohorts = daily_curve[daily_curve["cohort_users"] >= 100]
    fig, ax = plt.subplots(figsize=(10, 5))
    for first_seen_date, group in large_cohorts.groupby("first_seen_date"):
        if group["day_since_first_seen"].max() >= 7:
            ax.plot(
                group["day_since_first_seen"],
                group["retention_rate"],
                alpha=0.45,
                label=str(first_seen_date.date()),
            )
    ax.set_title("Daily Retention Curves by First Seen Date")
    ax.set_xlabel("Days since first observed")
    ax.set_ylabel("Retention rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase2_daily_retention_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    events = load_standard_logs()
    cohorts = build_user_cohorts(events)

    by_activity = retention_by_activity(cohorts)
    by_active_degree = retention_by_user_active_degree(cohorts)
    by_first_seen = retention_by_first_seen_date(cohorts)
    daily_curve = daily_retention_curve(events)

    cohorts.to_csv(REPORTS / "phase2_user_retention_cohorts.csv", index=False)
    by_activity.to_csv(REPORTS / "phase2_retention_by_activity_cohort.csv", index=False)
    by_active_degree.to_csv(REPORTS / "phase2_retention_by_user_active_degree.csv", index=False)
    by_first_seen.to_csv(REPORTS / "phase2_retention_by_first_seen_date.csv", index=False)
    daily_curve.to_csv(REPORTS / "phase2_daily_retention_curve.csv", index=False)

    save_figures(by_activity, by_active_degree, by_first_seen, daily_curve)

    print("\n=== Retention by early activity cohort ===")
    print(by_activity.to_string(index=False))

    print("\n=== Retention by user active degree ===")
    print(by_active_degree.to_string(index=False))

    print("\n=== Retention by first seen date ===")
    print(by_first_seen.to_string(index=False))

    print("\nSaved Phase 2 tables to reports/ and figures to reports/figures/.")


if __name__ == "__main__":
    main()
