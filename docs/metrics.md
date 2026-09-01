# Metrics

This document defines product and model metrics used in the project. It is both project documentation and an interview study guide.

## Product Metrics

### Active Users

Definition: Number of distinct users observed during a time period.

Formula:

```text
active_users_t = count_distinct(user_id where event_date in period t)
```

Intuition: Measures how many users showed up in the product during a given day or period.

Why it matters: Retention and disengagement analysis depend on whether users continue to appear in later windows.

Limitations: In this dataset, active means "observed in the logs," not necessarily active in the entire Kuaishou app.

### Event Count Per User

Definition: Number of logged interactions per user in a period.

Intuition: A rough measure of activity frequency or intensity.

Why it matters: Users with more interactions may be more engaged and easier to model.

Limitations: It can be heavily skewed. Means can be dominated by highly active users, so medians and percentiles are needed.

### Click Rate

Definition: Fraction of logged events where `is_click = 1`.

Formula:

```text
click_rate = sum(is_click) / number_of_events
```

Intuition: Measures how often exposed videos generated a click or valid play signal.

Why it matters: Clicks are a basic sign that the feed is matching user interest.

Limitations: Clicks are shaped by what the recommender chose to show. They are not pure preferences.

### Long-View Rate

Definition: Fraction of events where `long_view = 1`.

Formula:

```text
long_view_rate = sum(long_view) / number_of_events
```

Intuition: Measures deeper engagement than a click.

Why it matters: Long views may be a better satisfaction proxy than clicks alone.

Limitations: Long-view definitions depend on video duration rules and platform logging logic.

### Like Rate

Definition: Fraction of events where `is_like = 1`.

Intuition: Measures explicit positive feedback.

Why it matters: Likes are stronger preference signals than clicks, though much rarer.

Limitations: Many satisfied users never like content, so low like rates do not necessarily mean low satisfaction.

### Hate Rate

Definition: Fraction of events where `is_hate = 1`.

Intuition: Measures explicit negative feedback.

Why it matters: Negative feedback can identify mismatch between recommendations and user preferences.

Limitations: The action is rare and may represent only the most motivated negative responses.

### Watch Ratio

Definition: User watch time divided by video duration.

Formula:

```text
watch_ratio = play_time_ms / duration_ms
```

Intuition: A normalized measure of consumption that accounts for video length.

Why it matters: Raw watch time is hard to compare across short and long videos.

Limitations: Ratios can exceed 1 when users replay or watch longer than the nominal duration; rows with zero duration need special handling.

### Observed Return Rate

Definition: Fraction of users active in an earlier window who are also observed in a later window.

Formula:

```text
return_rate = users_active_in_both_windows / users_active_in_early_window
```

Intuition: A first-pass retention proxy.

Why it matters: It helps determine whether the dataset supports churn or disengagement modeling.

Limitations: This is not confirmed retention for the whole product. Users not observed later may be censored, filtered out, or temporarily inactive.

### Cohort Retention Rate

Definition: Fraction of users in a defined cohort who are active in a later period.

Formula:

```text
cohort_retention = users_in_cohort_active_later / users_in_cohort_at_start
```

Intuition: Measures whether different user groups come back at different rates.

Why it matters: Churn and personalization strategies should not assume all users behave the same way. Early activity level, user active degree, or first observed date may identify groups with different retention patterns.

Limitations: Cohort retention is descriptive unless cohort assignment is randomized. Differences across cohorts may reflect selection effects or pre-existing user intent.

### Daily Retention Curve

Definition: For a first-seen cohort, the fraction of users active on each day after first observation.

Formula:

```text
daily_retention_d = users_active_on_day_d_since_first_seen / users_in_first_seen_cohort
```

Intuition: Shows how quickly users stop appearing after their first observed day.

Why it matters: Curves reveal timing patterns that a single return-rate number can hide.

Limitations: Later first-seen cohorts have less follow-up time. This is a censoring problem, so late-cohort curves should be interpreted cautiously.

### Recency

Definition: Time between the end of the observation window and the user's last observed event.

Formula:

```text
recency_days = observation_window_end - user_last_event_time
```

Intuition: Users who have not interacted recently may be at higher risk of future disengagement.

Why it matters: Recency is often one of the strongest behavioral predictors in churn models.

Limitations: A high recency value may reflect normal usage rhythm, logging gaps, or the dataset boundary rather than true loss of interest.

### Frequency

Definition: Count of observed events or active days in the observation window.

Formula:

```text
frequency = count(events for user during observation window)
```

Intuition: Frequent users may be more habitual or more satisfied with the feed.

Why it matters: Phase 2 showed that early activity level is strongly associated with later observed return.

Limitations: Frequency is predictive, not necessarily causal. High frequency may reflect pre-existing user interest.

### Content Diversity

Definition: Number or entropy of distinct content categories, tags, authors, or upload types a user consumed.

Entropy formula:

```text
entropy = -sum(p_i * log2(p_i))
```

Intuition: Higher entropy means the user's attention is spread across more content types. Lower entropy means concentrated preferences.

Why it matters: Diversity can help us ask whether narrow or broad content diets relate to disengagement and personalization needs.

Limitations: Diversity depends on what the recommender exposed, not only what the user prefers.

### Activity Change Ratio

Definition: Ratio of events in the second half of the observation window to events in the first half.

Formula:

```text
activity_change_ratio = (second_half_events + 1) / (first_half_events + 1)
```

Intuition: Captures whether a user's activity is rising or falling before the prediction window.

Why it matters: Declining activity may be an early warning signal for disengagement.

Limitations: The split is simple and may be sensitive to short-term noise, day-of-week effects, or irregular usage.

### Disengagement Label

Definition: Indicator that a user observed in the early standard period was not observed in the late standard period.

Formula:

```text
disengaged_late = 1 if future_events == 0 else 0
```

Intuition: A supervised-learning target for future churn/disengagement modeling.

Why it matters: Phase 4 will train models to predict this label.

Limitations: This is not literal churn. It is observed non-return in the dataset's late standard log.

## Model Metrics

### ROC-AUC

Definition: Area under the receiver operating characteristic curve.

Intuition: Measures how well the model ranks a randomly chosen disengaged user above a randomly chosen retained user.

Why it matters: It evaluates risk ranking across all possible thresholds.

Limitations: ROC-AUC can look strong even when the rare positive class is hard to identify precisely.

### PR-AUC

Definition: Area under the precision-recall curve. Also called average precision in scikit-learn.

Intuition: Measures the tradeoff between finding disengaged users and avoiding false alarms.

Why it matters: Disengagement is rare, so PR-AUC is more informative than accuracy and often more revealing than ROC-AUC.

Limitations: PR-AUC depends on class prevalence, so values should be compared against the baseline positive rate.

### Precision

Definition: Of users flagged as at risk, the fraction who truly disengaged.

Formula:

```text
precision = true_positives / (true_positives + false_positives)
```

Why it matters: If interventions are costly or annoying, high precision matters.

Limitations: High precision can come with low recall if the model flags very few users.

### Recall

Definition: Of truly disengaged users, the fraction the model successfully flagged.

Formula:

```text
recall = true_positives / (true_positives + false_negatives)
```

Why it matters: If missing at-risk users is expensive, high recall matters.

Limitations: High recall can create many false positives.

### F1 Score

Definition: Harmonic mean of precision and recall.

Formula:

```text
F1 = 2 * precision * recall / (precision + recall)
```

Why it matters: Useful when precision and recall are both important.

Limitations: F1 hides the business tradeoff between false positives and false negatives.

### Brier Score

Definition: Mean squared error between predicted probabilities and actual binary outcomes.

Formula:

```text
brier = mean((predicted_probability - actual_outcome)^2)
```

Why it matters: Measures probability calibration, not only ranking.

Limitations: A low Brier score can partly reflect class imbalance, so calibration curves are also useful.

### Top-Share Intervention Metrics

Definition: Precision and recall when flagging only the top X% highest-risk users.

Intuition: Mimics a product team with limited intervention capacity.

Why it matters: A churn model often feeds an action list, not a universal yes/no decision.

Limitations: The best top-share threshold depends on intervention cost, user annoyance risk, and expected benefit.

### Permutation Importance

Definition: Change in model performance after randomly shuffling one feature while keeping all other features unchanged.

Formula:

```text
permutation_importance_j = original_score - score_after_shuffling_feature_j
```

Intuition: If shuffling a feature hurts PR-AUC a lot, the model relied on that feature for ranking disengaged users.

Why it matters: It gives a more direct model-interpretation signal than built-in tree impurity importance.

Limitations: Correlated features can split importance. If two features contain similar information, shuffling one may not hurt much because the model can still use the other.

### Odds Ratio

Definition: Exponentiated logistic regression coefficient.

Formula:

```text
odds_ratio = exp(coefficient)
```

Intuition: Values above 1 are associated with higher odds of disengagement; values below 1 are associated with lower odds, holding other model features constant.

Why it matters: Odds ratios help explain logistic regression in business language.

Limitations: With standardized numeric features, the odds ratio corresponds to a one-standard-deviation change. It is not causal and can be distorted by correlated predictors.

### Precision@K

Definition: Fraction of the top K recommended items that are relevant.

Formula:

```text
precision@K = relevant_recommended_items_in_top_K / K
```

Intuition: Of the videos we recommended, how many did the user later engage with?

Why it matters: It measures the quality of limited recommendation slots.

Limitations: It penalizes users with fewer than K future relevant items and depends on observed exposure.

### Recall@K

Definition: Fraction of a user's relevant future items that appear in the top K recommendations.

Formula:

```text
recall@K = relevant_recommended_items_in_top_K / all_relevant_future_items
```

Intuition: Of the videos the user later engaged with, how many did we recover?

Why it matters: It measures coverage of future user interest.

Limitations: A user may have many relevant future items, and the dataset only observes items the production system exposed.

### Hit Rate@K

Definition: Fraction of users with at least one relevant item in the top K recommendations.

Formula:

```text
hit_rate@K = users_with_at_least_one_hit / evaluated_users
```

Intuition: Did the recommendation list contain anything useful for the user?

Why it matters: It is easy to explain to product partners.

Limitations: It does not distinguish between one hit and many hits.

### NDCG@K

Definition: Normalized discounted cumulative gain at K.

Formula:

```text
DCG@K = sum(relevance_at_rank_i / log2(i + 1))
NDCG@K = DCG@K / ideal_DCG@K
```

Intuition: Rewards relevant recommendations more when they appear near the top.

Why it matters: Feed ranking position matters; users are more likely to see and act on top-ranked items.

Limitations: Offline relevance comes from observed future interactions, not the complete set of videos the user would have liked.

### Experiment Primary Metric: 7-Day Observed Return Rate

Definition: Fraction of randomized users who are observed active within 7 days after assignment.

Formula:

```text
return_rate = users_observed_active_within_7_days / randomized_users
```

Intuition: Measures whether users come back after receiving the control or treatment experience.

Why it matters: It directly matches the retention goal.

Limitations: It captures observed return, not long-term satisfaction or permanent retention.

### Minimum Detectable Effect

Definition: The smallest treatment-control difference the experiment is designed to detect with chosen power and significance level.

Intuition: Smaller effects require larger samples.

Why it matters: A product team should decide what improvement is practically meaningful before running the test.

Limitations: MDE is a planning assumption, not an observed result.

### Statistical Power

Definition: Probability of detecting an effect if the true effect is at least the chosen MDE.

Common target:

```text
power = 80%
```

Why it matters: Low-powered experiments can miss meaningful effects.

Limitations: Higher power requires more users or longer runtime.

### P-Value

Definition: Under the null hypothesis, the probability of observing a result at least as extreme as the one measured.

Why it matters: Helps decide whether an observed difference is statistically surprising under no effect.

Limitations: A p-value is not the probability the treatment works, and it does not measure business importance.

### Confidence Interval

Definition: A range of plausible values for the treatment effect under repeated sampling.

Why it matters: Shows uncertainty around the estimated lift.

Limitations: A narrow statistically significant interval can still describe a practically small effect.
