# Modeling Decisions

This document records analytical choices as the project evolves. The goal is to preserve why decisions were made, not only what code was written.

## Project Decision Log

### 2026-08-30 / Phase 0

Decision: Use KuaiRand-Pure as the project dataset.

Why: It contains repeated user-video interactions, timestamps, rich feedback signals, user features, video features, standard recommendation logs, and random exposure logs. This supports product analytics, disengagement modeling, recommendation systems, experiment design, and careful discussion of exposure bias.

Alternatives considered: RetailRocket, MIND, KuaiRec, MovieLens, YooChoose, and Tenrec.

Risks / limitations: KuaiRand-Pure is not complete user lifetime data. It is also a filtered version of the larger KuaiRand dataset, so disengagement must be framed as observed inactivity or decline within the available logs.

### 2026-08-30 / Phase 1

Decision: Keep standard recommendation logs and random exposure logs separate during initial exploration.

Why: Standard logs reflect the production recommender's choices, while random logs reflect randomized video exposure. Their engagement rates are not directly interchangeable.

Alternatives considered: Pooling all logs into one event table for simpler summary metrics.

Risks / limitations: Keeping logs separate adds complexity, but pooling them too early would blur product behavior with intervention behavior.

### 2026-08-30 / Phase 1

Decision: Treat churn as a future behavioral disengagement proxy, not literal app churn.

Why: The dataset records observed interactions during a limited time window. It does not show account deletion, app uninstall, or complete user lifetime.

Alternatives considered: Calling users who disappear from the late standard log "churned."

Risks / limitations: Behavioral non-return may reflect censoring, sampling, logging coverage, or normal usage gaps rather than permanent churn.

### 2026-08-30 / Phase 1

Decision: Use a 30-minute inactivity gap only as an exploratory sessionization heuristic.

Why: The raw logs do not include explicit session IDs. A gap-based rule is common for behavioral data, but should be validated before it becomes central to the analysis.

Alternatives considered: Avoiding sessions entirely in Phase 1 or using a different inactivity threshold.

Risks / limitations: Many inferred sessions have one event and zero duration, so session metrics may be noisy and should not drive major conclusions yet.

### 2026-08-30 / Phase 2

Decision: Define the main Phase 2 retention denominator as users observed in the early standard recommendation log.

Why: The early standard period gives us pre-outcome behavior for cohort assignment. The late standard period can then be used as a future observation window.

Alternatives considered: Using all users across standard and random logs, or using first observed date across the entire dataset.

Risks / limitations: This estimates observed return among users already active in the standard logs. It is not full-platform retention and may overstate return rates.

### 2026-08-30 / Phase 2

Decision: Use early event-count quartiles for initial activity cohorts.

Why: Quartiles create simple, interpretable groups with similar numbers of users and avoid choosing arbitrary event-count thresholds before understanding the distribution.

Alternatives considered: Fixed thresholds, active-day cohorts, watch-time cohorts, or model-based clustering.

Risks / limitations: Quartiles are relative to this dataset. A "low" activity user here may not be low activity in the broader Kuaishou population.

### 2026-08-30 / Phase 2

Decision: Do not use Kaplan-Meier survival analysis as the main Phase 2 method.

Why: Survival analysis is useful when exact time-to-event and right-censoring structure are central to the problem. Here, the logs are short, the return rate is very high, and "churn" is only an observed non-return proxy. Cohort tables and retention curves are more transparent at this stage.

Alternatives considered: Kaplan-Meier curves for time until last observed activity or time until non-return.

Risks / limitations: We still need to discuss censoring carefully. A later phase may revisit survival analysis if we define a better event process.

### 2026-08-30 / Phase 3

Decision: Build the modeling table at the user level, with one row per user observed in the early standard recommendation period.

Why: The future task is to predict user disengagement, so the prediction unit should be a user at a defined point in time, not an individual event.

Alternatives considered: Event-level modeling or session-level modeling.

Risks / limitations: User-level aggregation loses sequence detail. Later models may benefit from sequence-aware features, but starting at the user level is more interpretable.

### 2026-08-30 / Phase 3

Decision: Use early standard-period behavior as the observation window and late standard-period non-return as the disengagement label.

Why: This creates a clear temporal order: features are measured before the outcome.

Alternatives considered: Combining standard and random logs, or using all dates to create rolling labels.

Risks / limitations: The label is imbalanced and represents observed non-return, not confirmed churn. The current disengagement rate is about 4.6%.

### 2026-08-30 / Phase 3

Decision: Exclude `video_features_statistic_pure.csv` from Phase 3 model features.

Why: Those aggregate video statistics may include future behavior or global post-period information. Using them could leak information into churn and recommendation models.

Alternatives considered: Joining all video features immediately for richer predictors.

Risks / limitations: Excluding aggregate video statistics may leave out useful item-quality signals. We can revisit them only if we can make their timing defensible.

### 2026-08-30 / Phase 3

Decision: Create interpretable behavioral feature families before more complex modeling.

Why: Features such as recency, frequency, engagement rates, content diversity, repeat behavior, and activity change can be explained clearly to product partners and interviewers.

Alternatives considered: More complex sequence embeddings or clustering.

Risks / limitations: Hand-engineered features may miss temporal patterns that sequence models could capture, but they are the right baseline for this project stage.

### 2026-08-30 / Phase 4

Decision: Use `disengaged_late = True` as the positive class.

Why: The product action is to identify users at risk of disengagement, so evaluation should focus on the rare at-risk class.

Alternatives considered: Predicting return instead of disengagement.

Risks / limitations: The positive class is only about 4.6%, so threshold metrics are sensitive and accuracy is not useful as the primary metric.

### 2026-08-30 / Phase 4

Decision: Compare models against an all-users-return baseline.

Why: With an imbalanced label, a naive model can look good on accuracy while finding zero at-risk users. The baseline makes that failure visible.

Alternatives considered: Majority-class accuracy only or a random-score baseline.

Risks / limitations: This baseline is intentionally simple. Later comparisons may include heuristic baselines such as recency-only ranking.

### 2026-08-30 / Phase 4

Decision: Use stratified user-level train/test splitting for initial model comparison.

Why: The Phase 3 table already enforces temporal order between features and label. Stratification preserves the rare disengagement rate in train and test sets.

Alternatives considered: Splitting event rows or using first-seen-date temporal splits.

Risks / limitations: This is not a full multi-period temporal validation. We only have one clean future outcome window, so generalization to future calendar periods remains uncertain.

### 2026-08-30 / Phase 4

Decision: Prioritize ROC-AUC, PR-AUC, recall, precision, calibration, and top-share threshold metrics rather than accuracy.

Why: The positive class is rare and the product use case is risk ranking plus targeted intervention.

Alternatives considered: Accuracy or default-threshold F1 as the main metric.

Risks / limitations: Offline classification metrics do not prove a retention intervention will work. Threshold choice requires product cost assumptions.

### 2026-08-30 / Phase 5

Decision: Use logistic regression coefficients and random forest permutation importance as the main interpretation tools.

Why: Logistic coefficients give an interpretable linear baseline, while permutation importance asks how much random forest PR-AUC depends on each feature.

Alternatives considered: SHAP values or only built-in random forest impurity importance.

Risks / limitations: SHAP was not installed and would add complexity. Permutation importance is still model-specific and can be unstable with correlated features.

### 2026-08-30 / Phase 5

Decision: Frame model interpretation as product hypotheses rather than product recommendations.

Why: The models are predictive, not causal. They can suggest where disengagement risk is concentrated, but they cannot prove an intervention would improve retention.

Alternatives considered: Directly recommending interventions based on feature importance.

Risks / limitations: Product hypotheses still require experiment design and business judgment before launch.

### 2026-08-30 / Phase 5

Decision: Explicitly call out counterintuitive logistic coefficients as an interpretation issue.

Why: Some user-active-degree coefficients become harder to interpret after controlling for direct activity features. This is a useful example of multicollinearity and conditional interpretation.

Alternatives considered: Hiding confusing coefficients or dropping categorical user-active-degree fields.

Risks / limitations: Keeping these coefficients requires careful explanation so readers do not overinterpret them causally.

### 2026-08-31 / Phase 6

Decision: Define implicit positive recommendation feedback as click, long view, like, follow, comment, or forward.

Why: KuaiRand-Pure does not provide explicit ratings. These actions are reasonable positive behavioral signals, with long views and social actions indicating deeper engagement.

Alternatives considered: Using clicks only, long views only, or weighted feedback.

Risks / limitations: Clicks and long views are not identical to satisfaction. A future iteration could test weighted implicit feedback.

### 2026-08-31 / Phase 6

Decision: Train recommenders on the early standard log and evaluate on late standard-log positives.

Why: This creates a temporal offline evaluation where recommendations are trained on past behavior and evaluated against future observed engagement.

Alternatives considered: Random train/test splitting of interactions.

Risks / limitations: Offline relevance is still shaped by what the production recommender exposed later. Unobserved items are not necessarily irrelevant.

### 2026-08-31 / Phase 6

Decision: Compare SVD matrix factorization against a popularity baseline.

Why: Popularity is a strong simple recommender. Matrix factorization must beat it to show personalization value.

Alternatives considered: Item-item collaborative filtering, neural recommenders, or ranking models.

Risks / limitations: SVD captures broad collaborative patterns but may struggle with cold-start users/items and does not model sequence or context.

### 2026-08-31 / Phase 7

Decision: Evaluate recommendation quality only for users with future positive interactions on videos seen in the training item universe.

Why: Ranking metrics require known relevant future items. Users who disappear in the late period have no observed relevant items, so Precision@K, Recall@K, Hit Rate@K, and NDCG@K are undefined for them.

Alternatives considered: Assigning zero recommendation quality to disengaged users.

Risks / limitations: Assigning zeros would mix non-return with recommendation relevance and could falsely imply poor recommendations caused disengagement.

### 2026-08-31 / Phase 7

Decision: Compare SVD lift over popularity across activity and activity-change segments.

Why: Phase 6 showed overall personalization gains. Segment analysis asks whether those gains are evenly distributed or concentrated among users with richer histories.

Alternatives considered: Reporting only aggregate recommendation metrics.

Risks / limitations: Segment-level recommendation lift is descriptive and conditional on users returning with future positives.

### 2026-08-31 / Phase 7

Decision: Treat personalization-disengagement findings as hypotheses for an experiment.

Why: Offline recommendation quality and churn risk are not causal evidence. A live A/B test is needed to measure whether personalization changes retention.

Alternatives considered: Recommending a rollout based on offline SVD gains.

Risks / limitations: Offline gains may not translate into user satisfaction or retention impact.

### 2026-09-01 / Phase 8

Decision: Design, but do not analyze results from, an A/B test for personalization.

Why: KuaiRand-Pure does not contain a randomized experiment for our proposed personalization strategy. We can design a valid test, but we cannot fabricate treatment effects.

Alternatives considered: Simulating artificial experiment results from the observational dataset.

Risks / limitations: The design is hypothetical until implemented in production.

### 2026-09-01 / Phase 8

Decision: Use user-level randomization.

Why: The primary outcome is user retention. Randomizing individual events could expose the same user to both control and treatment, contaminating the comparison.

Alternatives considered: Event-level or session-level randomization.

Risks / limitations: User-level randomization requires enough users and careful handling of social/network spillovers.

### 2026-09-01 / Phase 8

Decision: Use 7-day observed return rate as the primary experiment metric.

Why: It aligns directly with the retention objective and is easy for product stakeholders to understand.

Alternatives considered: Click rate, long-view rate, watch time, or recommendation NDCG as primary metrics.

Risks / limitations: Return rate is still an observed behavioral proxy. Engagement-quality metrics should be monitored as secondary or guardrail metrics.

### 2026-09-01 / Phase 8

Decision: Use a two-sided two-proportion z-test sample-size approximation.

Why: The primary metric is binary at the user level: returned or did not return.

Alternatives considered: One-sided test, sequential testing, bootstrap simulation, or Bayesian experiment design.

Risks / limitations: The formula assumes independent users, stable assignment, and a fixed analysis plan.

### 2026-09-01 / Phase 9

Decision: Do not estimate a causal retention effect from KuaiRand-Pure.

Why: Users were not randomized to our proposed personalization strategy versus a control group. The dataset supports predictive modeling and offline recommendation evaluation, but not the causal effect of the proposed intervention on retention.

Alternatives considered: Propensity score adjustment, regression adjustment, or treating random exposure as the treatment.

Risks / limitations: Declining to run causal inference may seem less sophisticated, but it is statistically more defensible than forcing a weak causal design.

### 2026-09-01 / Phase 9

Decision: Use random exposure descriptively to discuss exposure bias, not to claim retention impact.

Why: Random exposure helps show how engagement differs when videos are not fully selected by the production recommender. However, it is not a user-level randomized test of our personalization strategy.

Alternatives considered: Comparing standard and random logs as if they were treatment and control groups.

Risks / limitations: Standard and random exposures may differ in context, placement, timing, and policy rules.

### 2026-09-01 / Phase 10

Decision: Recommend an A/B test of segment-aware personalization rather than a direct rollout.

Why: The evidence supports predictive risk ranking and offline recommendation gains, but not causal retention impact.

Alternatives considered: Recommending full rollout of SVD personalization or only recommending more analysis.

Risks / limitations: The proposed intervention still needs product design, engineering implementation, and online measurement.

### 2026-09-01 / Phase 11

Decision: Package the project as a case study with explicit evidence labels.

Why: Recruiters need a quick business summary, while interviewers need to see technical reasoning, assumptions, limitations, and reproducibility.

Alternatives considered: Waiting until the end to write a README only.

Risks / limitations: The documentation must stay consistent with the actual analysis and avoid inflated impact claims.
