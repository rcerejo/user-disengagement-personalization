# Methodology

This document explains the statistical and machine learning methods used in the project in context.

## Phase 1: Data Understanding And Product Metrics

### Behavioral Event Data

What it does: Treats each row as a logged interaction between a user and a video at a point in time.

Why we used it: The raw data is event-level, so product metrics must start from events before aggregating to users, videos, days, or sessions.

Assumptions: Logged events represent real opportunities for engagement, and the core identifiers and timestamps are reliable.

Interpretation: Event-level metrics answer questions like "what fraction of impressions were clicked?" User-level metrics answer different questions, such as "how active was a typical user?"

Common mistakes: Treating event rows as independent users, averaging across all events without checking user-level skew, or mixing exposure policies without understanding them.

### Distributional Summaries

What they do: Use medians, percentiles, minimums, maximums, and histograms to understand skewed behavior.

Why we used them: Consumer product usage is rarely normally distributed. A small number of highly active users can dominate averages.

Interpretation: The median describes a typical user better than the mean when the distribution is skewed. Upper percentiles show power-user behavior.

Common mistakes: Reporting only the mean, ignoring outliers, or removing high-activity users without a product reason.

### Exposure Bias

What it does: Describes the fact that users can only engage with videos they were shown.

Why it matters here: Standard recommendation logs are shaped by the production recommender. Observed engagement is a combination of user preference, item appeal, context, and algorithmic exposure.

Interpretation: A clicked video is not simply a preferred video; it is a preferred video among those exposed.

Common mistakes: Treating unshown videos as negative examples, or assuming standard-log clicks reveal unbiased preferences.

### Exploratory Sessionization

What it does: Groups user events into inferred sessions when gaps between events are less than or equal to 30 minutes.

Why we used it: The raw data has no explicit session ID, but sessions can help describe browsing intensity if the heuristic behaves reasonably.

Interpretation: In Phase 1, session metrics are exploratory only because many inferred sessions contain one event.

Common mistakes: Treating heuristic sessions as ground truth, choosing a gap threshold without sensitivity analysis, or overinterpreting zero-duration sessions.

## Phase 2: Cohort And Retention Analysis

### Cohort Analysis

What it does: Groups users by characteristics known before the outcome window, then compares later behavior across those groups.

Why we used it: Before building a churn model, we need to know whether retention differs across intuitive user groups such as early activity level and user active degree.

Assumptions: Cohort membership is measured before the retention outcome. Users in the early standard log are eligible to be observed again in the late standard log.

Implementation: Users were grouped into early event-count quartiles using only standard early-period behavior. We then measured whether each user appeared again in the standard late-period log.

Interpretation: The lowest early-activity cohort had an observed late-period return rate of 85.8%, while the highest early-activity cohort had a return rate of 99.8%. This is strong descriptive evidence that early activity and later observed return are associated.

Common mistakes: Defining cohorts using future behavior, treating cohort differences as causal, or forgetting that retention denominators determine the meaning of the metric.

### Retention Curves

What they do: Show the fraction of a cohort that is active on each day after first observation.

Why we used them: A single return-rate number hides timing. Curves help reveal whether users disappear quickly, gradually, or irregularly.

Assumptions: First observed date is treated as the cohort start, but it is not true signup date.

Implementation: For each user's first observed standard-log date, we counted whether the user was active on each later day and divided by the first-seen cohort size.

Interpretation: Curves should be compared most carefully for cohorts with enough follow-up time and enough users.

Common mistakes: Comparing late first-seen cohorts to early first-seen cohorts without accounting for different available follow-up windows.

### Censoring

What it does: Describes incomplete observation. If a user first appears near the end of the dataset, we cannot know whether they would return after the dataset ends.

Why it matters here: Users first observed later have less opportunity to be counted as retained on future days.

How we handled it: We reported first-seen cohorts and daily retention curves, but avoided using late-cohort curves as definitive churn evidence.

Common mistakes: Treating all non-observed future activity as churn, even when the dataset ends too soon to observe future behavior.

### Why We Did Not Lead With Survival Analysis

Survival analysis can be valuable when the event time is meaningful and censoring is central. In this phase, the dataset has a short window, very high observed return among standard-log users, and no true churn event. A simpler cohort analysis is easier to explain and better matched to the evidence.

## Phase 3: Behavioral Feature Engineering

### Observation And Prediction Windows

What it does: Separates the time used to build features from the future time used to define the label.

Why we used it: Predictive modeling should mimic the real product question: given what we know now, can we identify users at risk later?

Implementation: Early standard-log behavior was used for features. Late standard-log activity was used to define `disengaged_late`.

Common mistakes: Building features from the same period as the label or accidentally using future activity to summarize past behavior.

### User-Level Aggregation

What it does: Converts event-level logs into one row per user.

Why we used it: The Phase 4 prediction task is user disengagement, so each training example should be a user with known prior behavior.

Implementation: We aggregated counts, rates, unique videos, unique authors, unique tags, watch-time summaries, recency, repeat rates, and activity-change measures.

Common mistakes: Treating every event as an independent example for a user-level churn problem, which overweights active users and violates the intended prediction unit.

### Feature Leakage

What it does: Describes when predictors contain information that would not be available at prediction time.

Why it matters here: Leakage can make churn models look strong offline but fail in production.

Implementation: We excluded late-period behavior from features and avoided the aggregate video statistics file for now because its timing is not yet defensible.

Common mistakes: Joining future item popularity, using post-outcome engagement rates, or creating cohorts from the outcome period before modeling.

### Behavioral Feature Families

What they do: Represent product hypotheses about disengagement risk.

Why we used them: Interpretable features help connect model results to product decisions.

Examples:

- Recency: users who disappeared late in the observation window may be at risk.
- Frequency: users with more active days may be more habitual.
- Engagement quality: long-view and like rates may reflect satisfaction.
- Negative feedback: hate rate may signal poor recommendation fit.
- Diversity: tag entropy may show whether a user has broad or narrow interests.
- Activity change: a drop in events may indicate weakening engagement.

Common mistakes: Assuming a predictive feature is a causal lever. For example, if low frequency predicts churn, it does not mean forcing more sessions would cause retention.

## Phase 4: Churn And Disengagement Modeling

### Prediction Problem

What it does: Uses early-period user behavior to estimate the probability that a user will not appear in the late standard period.

Why we used it: The product goal is to identify at-risk users before they disengage.

Implementation: The target is `disengaged_late`, where 1 means the user had zero events in the late standard period.

Common mistakes: Treating this as literal churn, or using features measured after the prediction point.

### Baseline Model

What it does: Predicts that no users disengage.

Why we used it: Because the label is imbalanced, this baseline would look strong on accuracy but has zero recall for at-risk users.

Interpretation: Any useful churn model must beat this baseline on rare-class detection and ranking metrics.

### Logistic Regression

What it does: Models the log odds of disengagement as a linear function of features.

Mathematical intuition:

```text
p(disengaged) = 1 / (1 + exp(-(b0 + b1*x1 + ... + bk*xk)))
```

Why we used it: It is an interpretable baseline and gives coefficients that can be translated into odds ratios.

Implementation: Numeric features were imputed and scaled; categorical features were one-hot encoded. Class weighting helped account for the rare positive class.

Common mistakes: Interpreting coefficients causally, ignoring multicollinearity, or forgetting that scaling changes the coefficient unit.

### Tree Ensembles

What they do: Learn nonlinear splits and interactions among features.

Why we used them: Disengagement risk may depend on combinations of behavior, such as low frequency plus high recency plus declining activity.

Implementation: We trained random forest and histogram gradient boosting models as nonlinear comparisons against logistic regression.

Common mistakes: Treating feature importance as causal, overfitting, or choosing a complex model without checking whether it improves evaluation.

### Threshold Selection

What it does: Converts predicted probabilities into a product action rule.

Why it matters: A model can rank users well, but the product team still needs to decide how many users to target.

Implementation: We evaluated default 0.5 threshold metrics and top-share thresholds for the riskiest 1%, 5%, 10%, and 20% of users.

Interpretation: At the riskiest 10% of users, random forest caught about 50.7% of disengaged users with 23.2% precision.

Common mistakes: Using 0.5 automatically, or choosing a threshold without considering intervention cost.

## Phase 5: Model Interpretation And Product Insights

### Logistic Regression Coefficients

What they do: Describe how each feature is associated with the log odds of disengagement, holding other model features fixed.

Why we used them: Logistic regression is easier to explain than tree ensembles and helps connect model behavior to interpretable feature directions.

Interpretation: Positive coefficients indicate higher predicted disengagement risk; negative coefficients indicate lower predicted risk. Odds ratios translate coefficients from log-odds units into multiplicative odds.

Common mistakes: Treating coefficients as causal, forgetting that numeric features were standardized, or ignoring correlation among predictors.

### Permutation Importance

What it does: Measures how much model performance drops when a feature is randomly shuffled.

Why we used it: It evaluates whether the fitted random forest relies on a feature for predicting disengagement risk.

Implementation: We measured average PR-AUC drop after shuffling each feature on the test set.

Interpretation: Features such as second-half event count, unique authors, total events, unique videos, events per active day, recency, and tag diversity were important to the random forest's risk ranking.

Common mistakes: Assuming importance means causality, or assuming low importance means a feature is useless when correlated features exist.

### Product Hypotheses

What they do: Convert predictive signals into ideas that a product team could test.

Why we used them: Data Science work should connect model findings to decisions, but the strength of the recommendation should match the evidence.

Examples:

- Declining activity may indicate users whose feed is losing relevance.
- High recency may indicate users already drifting away.
- Content diversity may help segment users who need novelty versus familiar content.
- Weak deep engagement may suggest ranking should consider long-view quality, not only clicks.

Common mistakes: Saying the intervention will work without running an experiment, or recommending one global intervention for all at-risk users.

## Phase 6: Recommendation And Personalization

### Implicit Feedback

What it does: Infers user preference from behavioral signals such as clicks, long views, likes, follows, comments, and forwards.

Why we used it: The dataset does not contain explicit ratings. Short-video products usually rely on behavioral feedback.

Implementation: We treated a user-video pair as positive if the user had at least one positive interaction signal in the standard logs.

Common mistakes: Treating missing interactions as dislikes, or assuming every click means satisfaction.

### Popularity Baseline

What it does: Recommends the same most popular videos to all users, excluding videos the user already engaged with in training.

Why we used it: Popularity is simple, strong, and easy to explain. A personalized recommender must beat it.

Interpretation: If personalization only barely beats popularity, the product value may be limited or concentrated in certain segments.

### Matrix Factorization

What it does: Represents users and videos as lower-dimensional vectors learned from the sparse user-item interaction matrix.

Why we used it: Users with similar interaction histories should have similar latent preferences, and videos engaged by similar users should have similar latent representations.

Implementation: We used truncated SVD on the early-period implicit-feedback matrix and recommended videos with high user-vector and item-vector similarity.

Common mistakes: Ignoring cold start, using random splits that leak future behavior, or treating offline hits as guaranteed product engagement.

### Recommendation Evaluation

What it does: Evaluates whether future engaged videos appear in each user's top-K recommendation list.

Why we used it: Recommendation is a ranking problem. The top few items matter more than the full item catalog.

Implementation: We evaluated Precision@K, Recall@K, Hit Rate@K, and NDCG@K for K values of 5, 10, and 20.

Interpretation: SVD improved NDCG@10 from 0.037 to 0.043 and Hit Rate@10 from 17.6% to 22.7% compared with popularity.

Common mistakes: Overclaiming offline ranking gains as retention gains, or ignoring that future positives are still affected by the production recommender's exposure policy.

## Phase 7: Connecting Personalization To Disengagement

### Segment-Level Personalization Analysis

What it does: Compares recommendation quality and personalization lift across user segments.

Why we used it: A recommender can improve average metrics while helping some users more than others. Product teams need to know where personalization is useful and where it struggles.

Implementation: We joined per-user recommendation metrics with behavioral segments from the Phase 3 modeling table, including early activity and activity-change segments.

Interpretation: SVD lift over popularity was stronger for high and very-high activity users than for low-activity users. That suggests collaborative filtering benefits from richer user histories.

Common mistakes: Assuming a segment with higher offline lift will necessarily retain better after intervention.

### Recommendation Evaluability

What it does: Identifies which users can be included in offline recommendation evaluation.

Why it matters here: Ranking metrics require future relevant items. Users who fully disengage have no late positive interactions, so their recommendation quality cannot be directly measured using this offline setup.

Implementation: We created a coverage table showing that 88.5% of returned users were recommendation-evaluable, while 0% of disengaged users were recommendation-evaluable.

Common mistakes: Giving disengaged users recommendation scores of zero and then claiming bad recommendations caused disengagement.

### Predictive, Observational, And Causal Evidence

What it does: Separates different strengths of evidence.

Why we used it: This project combines churn prediction and recommendation evaluation, but neither alone proves a retention intervention will work.

Interpretation: Phase 7 supports hypotheses such as using matrix-factorization personalization for users with enough history and exploring alternative strategies for low-history users.

Common mistakes: Treating offline personalization lift as proof of retention lift.

## Phase 8: Experiment Design

### A/B Testing

What it does: Randomly assigns eligible users to control or treatment, then compares outcomes.

Why we used it: Prior phases produced descriptive, predictive, and offline recommendation evidence. An A/B test is needed to estimate causal impact on retention.

Implementation: The proposed control is the current recommendation experience. The proposed treatment is a personalized feed strategy informed by collaborative filtering and disengagement-risk segments.

Common mistakes: Running an experiment without a primary metric, changing the analysis plan after seeing results, or interpreting offline model gains as experiment results.

### Unit Of Randomization

What it does: Defines what gets randomly assigned.

Why we chose users: Retention is a user-level outcome, and user-level assignment avoids giving the same user both control and treatment experiences.

Common mistakes: Randomizing impressions or sessions when the product question is user retention.

### Hypothesis Testing

What it does: Tests whether the observed treatment-control difference is compatible with the null hypothesis of no effect.

Null hypothesis:

```text
control_return_rate = treatment_return_rate
```

Alternative hypothesis:

```text
control_return_rate != treatment_return_rate
```

Why two-sided: The new personalization strategy could help or hurt.

### Sample Size Planning

What it does: Estimates how many users are needed to detect a chosen effect size with a chosen false-positive rate and power.

Why we used it: The baseline observed return rate is high, so small retention lifts require large samples.

Implementation: We used a two-sided two-proportion z-test approximation with alpha 0.05 and power 0.80.

Interpretation: Detecting a 1 percentage-point absolute retention lift from a 95.4% baseline requires about 6,140 users per group.

Common mistakes: Choosing sample size after looking at results, ignoring power, or treating statistical significance as business significance.

## Phase 9: Causal Inference Assessment

### Potential Outcomes

What it does: Frames causality by asking what would happen to the same user under treatment versus control.

Why it matters: We only observe one outcome per user in reality. Causal inference requires assumptions or randomization to estimate the missing counterfactual.

In this project: For the proposed personalization strategy, we do not observe the same eligible users randomized between current feed and SVD-informed feed.

### Treatment, Outcome, And Confounders

What they do: Define the causal question.

Treatment: The proposed personalization or re-entry strategy.

Outcome: Future observed return, disengagement, long-view rate, or session frequency.

Confounders: User intent, prior engagement, recommendation quality, notification exposure, device/session context, creator preferences, and external life factors.

Why this matters: A causal model is only credible if treatment assignment can be treated as random after adjustment or through design.

### Why We Declined Causal Modeling

What it does: Explicitly states that causal inference is not credible for the proposed retention intervention.

Why: KuaiRand-Pure does not randomly assign users to our proposed treatment. Observational adjustment would require assuming no unobserved confounding, which is not defensible here.

Interpretation: Phase 9 strengthens the project by drawing a clear boundary between prediction, offline recommendation quality, and causal evidence.

Common mistakes: Running propensity scores or regression adjustment simply because they sound advanced, or treating random exposure as proof that personalization improves retention.

## Phase 10: Final Product Recommendation

### Evidence Synthesis

What it does: Combines descriptive, predictive, recommendation, experimental, and causal-readiness evidence into a product recommendation.

Why we used it: Product teams need decisions, not only metrics. The recommendation should match the strength of the evidence.

Implementation: Findings were separated into descriptive findings, predictive findings, offline recommendation findings, and causal limitations.

Interpretation: The project supports testing segment-aware personalization, not claiming that personalization has already improved retention.

Common mistakes: Treating all findings as equally strong, or turning predictive associations into launch claims.

## Phase 11: Portfolio And Interview Preparation

### Portfolio Communication

What it does: Turns the analysis into a recruiter-readable and interviewer-defensible project.

Why we used it: A strong portfolio project needs narrative, reproducibility, limitations, and clear technical explanations.

Implementation: The project now includes a polished README, final report, resume bullets, 60-second explanation, 3-minute explanation, and interview question guide.

Common mistakes: Overstating impact, hiding limitations, or making the README too technical before explaining the business problem.

## Interview Questions

### Phase 1

Question: What is the unit of analysis in the raw logs?

Answer: The raw unit is an interaction event: one user exposed to or interacting with one video at one timestamp. Users, videos, days, and sessions are aggregations built from the event log.

Question: Why separate standard and random logs during exploration?

Answer: They come from different exposure mechanisms. Standard logs reflect the production recommender, while random logs include randomized video exposure. Combining them too early would mix product behavior with intervention behavior.

Question: Why are medians and percentiles important for user activity?

Answer: User activity is skewed. Highly active users can pull the mean upward, while medians and percentiles show typical and tail behavior.

Question: Does higher standard-log engagement prove the recommender caused higher engagement?

Answer: Not by itself. The result is consistent with better targeting, but differences may also reflect context, policy, ranking, or user intent. Causal claims require stronger assumptions or an experimental design.

Question: Why is churn hard to define in this dataset?

Answer: The data only shows observed activity during a limited window. It does not show account deletion or complete lifetime behavior, so churn must be framed as observed disengagement.

### Phase 2

Question: What was the retention denominator in Phase 2?

Answer: Users observed in the early standard recommendation log. We then measured whether those users appeared again in the late standard recommendation log.

Question: Why use early activity quartiles?

Answer: Quartiles produce simple, balanced, interpretable cohorts without arbitrary thresholds. They let us test whether relative early activity is associated with later return.

Question: Does higher retention among high-activity users mean activity causes retention?

Answer: No. The relationship is descriptive. High activity may reflect stronger pre-existing interest, better recommendations, user habits, or other confounders.

Question: What is censoring in this analysis?

Answer: Censoring means we do not observe all future behavior. Users first seen near the end of the dataset have less future time available, so their non-return is harder to interpret.

Question: Why not use survival analysis immediately?

Answer: The dataset does not contain true churn events, the window is short, and return rates are very high. Cohort analysis is more transparent for the current evidence.

### Phase 3

Question: What is the prediction unit in the Phase 3 modeling table?

Answer: One user observed in the early standard recommendation period.

Question: What is the difference between the observation window and prediction window?

Answer: The observation window is the earlier time period used to build features. The prediction window is the later period used to define the outcome label.

Question: Why did we exclude video aggregate statistics?

Answer: They may include future or global engagement information unavailable at prediction time, which could create leakage.

Question: Why might recency predict disengagement?

Answer: A user whose last event was far before the observation window ended may already be drifting away.

Question: Why is feature importance later not the same as causality?

Answer: A feature can help predict disengagement because it is correlated with user intent or context. That does not mean changing the feature would cause retention to improve.

### Phase 4

Question: What is the positive class in the churn model?

Answer: `disengaged_late = 1`, meaning the user was observed in the early standard period but not observed in the late standard period.

Question: Why is accuracy not the primary metric?

Answer: Only about 4.6% of users disengage. A model predicting every user returns would be about 95.4% accurate but would catch zero disengaged users.

Question: Why start with logistic regression?

Answer: It is interpretable, provides a strong baseline, and helps connect features to odds of disengagement before using more complex nonlinear models.

Question: What does PR-AUC tell us here?

Answer: It summarizes precision-recall tradeoffs for the rare disengaged class. It is especially useful because the positive class is uncommon.

Question: How should threshold choice be tied to product strategy?

Answer: If intervention is cheap and low-risk, we may choose higher recall and flag more users. If intervention is costly or annoying, we may choose higher precision and target fewer users.

### Phase 5

Question: What does a logistic regression coefficient mean?

Answer: It is the change in log odds of disengagement associated with a one-unit feature change, holding other model features fixed. For standardized numeric features, one unit means one standard deviation.

Question: Why can coefficients become counterintuitive?

Answer: Correlated predictors can change coefficient interpretation. A coefficient describes the association after controlling for other features, not the simple raw relationship.

Question: What does permutation importance measure?

Answer: It measures how much model performance drops when one feature is shuffled. In this project, we measured PR-AUC drop.

Question: Why is feature importance not causality?

Answer: A feature can help the model predict because it is correlated with hidden user intent, exposure policy, or other factors. Changing that feature may not change the outcome.

Question: How should Phase 5 insights be used by a product team?

Answer: As hypotheses for segmentation, personalization, or re-engagement experiments, not as proof that an intervention will improve retention.

### Phase 6

Question: Why use implicit feedback for recommendations?

Answer: The dataset does not contain ratings, so we infer preference from behaviors such as clicks, long views, likes, comments, follows, and forwards.

Question: Why compare against a popularity baseline?

Answer: Popularity is simple and often strong. A personalized model should beat it before we claim personalization adds value.

Question: What does NDCG@K measure?

Answer: It measures whether relevant items appear in the top K recommendations, giving more credit when relevant items are ranked higher.

Question: Why is temporal recommendation evaluation important?

Answer: It trains on past behavior and tests on future behavior, which better matches a real recommendation setting than random interaction splitting.

Question: Why do offline recommendation metrics not prove retention impact?

Answer: Offline metrics show ranking quality against observed future interactions. They do not prove that showing those recommendations would cause users to stay.

### Phase 7

Question: Why can we not compute recommendation metrics for fully disengaged users?

Answer: Ranking metrics require future relevant items. Fully disengaged users have no late positive interactions, so there is nothing observed to score recommendations against.

Question: Why is assigning zero recommendation quality to disengaged users risky?

Answer: It confuses missing future behavior with known bad recommendations and can create a false causal story.

Question: Which users benefited most from SVD personalization?

Answer: Higher-activity users benefited most in offline metrics, likely because collaborative filtering has more history to learn from.

Question: What does weak low-activity performance suggest?

Answer: Low-history users may need cold-start approaches such as popularity, exploration, content-based recommendations, or onboarding signals.

Question: What evidence is still needed before claiming personalization improves retention?

Answer: A randomized A/B test or another credible causal design measuring retention outcomes.

### Phase 8

Question: What is the control group?

Answer: Users receiving the current production recommendation experience.

Question: What is the treatment group?

Answer: Users receiving the proposed personalized feed strategy informed by collaborative filtering and risk segments.

Question: Why randomize at the user level?

Answer: Retention is user-level, and event-level randomization could expose the same user to both experiences.

Question: What is the primary metric?

Answer: 7-day observed return rate after experiment assignment.

Question: Why did we not report experiment results?

Answer: The dataset does not contain a randomized test of our proposed treatment, so reporting treatment effects would be fabricated.

### Phase 9

Question: What causal effect did we want to estimate?

Answer: The effect of a proposed personalization strategy on user retention or future observed return.

Question: Why did we not estimate that causal effect?

Answer: Users were not randomized to the proposed strategy, and observational adjustment would require strong unverified assumptions about unobserved confounding.

Question: What role does random exposure play?

Answer: It helps discuss exposure bias and engagement under less algorithmic selection, but it is not a clean randomized test of the proposed personalization strategy on retention.

Question: Why not use propensity scores?

Answer: Propensity scores only adjust for observed confounders. Important unobserved factors like user intent and context likely remain.

Question: What evidence would support a causal claim?

Answer: A randomized A/B test, or a credible quasi-experimental design with clearly justified identification assumptions.

### Phase 10

Question: What is the final product recommendation?

Answer: Test segment-aware personalization through a user-level A/B experiment instead of immediately claiming it improves retention.

Question: Which findings are causal?

Answer: None for retention impact. The project contains descriptive, predictive, and offline recommendation evidence, plus an experiment design.

Question: Which users would you target first?

Answer: Users with elevated disengagement risk and enough interaction history for collaborative filtering.

### Phase 11

Question: How would you summarize the project in one minute?

Answer: I used KuaiRand-Pure to analyze short-video engagement, model disengagement risk, evaluate personalization against popularity, and design an A/B test while carefully separating predictive evidence from causal claims.

Question: What makes this portfolio project defensible?

Answer: It uses realistic behavioral data, prevents leakage, evaluates imbalanced prediction properly, compares recommenders to a baseline, documents limitations, and avoids fabricated causal impact.
