# Interview Prep: User Disengagement And Personalization Project

This document summarizes the completed KuaiRand-Pure short-video recommendation project. Use it as a working study guide before discussing the project in Data Science interviews.

## 1. One-Minute Project Summary

I built a Data Science project using KuaiRand-Pure, a public short-video recommendation dataset from Kuaishou. The project studies user engagement, retention, disengagement risk, and personalization in a recommendation feed.

I started with product analytics to understand user-video interaction logs, then performed cohort retention analysis to see how early behavior related to later return. I engineered leakage-aware user-level behavioral features from early-period activity and trained churn/disengagement models to predict whether users would disappear in a later period. Because disengagement was rare, I evaluated models using PR-AUC, ROC-AUC, precision, recall, calibration, and top-risk threshold metrics instead of accuracy. I then interpreted the models using logistic coefficients and permutation importance and built a recommendation system using popularity and SVD matrix factorization. Finally, I connected recommendation quality to disengagement segments, while being careful not to make causal claims without an experiment.

## 2. Business Problem

The main question is:

> What behaviors predict user disengagement, which users are most at risk, and how can personalization improve their experience?

The product context is a short-video recommendation feed. The product team wants to understand which users may stop engaging and whether better recommendations could improve their experience.

Important distinction:

- We can predict and describe disengagement.
- We cannot claim an intervention improves retention until we run an experiment or have credible causal evidence.

## 3. Dataset

Dataset: KuaiRand-Pure.

Product: short-video recommendation feed.

User: anonymized app viewer.

Item: short video.

Event: a user-video interaction or exposure with a timestamp.

Engagement signals include:

- click
- long view
- like
- follow
- comment
- forward
- profile enter
- watch time
- negative feedback such as hate

Main logs:

- standard early log: production recommender behavior from April 9-April 21, 2022
- standard late log: production recommender behavior from April 21-May 8, 2022
- random late log: randomly exposed videos in the later period

Important caveat:

This is not complete app lifetime data. “Churn” in this project means observed future disengagement or non-return in the dataset, not confirmed account deletion.

## 4. What Has Been Done So Far

### Phase 1: Data Understanding And Product Metrics

Goal: Understand the raw behavioral logs before modeling.

Analysis performed:

- counted users, videos, events, and active days
- inspected timestamps
- checked missing values and duplicates
- compared standard recommendation logs with random exposure logs
- calculated click rate, long-view rate, like rate, hate rate, watch ratio, and daily active users
- examined user activity distributions
- created exploratory session metrics using a 30-minute inactivity rule

Main finding:

Standard recommendation engagement was much higher than random exposure engagement. Standard early click rate was about 46.3%, while random exposure click rate was about 17.6%.

How to explain it:

The production recommender is selecting videos that users are more likely to engage with than randomly selected videos. But this is descriptive; we should not overclaim causality.

### Phase 2: Cohort And Retention Analysis

Goal: Understand whether different user groups return at different rates.

Analysis performed:

- used early standard-period users as the retention denominator
- measured whether they appeared again in the late standard period
- created early activity cohorts: low, medium, high, very high
- compared observed return rates across cohorts
- examined first-observed-date cohorts and censoring issues

Main finding:

Early activity was strongly associated with later observed return.

Results:

- low early-activity users: 85.8% observed return rate
- medium: 96.9%
- high: 99.2%
- very high: 99.8%

How to explain it:

Users who are more active early are more likely to return later, but this is correlation, not causation. High activity may reflect stronger pre-existing interest or better recommendation fit.

### Phase 3: Behavioral Feature Engineering

Goal: Create a user-level modeling table for churn/disengagement prediction.

Analysis performed:

- converted event-level logs into one row per user
- built features only from early standard-period behavior
- defined the future label using late standard-period non-return
- excluded aggregate video statistics because they may leak future information

Feature families:

- frequency: event count, active days, events per active day
- recency: days since last early-period event
- engagement quality: click rate, long-view rate, like rate
- negative feedback: hate rate
- watch intensity: play time and watch ratio
- diversity: unique videos, authors, tags, tag entropy
- repeat behavior: repeat video and author rates
- trend: second-half vs first-half activity change

Main finding:

The modeling table had 26,210 users and a 4.6% disengagement rate.

How to explain it:

I separated features and labels by time. Early behavior became model inputs, and late behavior became the prediction target. This prevents leakage.

### Phase 4: Churn / Disengagement Modeling

Goal: Predict whether users will disengage later.

Models trained:

- all-users-return baseline
- logistic regression
- random forest
- histogram gradient boosting

Evaluation metrics:

- ROC-AUC
- PR-AUC
- precision
- recall
- F1
- Brier score
- calibration curves
- top-share threshold metrics

Main finding:

Random forest performed best in the first pass:

- ROC-AUC: 0.867
- PR-AUC: 0.233
- baseline PR-AUC: 0.046

At the riskiest 10% of users:

- precision: 23.2%
- recall: 50.7%

How to explain it:

The model ranks users by disengagement risk much better than random ranking. Because disengagement is rare, PR-AUC and recall/precision are more useful than accuracy.

### Phase 5: Model Interpretation And Product Insights

Goal: Understand what behaviors are associated with predicted disengagement.

Methods used:

- logistic regression coefficients
- odds ratios
- random forest permutation importance
- risk summaries by feature bins
- product hypothesis table

Important predictive signals:

- second-half event count
- total events
- unique videos
- unique authors
- events per active day
- recency
- tag diversity

Main product hypotheses:

- users with declining activity may need a refreshed recommendation mix
- users with high recency may need a better re-entry experience
- users with different content-diversity patterns may need different personalization strategies
- weak deep engagement suggests ranking should consider long-view quality, not only clicks

How to explain it:

These are predictive associations, not causal findings. Feature importance tells us what the model used, not what would happen if we changed a product experience.

### Phase 6: Recommendation / Personalization

Goal: Test whether personalized recommendations outperform a simple popularity baseline.

Models built:

- popularity baseline
- SVD matrix factorization

Recommendation setup:

- train on early standard positive interactions
- evaluate on late standard positive interactions
- define positive feedback as click, long view, like, follow, comment, or forward

Metrics:

- Precision@K
- Recall@K
- Hit Rate@K
- NDCG@K

Main result at K=10:

- popularity NDCG@10: 0.0368
- SVD NDCG@10: 0.0431
- popularity Hit Rate@10: 17.6%
- SVD Hit Rate@10: 22.7%

How to explain it:

SVD personalization beat popularity offline, but the improvement was modest. That is realistic because popularity is often a strong baseline.

### Phase 7: Connecting Personalization To Disengagement

Goal: Understand which user segments benefit from personalization and how this relates to disengagement.

Analysis performed:

- joined user-level recommendation quality with behavioral segments
- compared SVD lift over popularity by activity segment
- checked which users could be evaluated with recommendation metrics

Main finding:

Fully disengaged users had no future positive interactions, so recommendation metrics were undefined for them.

Among evaluable returning users, SVD lift over popularity was strongest for high-activity users:

- low activity NDCG@10 lift: -0.001
- medium: 0.003
- high: 0.009
- very high: 0.011

How to explain it:

Collaborative filtering works better when users have enough interaction history. Low-activity users remain a cold-start challenge.

### Phase 8: Experiment Design

Goal: Design a valid A/B test to measure whether personalization causes better retention.

Important note:

KuaiRand-Pure does not contain a randomized experiment for our proposed personalization strategy, so we designed an experiment but did not report treatment results.

Experiment design:

- control: current production recommendation experience
- treatment: personalized feed strategy informed by collaborative filtering and disengagement-risk segments
- unit of randomization: user
- eligibility: active users with enough prior interaction history for personalization
- primary metric: 7-day observed return rate
- secondary metrics: long-view rate, sessions per user, watch time per active user, recommendation Hit Rate@K
- guardrails: hate rate, short-view rate, app exits if available, creator/content concentration

Hypotheses:

```text
H0: treatment_return_rate = control_return_rate
H1: treatment_return_rate != control_return_rate
```

Sample-size planning:

Using the observed baseline return rate of about 95.4%, alpha 0.05, and 80% power:

- detecting a 0.5 percentage-point lift requires about 26,001 users per group
- detecting a 1.0 percentage-point lift requires about 6,140 users per group
- detecting a 2.0 percentage-point lift requires about 1,353 users per group

How to explain it:

Offline models helped identify what to test, but only a randomized experiment can estimate whether the personalization change causes retention to improve.

### Phase 9: Causal Inference Assessment

Goal: Decide whether the dataset supports causal inference beyond experiment design.

Main conclusion:

We should not estimate a causal retention effect from KuaiRand-Pure for the proposed personalization strategy.

Why:

- users were not randomized to our proposed SVD-informed personalization treatment
- standard recommendation logs are generated by the existing recommender, not a clean control group
- random exposure helps with exposure-bias discussion, but it is not a user-level retention experiment
- important confounders such as user intent, notification exposure, session context, and external factors are unobserved

What random exposure can support:

It helps show that engagement differs when videos are selected randomly rather than by the standard recommender. For example, random exposure had much lower click and long-view rates than standard recommendations.

What random exposure cannot support:

It cannot prove that our proposed personalization strategy causes better retention.

How to explain it:

> I considered causal inference, but concluded the dataset does not support a credible causal estimate of personalization's effect on retention. I used the random exposure logs to reason about exposure bias, but I would rely on the proposed A/B test to estimate causal impact.

### Phase 10: Final Product Recommendation

Goal: Synthesize the evidence into a product recommendation.

Final recommendation:

> Test segment-aware personalization in a user-level A/B experiment. Use collaborative filtering for users with enough history, and use cold-start strategies such as popularity, exploration, or content-based recommendations for low-history users.

Evidence behind the recommendation:

- churn models can identify elevated-risk users better than baseline
- SVD personalization beats popularity offline
- SVD helps high-activity users more than low-activity users
- the dataset does not prove retention impact causally

How to explain it:

> My recommendation is not to ship the model as a proven retention solution. It is to test a focused personalization strategy with retention as the primary metric and user-experience guardrails.

### Phase 11: Portfolio And Interview Preparation

Goal: Package the project for recruiters and prepare to defend it in interviews.

Created assets:

- polished README
- final product report
- resume bullets
- 60-second explanation
- 3-minute technical explanation
- mock interview questions
- limitations and decision logs

How to explain it:

> The project is structured as a complete Data Science case study: business problem, data, behavioral insights, predictive model, personalization, experiment design, causal assessment, and product recommendation.

## 5. Key Concepts To Know

### Unit Of Analysis

The raw data is event-level. Each row is a user-video interaction at a timestamp.

For churn modeling, the unit becomes user-level: one row per user.

For recommendation modeling, the unit becomes user-item-level: a user-video pair.

### Standard Logs Vs Random Exposure

Standard logs come from the production recommender. Random exposure logs contain videos shown randomly from a candidate pool.

Why this matters:

Standard logs are affected by the recommender’s choices. Users can only click videos they were shown. This creates exposure bias.

### Engagement

Engagement is measured using behavioral signals such as clicks, long views, likes, comments, follows, forwards, and watch time.

Important caveat:

Engagement is a proxy for satisfaction. A click does not always mean the user was satisfied.

### Retention / Return Rate

Return rate means:

```text
users_seen_in_both_early_and_late_periods / users_seen_in_early_period
```

In this project, retention means observed return in the dataset, not confirmed app retention.

### Cohort Analysis

Cohort analysis groups users by something known before the outcome period and compares future behavior.

Example:

Group users by early activity level, then compare later return rates.

### Censoring

Censoring means we do not observe a user’s full future.

Example:

A user first seen near the end of the dataset has less time to return, so non-return is harder to interpret.

### Feature Engineering

Feature engineering turns raw logs into model inputs.

Examples:

- number of events
- active days
- recency
- click rate
- long-view rate
- tag entropy
- activity change ratio

### Feature Leakage

Leakage happens when a feature contains information that would not be available at prediction time.

In this project:

```text
early period = features
late period = label
```

We avoided late-period behavior and suspicious aggregate video statistics as features.

### Class Imbalance

Only about 4.6% of users disengaged.

Why this matters:

A model could predict every user returns and still look highly accurate. That is why accuracy is not the main metric.

### Logistic Regression

Logistic regression estimates:

```text
p(disengagement) = 1 / (1 + exp(-(b0 + b1*x1 + ... + bk*xk)))
```

It models log odds, not direct probability changes.

Odds ratio:

```text
odds_ratio = exp(coefficient)
```

### Random Forest

A random forest is an ensemble of decision trees. It can capture nonlinear patterns and feature interactions better than logistic regression.

### Gradient Boosting

Gradient boosting builds trees sequentially, where each new tree tries to correct previous errors.

### ROC-AUC

ROC-AUC measures how well the model ranks a randomly chosen disengaged user above a randomly chosen non-disengaged user.

Limitation:

It can look good even when rare-class precision is modest.

### PR-AUC

PR-AUC summarizes precision-recall tradeoffs for the positive class.

Why it matters here:

Disengagement is rare, so PR-AUC is more informative than accuracy.

Baseline PR-AUC is approximately the positive-class prevalence, about 0.046.

### Precision

```text
precision = true_positives / (true_positives + false_positives)
```

Meaning:

Of users flagged as high risk, how many actually disengaged?

### Recall

```text
recall = true_positives / (true_positives + false_negatives)
```

Meaning:

Of users who actually disengaged, how many did the model catch?

### Calibration

Calibration asks whether predicted probabilities match observed outcomes.

Example:

If the model says a group of users has 20% risk, about 20% of them should actually disengage.

### Brier Score

```text
brier = mean((predicted_probability - actual_outcome)^2)
```

It measures probability prediction error.

### Permutation Importance

```text
importance = original_score - score_after_shuffling_feature
```

If shuffling a feature hurts performance, the model relied on that feature.

Limitation:

It is not causal and can be affected by correlated features.

### Implicit Feedback

Recommendation systems often use behavior instead of ratings.

In this project, positive feedback includes:

- click
- long view
- like
- follow
- comment
- forward

### Popularity Baseline

The popularity baseline recommends the same popular videos to everyone.

Why it matters:

Personalized recommenders should beat a simple popularity baseline.

### Matrix Factorization

Matrix factorization represents users and videos as vectors in a lower-dimensional latent space.

Intuition:

Users with similar behavior get similar vectors, and videos consumed by similar users get similar vectors.

### Recommendation Metrics

Precision@K:

```text
relevant recommended items in top K / K
```

Recall@K:

```text
relevant recommended items in top K / all relevant future items
```

Hit Rate@K:

```text
share of users with at least one relevant item in top K
```

NDCG@K:

Rewards relevant items more when they appear higher in the ranking.

### Cold Start

Cold start means the model has little or no history for a user or item.

In this project:

Low-activity users were harder for SVD to personalize.

### Offline Metrics Vs Product Impact

Offline recommendation metrics show whether the model recovers future observed engaged videos.

They do not prove:

```text
better recommendations cause better retention
```

That requires an experiment.

### A/B Testing

A/B testing randomly assigns users to product experiences and compares outcomes.

In this project:

- A = current recommendation system
- B = proposed personalization strategy

The goal is to estimate causal impact on retention.

### Unit Of Randomization

The unit of randomization is what gets assigned to control or treatment.

In this project, the unit should be the user because retention is measured at the user level.

### Null And Alternative Hypotheses

Null hypothesis:

```text
treatment effect = 0
```

Alternative hypothesis:

```text
treatment effect != 0
```

A two-sided test is safer because the treatment could help or hurt.

### Type I And Type II Error

Type I error:

> False positive. We conclude the treatment worked when it did not.

Type II error:

> False negative. We fail to detect a real effect.

### Alpha

Alpha is the false-positive rate we are willing to tolerate.

Common value:

```text
alpha = 0.05
```

### Power

Power is the probability of detecting a real effect of a given size.

Common value:

```text
power = 0.80
```

### Minimum Detectable Effect

MDE is the smallest effect the experiment is designed to reliably detect.

Important idea:

Smaller MDEs require larger sample sizes.

### Practical Vs Statistical Significance

Statistical significance asks whether the result is unlikely under the null hypothesis.

Practical significance asks whether the effect is large enough to matter for the business.

### Potential Outcomes

Potential outcomes are the outcomes a user would have under different treatments.

Example:

```text
Y_user(control) = whether the user returns under current recommendations
Y_user(treatment) = whether the user returns under the new personalization strategy
```

We only observe one of these for each user, so causal inference requires randomization or strong assumptions.

### Confounding

Confounding happens when a factor affects both treatment assignment and the outcome.

Example:

Highly motivated users may receive better recommendations and also be more likely to return. If we do not account for motivation, we may falsely attribute retention to recommendations.

### Identification Assumption

An identification assumption is what allows us to interpret an estimate causally.

Example:

Random assignment supports the assumption that treatment and control users are comparable on average.

In this project, observational causal assumptions are not credible enough for retention impact.

## 6. Most Challenging Topics

### Feature Leakage

You must explain why features must be measured before the outcome.

Strong answer:

> I used early-period behavior as features and late-period behavior only as the outcome label. I avoided aggregate video statistics because their timing was unclear and they could contain future engagement.

### Class Imbalance

You must explain why accuracy is misleading.

Strong answer:

> Only about 4.6% of users disengaged. A model predicting everyone returns would appear about 95.4% accurate but would catch zero disengaged users.

### PR-AUC Vs ROC-AUC

Strong answer:

> ROC-AUC measures ranking across positives and negatives. PR-AUC focuses on precision and recall for the rare positive class, which matters more when trying to find disengaged users.

### Correlation Vs Causation

Strong answer:

> The model can show that declining activity predicts disengagement, but it cannot prove that increasing activity would prevent disengagement. That requires an experiment.

### Feature Importance

Strong answer:

> Feature importance tells us what the model relies on. It does not prove the feature causes churn, especially when features are correlated.

### Recommendation Evaluation Bias

Strong answer:

> Offline recommendation metrics are based on future observed positives, but those positives were shaped by the production recommender. Missing interactions are not true negatives.

### Cold Start

Strong answer:

> Matrix factorization needs user history. It helped high-activity users more than low-activity users, so low-history users may need popularity, exploration, content-based recommendations, or onboarding signals.

### Experiment Sample Size

Strong answer:

> With a high baseline return rate, a tiny absolute lift is hard to distinguish from random noise. For example, detecting a 1 percentage-point lift requires about 6,140 users per group, while detecting a 0.5-point lift requires about 26,001 per group.

### Declining Causal Inference

Strong answer:

> I did not force causal inference because the proposed personalization treatment was not randomized in the historical data. Running propensity scores would only adjust for observed variables and would not solve unobserved user intent or context.

## 7. Common Interview Questions And Answers

### Project Framing

Question: What problem were you solving?

Answer: I studied user disengagement in a short-video recommendation feed. The goal was to identify behavioral predictors of future non-return and evaluate whether personalization could improve recommendation quality.

Question: What dataset did you use?

Answer: I used KuaiRand-Pure, a public short-video recommendation dataset from Kuaishou with user-video interactions, timestamps, engagement signals, user features, video metadata, standard recommendation logs, and random exposure logs.

Question: What is the unit of analysis?

Answer: The raw data is event-level. For churn modeling I aggregated to one row per user. For recommendation modeling I used user-video interactions.

### Product Metrics

Question: How did you define engagement?

Answer: I treated engagement as a spectrum: clicks and valid plays are lighter engagement, long views are deeper engagement, and likes/comments/follows/forwards are stronger explicit positive signals.

Question: How did you define retention?

Answer: Retention was defined as users observed in the early standard period appearing again in the late standard period.

Question: Is that true churn?

Answer: No. It is observed non-return in the dataset. We do not observe account deletion or full user lifetime.

### Cohort Analysis

Question: What did your cohort analysis show?

Answer: Early activity level was strongly associated with later observed return. Low-activity users returned at about 85.8%, while very-high-activity users returned at about 99.8%.

Question: Does that mean activity causes retention?

Answer: No. It is descriptive. High activity may reflect existing user interest, better recommendation fit, or user habits.

### Feature Engineering

Question: What features did you create?

Answer: I created recency, frequency, active days, engagement rates, watch intensity, content diversity, repeat behavior, and activity-change features.

Question: How did you prevent leakage?

Answer: I built features only from the early standard period and used the late standard period only to define the outcome. I excluded aggregate video statistics because their timing was unclear.

### Modeling

Question: Why did you not use accuracy as the main metric?

Answer: The positive class was rare: only about 4.6% of users disengaged. Accuracy would reward a model that predicts everyone returns.

Question: Which model performed best?

Answer: Random forest performed best in the first pass, with ROC-AUC around 0.867 and PR-AUC around 0.233.

Question: Is PR-AUC of 0.233 good?

Answer: It should be compared with the baseline prevalence of about 0.046. So it is meaningfully better than random ranking, though precision is still modest.

Question: How would the product team use the model?

Answer: As a risk-ranking tool. For example, if the team can target the riskiest 10% of users, the random forest identified about 50.7% of disengaged users with 23.2% precision.

### Interpretation

Question: What features mattered most?

Answer: Important signals included second-half event count, total events, unique videos, unique authors, events per active day, recency, and tag diversity.

Question: What does an odds ratio mean?

Answer: It is `exp(coefficient)` from logistic regression. It describes the multiplicative change in odds of disengagement for a one-unit feature increase, holding other features fixed.

Question: Why can logistic coefficients be hard to interpret?

Answer: Many activity features are correlated. Once the model controls for one activity variable, another coefficient may become counterintuitive.

### Recommendation Systems

Question: Why did you use implicit feedback?

Answer: The dataset does not have explicit ratings, so I inferred positive preference from behaviors like clicks, long views, likes, follows, comments, and forwards.

Question: Why compare SVD to popularity?

Answer: Popularity is a simple, strong baseline. A personalized recommender needs to beat it before we can claim personalization adds offline value.

Question: What did SVD improve?

Answer: At K=10, SVD improved NDCG@10 from 0.0368 to 0.0431 and Hit Rate@10 from 17.6% to 22.7%.

Question: Who benefited most from personalization?

Answer: Higher-activity users. SVD needs enough interaction history to learn useful user preferences.

Question: Why can’t we evaluate recommendation quality for users who disengaged?

Answer: Recommendation metrics require future relevant items. Fully disengaged users have no late positive interactions, so there is nothing observed to score against.

### Causality And Experimentation

Question: Can you claim personalization improves retention?

Answer: Not yet. I can say SVD improved offline recommendation metrics. To claim retention impact, I would need an A/B test or credible causal design.

Question: What would you test next?

Answer: I would design an A/B test comparing the current feed against a personalization strategy, with retention or long-view engagement as the primary metric and guardrails for negative feedback.

Question: What is the unit of randomization for the experiment?

Answer: User-level randomization, because the outcome is user retention and we do not want the same user exposed to both control and treatment experiences.

Question: What is the primary metric in the proposed experiment?

Answer: 7-day observed return rate after assignment.

Question: Why use a two-sided test?

Answer: Because the personalization strategy could improve or harm retention. A two-sided test does not assume the direction in advance.

Question: Why did you not report an experiment treatment effect?

Answer: The dataset does not contain a randomized test of our proposed personalization treatment. Reporting an effect would fabricate business impact.

Question: What sample size would be needed for a 1 percentage-point lift?

Answer: Using the observed 95.4% baseline return rate, alpha 0.05, and 80% power, about 6,140 users per group.

Question: Why did you decline causal inference in Phase 9?

Answer: The dataset does not randomize users into the proposed personalization strategy, and important confounders are unobserved. A causal estimate would rely on assumptions I cannot defend.

Question: Could random exposure be used as a treatment?

Answer: It can be used descriptively to discuss exposure bias and engagement under random item exposure, but it is not a clean user-level randomized treatment for retention.

Question: Why not use propensity score matching?

Answer: Propensity scores only adjust for observed covariates. If user intent, context, or notification exposure are unobserved, matching does not remove that confounding.

Question: What would you need for a credible causal claim?

Answer: A randomized A/B test or a strong quasi-experimental design with a clear treatment, outcome, comparison group, and defensible identification assumptions.

## 8. What To Say On A Resume

Two-bullet version:

- Analyzed 2.6M+ KuaiRand-Pure short-video recommendation events to study engagement, retention, and user disengagement using product metrics, cohort analysis, and leakage-aware feature engineering.

- Trained churn models and recommendation baselines using logistic regression, random forest, gradient boosting, popularity ranking, and SVD matrix factorization; improved churn PR-AUC from a 0.046 baseline to 0.233 and SVD Hit Rate@10 from 17.6% to 22.7%.

Short project title:

```text
User Disengagement And Personalization In Short-Video Recommendations
```

## 9. What Is Next

Next step: interview practice and optional repository cleanup.

The next question is:

> Can you explain every major choice, metric, model, limitation, and recommendation without reading from the docs?

You should practice:

- descriptive findings
- predictive findings
- recommendation findings
- experiment design
- causal limitations
- the 60-second explanation
- the 3-minute technical explanation
- defending PR-AUC vs accuracy
- explaining leakage prevention
- explaining SVD and cold start
- explaining why there is no causal retention claim
- walking through the A/B test design

The key final idea:

> The project is strongest when you clearly separate what was observed, what was predicted, what was evaluated offline, and what still needs experimental proof.
