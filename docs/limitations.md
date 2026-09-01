# Limitations

This document records weaknesses and uncertainty as the project evolves.

## Dataset Representativeness

KuaiRand-Pure is a filtered public dataset, not complete Kuaishou app history. Results may not generalize to all users, all videos, or all time periods.

## Churn Definition

The dataset does not show app deletion, account cancellation, or complete future activity. Any churn label must be defined as observed disengagement or non-return within the available logs.

## Selection Bias

The observed early-to-late return rate among standard-log users is about 95.4%, which is unusually high for general consumer churn analysis. This suggests the data may overrepresent active users who were eligible for logging.

## Exposure Bias

Users only interact with videos they were shown. Standard recommendation logs are shaped by the existing recommender, so observed clicks and watches are not unbiased measures of preference.

## Random Exposure Scope

Random exposure helps study exposure bias, but it does not automatically identify the causal effect of a full personalization strategy on retention.

## Sessionization

The dataset does not include explicit session IDs. Any session definition based on inactivity gaps is heuristic and should be validated before being used for major conclusions.

## Item Aggregate Features

Video-level aggregate statistics may include information from outside a prediction window. These features can create leakage if used without time-aware controls.

## Cohort Analysis Is Descriptive

Phase 2 shows that early activity cohorts have different observed return rates. This does not prove that increasing a user's activity would cause higher retention. High activity may be a symptom of underlying interest rather than a cause.

## First Observed Date Is Not Signup Date

First observed date in the logs is not the user's true registration or first app-use date. Cohorts based on first observed date should be interpreted as dataset-entry cohorts, not acquisition cohorts.

## Censoring In Retention Curves

Users first observed later in the dataset have less follow-up time. Daily retention curves are therefore less reliable for later first-seen cohorts.

## Feature Engineering Loses Sequence Detail

Phase 3 aggregates event histories into user-level features. This improves interpretability but loses fine-grained ordering of events.

## Disengagement Label Imbalance

Only about 4.6% of early standard-log users are not observed in the late standard period. Phase 4 models must account for this class imbalance.

## Metadata Timing

Basic video metadata is used for diversity features. Aggregate video statistics are excluded for now because their timing is unclear and they may leak future engagement.

## Single Outcome Window

Phase 4 uses one early observation window and one late outcome window. This protects the feature-label time order, but it is not the same as validating across many future calendar periods.

## Offline Prediction Is Not Intervention Impact

The churn model can identify users associated with future disengagement, but it does not prove that any intervention will prevent disengagement.

## Thresholds Need Product Costs

Top-share threshold metrics show tradeoffs, but choosing an operating point requires assumptions about intervention cost, user annoyance, and expected benefit.

## Interpretation Is Model-Specific

Phase 5 interpretation explains how the fitted models behave. It does not prove that the identified features are universal drivers of disengagement.

## Correlated Features Complicate Interpretation

Activity features such as total events, active days, events per active day, and second-half events overlap conceptually. Their coefficients and importances should not be interpreted independently without caution.

## Product Hypotheses Require Experiments

Suggested interventions from Phase 5 are hypotheses. They need A/B testing or credible causal evidence before being treated as product recommendations.

## Recommendation Evaluation Uses Observed Positives

Phase 6 evaluates recommendations against videos users engaged with in the late standard log. Those future positives were still shaped by the production recommender's exposure policy.

## Missing Interactions Are Not True Negatives

If a user did not engage with a video, we usually do not know whether they disliked it or simply never had a fair chance to see it.

## Cold Start

Matrix factorization requires user and item history. It cannot directly personalize for users or videos with no training interactions.

## Offline Ranking Gains Are Not Retention Gains

SVD improved offline ranking metrics over popularity, but this does not prove it would improve retention in production.

## Recommendation Metrics Are Conditional On Return

Phase 7 can evaluate recommendation quality only for users with future positive interactions. This means offline recommendation metrics are conditional on users returning and engaging.

## Disengaged Users Have Undefined Recommendation Quality

Users who are not observed in the late standard period have no future positives. Their recommendation quality should not be coded as zero because that would treat unobserved outcomes as known failures.

## Segment Findings Are Not Targeting Proof

SVD lift varies by activity segment, but this does not prove which segment should be targeted in production. Targeting decisions require intervention costs and experimental evidence.

## Phase 8 Is A Design, Not A Result

The A/B test in Phase 8 is a proposed production experiment. KuaiRand-Pure does not contain randomized treatment results for our personalization strategy.

## Sample Size Estimates Depend On Assumptions

The sample-size table uses the observed dataset return rate as a planning baseline. Production baseline retention may differ.

## Interference And Spillovers

The experiment design assumes users are independently affected by treatment. Social features, creator exposure, or network effects could violate this assumption.

## No Credible Causal Estimate For Personalization Impact

Phase 9 concludes that KuaiRand-Pure does not support estimating the causal effect of our proposed personalization strategy on retention.

## Random Exposure Is Limited

Random exposure is useful for studying exposure bias, but it is not equivalent to randomizing users into our proposed treatment and control feed experiences.

## Unobserved Confounding

Important factors such as user intent, context, notification exposure, device behavior, and external circumstances are not fully observed. This limits observational causal inference.

## Final Recommendation Is A Test Proposal

The final product recommendation is to run an experiment, not to ship the personalization strategy immediately.

## Portfolio Metrics Are Offline

Resume and README metrics describe offline analysis results. They should not be described as production impact or business lift.
