# Final Product Recommendation

## Executive Summary

This project studied user disengagement and personalization in KuaiRand-Pure, a public short-video recommendation dataset from Kuaishou. The analysis combined product analytics, cohort retention analysis, churn prediction, recommendation modeling, experiment design, and causal-readiness assessment.

The main conclusion is:

> Early activity, recency, activity change, and content-diversity signals can identify users at higher risk of observed disengagement. Personalized recommendations using SVD matrix factorization outperform popularity offline, especially for users with enough interaction history. However, the dataset does not support a causal claim that personalization improves retention. The product team should test a targeted personalization strategy through a user-level A/B experiment.

## 1. What Did We Discover?

Standard recommendations generated substantially higher engagement than random exposure:

- standard early click rate: 46.3%
- random exposure click rate: 17.6%
- standard early long-view rate: 33.7%
- random exposure long-view rate: 8.5%

Early activity was strongly associated with later observed return:

- low early-activity return rate: 85.8%
- very-high early-activity return rate: 99.8%

The disengagement label was rare:

- disengagement rate: 4.6%
- disengaged users: 1,200 out of 26,210 early-period users

The random forest churn model ranked users by disengagement risk better than baseline:

- baseline PR-AUC: 0.046
- random forest PR-AUC: 0.233
- random forest ROC-AUC: 0.867

At the riskiest 10% of users, the random forest identified:

- 50.7% of disengaged users
- 23.2% precision

SVD matrix factorization outperformed popularity for recommendation:

- popularity Hit Rate@10: 17.6%
- SVD Hit Rate@10: 22.7%
- popularity NDCG@10: 0.0368
- SVD NDCG@10: 0.0431

Personalization lift was strongest for high-activity users and weakest for low-activity users.

## 2. Which Findings Are Descriptive?

Descriptive findings summarize what happened in the observed data:

- standard recommendation logs had higher engagement than random exposure logs
- user activity was highly skewed
- low-activity users had lower observed return than high-activity users
- higher-activity users received more offline recommendation benefit from SVD

These findings are useful for understanding the product, but they are not causal.

## 3. Which Findings Are Predictive?

Predictive findings identify patterns that help forecast later observed disengagement:

- second-half activity was important for risk ranking
- total events and events per active day helped identify risk
- recency was associated with disengagement risk
- unique videos/authors and tag diversity contributed to model performance
- random forest and logistic regression outperformed the naive baseline

These findings support risk ranking and segmentation, but they do not prove what intervention would work.

## 4. Which Findings Are Causal?

No causal retention effect was estimated.

KuaiRand-Pure does contain random exposure logs, which help discuss exposure bias. However, the dataset does not randomly assign users to our proposed personalization strategy. Therefore, it does not support a causal claim that SVD personalization improves retention.

The correct causal next step is a user-level A/B test.

## 5. Recommended Intervention To Test

Test a segment-aware personalization strategy:

- For users with enough history: use collaborative-filtering personalization, such as matrix factorization, because SVD outperformed popularity for higher-activity users.
- For low-history users: use a hybrid strategy combining popularity, exploration, and content-based signals because collaborative filtering has limited data for them.
- For high-risk users: test a re-entry or feed-refresh experience that balances familiar content with controlled exploration.

This should be framed as a hypothesis:

> Better segment-aware personalization may improve observed return and engagement quality.

It should not be framed as proven impact.

## 6. Which Users Would We Target?

Primary target segment for the first experiment:

- active users with enough prior history for personalization
- users showing declining activity or rising recency
- users ranked in the top risk deciles by the churn model

Reason:

This segment has enough behavioral history for personalization and enough disengagement risk to make intervention valuable.

Secondary segment:

- low-history users

Reason:

They are harder for collaborative filtering, so they need a separate cold-start strategy. They should not be treated as if SVD will work equally well for them.

## 7. What Metrics Determine Success?

Primary metric:

- 7-day observed return rate

Secondary metrics:

- long-view rate
- watch time per active user
- sessions per user
- recommendation Hit Rate@K or NDCG@K if online recommendation logging supports it

Guardrail metrics:

- hate rate
- short-view rate
- app exits if available
- creator/content concentration
- latency or feed-load performance if available in production

Sample-size planning:

Using the observed 95.4% return-rate baseline, alpha 0.05, and 80% power:

- 1 percentage-point lift requires about 6,140 users per group
- 0.5 percentage-point lift requires about 26,001 users per group

## 8. Biggest Limitations

- Churn is observed non-return, not confirmed account deletion.
- The dataset is not complete user lifetime history.
- Standard logs are affected by the production recommender.
- Missing interactions are not true dislikes.
- Recommendation evaluation is exposure-biased.
- Fully disengaged users cannot be evaluated with future recommendation metrics because they have no future positives.
- Offline recommendation metrics do not prove retention impact.
- Causal inference is not credible for the proposed treatment without a randomized experiment.

## 9. What Should The Team Investigate Next?

1. Run the Phase 8 A/B test for segment-aware personalization.
2. Add online logging for recommendation impressions, ranks, clicks, long views, skips, and exits.
3. Build a cold-start strategy for low-history users.
4. Compare click-optimized versus long-view-optimized ranking.
5. Monitor whether personalization narrows content diversity too much.
6. Validate churn models across multiple future calendar windows.

## Final Recommendation

The team should not immediately ship the proposed recommender as a retention solution. The evidence supports a targeted experiment:

> Test segment-aware personalization for users with sufficient history and elevated disengagement risk, while separately developing cold-start strategies for low-history users.

The launch decision should depend on measured retention lift, engagement quality, and guardrails from a randomized experiment.
