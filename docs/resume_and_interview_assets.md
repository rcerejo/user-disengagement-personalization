# Resume And Interview Assets

## Resume Bullets

- Analyzed 2.6M+ KuaiRand-Pure short-video recommendation events to study engagement, retention, and user disengagement using product metrics, cohort analysis, and leakage-aware feature engineering.

- Trained churn and recommendation models using logistic regression, random forest, gradient boosting, popularity ranking, and SVD matrix factorization; improved churn PR-AUC from a 0.046 baseline to 0.233 and SVD Hit Rate@10 from 17.6% to 22.7%.

## 60-Second Explanation

I built a Data Science project around user disengagement and personalization in a short-video recommendation feed using the KuaiRand-Pure dataset. I started by analyzing user-video interaction logs and product metrics like click rate, long-view rate, activity distributions, and observed return rate. Then I ran cohort retention analysis and found that early activity level was strongly associated with later return, though I treated that as correlation rather than causation.

Next, I engineered user-level behavioral features from early-period behavior, including recency, frequency, engagement rates, content diversity, and activity trends. I used those features to train churn models and evaluated them under class imbalance using PR-AUC, ROC-AUC, precision, recall, calibration, and threshold metrics. I also built a recommendation component comparing popularity against SVD matrix factorization. SVD improved offline Hit Rate@10 and NDCG@10, especially for higher-activity users. Finally, I designed an A/B test and concluded that the dataset does not support a causal retention claim without an experiment.

## 3-Minute Technical Explanation

The project uses KuaiRand-Pure, which contains short-video recommendation logs with user IDs, video IDs, timestamps, engagement signals, user features, video features, standard recommendation logs, and random exposure logs. The main product question is whether we can identify behaviors that predict disengagement and whether personalization could improve the user experience.

I began with data understanding. I treated each row as an event-level user-video interaction, then summarized the logs by users, videos, events, timestamp coverage, missingness, duplicates, and engagement rates. One early insight was that standard recommendations had much higher click and long-view rates than random exposure, which is consistent with the production recommender selecting more engaging content. I kept standard and random logs separate because they come from different exposure mechanisms.

For retention analysis, I used users in the early standard period as the denominator and measured whether they appeared again in the late standard period. I created early activity cohorts and found that low-activity users had an 85.8% observed return rate, while very-high-activity users had a 99.8% return rate. I framed this as descriptive because activity was not randomized and first observed date is not signup date.

For modeling, I created a user-level table with one row per early-period user. Features included recency, frequency, active days, engagement rates, watch intensity, content diversity, repeat behavior, and activity change. The label was `disengaged_late`, meaning the user did not appear in the late standard log. I avoided leakage by using only early-period features and excluding aggregate video statistics whose timing was unclear.

The disengagement rate was only 4.6%, so I avoided accuracy as the main metric. I trained logistic regression, random forest, and histogram gradient boosting models. The random forest achieved ROC-AUC of 0.867 and PR-AUC of 0.233 compared with a 0.046 prevalence baseline. At the top 10% risk threshold, it caught about 50.7% of disengaged users with 23.2% precision.

For interpretation, I used logistic coefficients and random forest permutation importance. Important predictive signals included second-half event count, total events, unique videos/authors, events per active day, recency, and tag diversity. I emphasized that feature importance is not causality.

For personalization, I defined implicit positive feedback using clicks, long views, likes, follows, comments, and forwards. I compared popularity recommendations to SVD matrix factorization using a temporal train/test split. At K=10, SVD improved Hit Rate from 17.6% to 22.7% and NDCG from 0.0368 to 0.0431. Segment analysis showed that SVD helped higher-activity users more than low-activity users, highlighting cold start.

Finally, I designed a user-level A/B test to measure whether segment-aware personalization improves 7-day observed return. I did not report treatment effects because the dataset does not contain a randomized test of our proposed intervention. The final recommendation is to test segment-aware personalization for users with enough history and elevated disengagement risk, while developing separate cold-start strategies for low-history users.

## Common Interview Challenges

### Why Is Accuracy Misleading?

Only 4.6% of users disengaged. Predicting every user returns would look about 95.4% accurate but catch zero disengaged users. Precision, recall, and PR-AUC are better aligned with finding at-risk users.

### What Is Feature Leakage Here?

Leakage would happen if future behavior from the late period were used as model input. I prevented this by using early-period behavior as features and late-period non-return only as the label.

### Why Is Feature Importance Not Causality?

A feature can help the model predict because it is correlated with hidden intent, exposure, or context. That does not mean changing the feature would change retention.

### Why Compare To Popularity?

Popularity is a simple, strong recommendation baseline. A personalized recommender must beat it before we can claim personalization adds offline value.

### Why Couldn’t You Estimate A Causal Effect?

Users were not randomized to the proposed personalization strategy. Random exposure helps discuss exposure bias, but it is not a clean user-level experiment for retention.

## Mock Interview Questions

1. What is the unit of analysis in each part of the project?
2. How did you define churn or disengagement?
3. What makes the churn label imperfect?
4. Why did you use a temporal feature-label split?
5. What features were most predictive?
6. Why was PR-AUC more useful than accuracy?
7. How would a PM use the top-risk threshold results?
8. What is the difference between logistic regression coefficients and permutation importance?
9. How does SVD matrix factorization work conceptually?
10. Why did personalization work better for high-activity users?
11. Why are offline recommendation metrics exposure-biased?
12. What A/B test would you run next?
13. What are the null and alternative hypotheses?
14. How did you think about sample size?
15. What causal claim can you make from this project?

## Short Answers

1. Raw logs are event-level, churn modeling is user-level, and recommendation modeling is user-item-level.
2. Disengagement means a user observed in the early standard period was not observed in the late standard period.
3. It is observed non-return, not true account deletion or full app churn.
4. To make sure model inputs are measured before the outcome.
5. Activity trend, total events, unique videos/authors, recency, events per active day, and tag diversity.
6. The positive class is rare, so accuracy rewards majority-class predictions.
7. They could target a limited intervention to the highest-risk users while monitoring cost and guardrails.
8. Coefficients explain a linear model’s conditional associations; permutation importance measures how much model performance drops when a feature is shuffled.
9. SVD learns lower-dimensional user and item vectors from a sparse user-item interaction matrix.
10. High-activity users provide more history for collaborative filtering to learn from.
11. Future positives are only observed for videos the production system exposed.
12. A user-level A/B test comparing current recommendations to segment-aware personalization.
13. Null: treatment return rate equals control return rate. Alternative: they differ.
14. I used a two-proportion sample-size approximation based on baseline return rate, alpha, power, and MDE.
15. No causal retention effect was estimated. The project makes descriptive and predictive claims, plus an experiment design.
