# User Disengagement and Personalization in a Short-Video Feed

This project analyzes short-video recommendation behavior to understand what predicts user disengagement and how personalization could improve the user experience.

## Project Overview

Consumer products such as short-video feeds depend on keeping users engaged with relevant content. This project uses the public KuaiRand-Pure dataset, collected from Kuaishou recommendation logs, to study user behavior, retention, churn risk, recommendation quality, and experiment design.

The project is being built in phases so that each analytical decision is documented and defensible in a Data Science interview.

## Business Problem

The central product question is:

> What behaviors predict user disengagement, which users are most at risk, and how can personalization improve their experience?

The eventual analysis will distinguish between:

- descriptive findings: what happened in the observed logs
- predictive findings: which behaviors help predict future disengagement
- causal claims: only made if the data supports them
- product recommendations: hypotheses that should be tested before rollout

## Dataset

Dataset: [KuaiRand-Pure](https://kuairand.com/)

KuaiRand-Pure is a public short-video recommendation dataset derived from Kuaishou app logs. It contains user-video interactions with timestamps, feedback signals, user features, video features, standard recommendation logs, and random exposure logs.

In this project:

- Product: short-video recommendation feed
- User: anonymized app viewer
- Item: short video
- Event: user exposure or interaction with a video
- Engagement: click, long view, like, follow, comment, forward, profile entry, comment stay time, and watch time
- Disengagement/churn proxy: reduced or absent future observed activity, not confirmed account deletion

## Current Status

All phases are complete.

Phase 1 established the data shape and initial product metrics:

- Standard early log: 1,141,112 interactions, 26,210 users, 7,538 videos
- Standard late log: 295,497 interactions, 25,877 users, 6,618 videos
- Random late log: 1,186,059 interactions, 27,285 users, 7,583 videos
- Standard recommendation click rate is much higher than random exposure click rate
- User activity is highly skewed, so medians and percentiles are more informative than averages alone
- The observed early-to-late return rate among standard-log users is about 95.4%, which suggests the dataset is not representative of all app users

Phase 2 added cohort and retention analysis:

- Users in the lowest early-activity quartile had an observed late-period return rate of 85.8%
- Users in the highest early-activity quartile had an observed late-period return rate of 99.8%
- First-observed-date cohorts varied meaningfully, but later first-seen cohorts are smaller and more sensitive to censoring
- These results are descriptive: high early activity is associated with later return, but this does not prove that increasing activity would cause retention

Phase 3 created a user-level behavioral modeling table:

- One row per early-period standard-log user
- Features built only from early-period behavior
- Future label based on whether the user appears again in the late standard period
- Late-period disengagement rate: 4.6%
- Key feature families: frequency, recency, engagement rates, watch intensity, content diversity, repeat behavior, and short-term activity change

Phase 4 trained churn/disengagement models:

- Baseline: predicts no users disengage
- Logistic regression: interpretable probability model
- Random forest: nonlinear ensemble model
- Histogram gradient boosting: boosted tree model
- Best ROC-AUC was about 0.867 from random forest
- Best PR-AUC was about 0.233 from random forest, compared with a baseline prevalence of 0.046
- At the riskiest 10% of users, random forest identified about 50.7% of disengaged users with 23.2% precision

Phase 5 interpreted the churn models:

- Important predictive signals included second-half activity, total events, unique videos/authors, events per active day, recency, and tag diversity
- Logistic regression coefficients were useful but required caution because correlated activity features can make coefficients look counterintuitive
- Random forest permutation importance was used to measure how much PR-AUC dropped when individual features were shuffled
- Model findings were translated into product hypotheses, not causal claims

Phase 6 built recommendation models:

- Popularity baseline: recommends globally popular engaged videos
- Matrix factorization with SVD: learns low-dimensional user and video representations from implicit feedback
- Temporal evaluation: train on early standard-log positives and test on late standard-log positives
- SVD outperformed popularity on NDCG@10: 0.043 vs 0.037
- SVD improved Hit Rate@10 from 17.6% to 22.7%
- Personalization gains were strongest for higher-activity users, while low-activity users remained harder to serve

Phase 7 connected personalization to disengagement:

- Users who fully disengaged had no late positive interactions, so offline recommendation quality is not directly measurable for them
- Among evaluable returning users, SVD NDCG@10 lift over popularity was strongest for high and very-high activity users
- Low-activity users had weak or negative NDCG lift, highlighting a cold-start/personalization challenge
- These results support segment-specific personalization hypotheses, not causal retention claims

Phase 8 designed an A/B test:

- Control: current production recommendation experience
- Treatment: personalized feed strategy informed by collaborative filtering and disengagement-risk segments
- Unit of randomization: user
- Primary metric: 7-day observed return rate
- Guardrails: negative feedback, short-view behavior, exits if available, and content concentration
- Sample-size planning showed that detecting a 1 percentage-point retention lift would require about 6,140 users per group
- No experiment results were fabricated because the dataset does not contain our proposed randomized treatment

Phase 9 assessed causal inference:

- The dataset does not support estimating the causal effect of our proposed personalization strategy on retention
- Random exposure helps reason about exposure bias and engagement under less algorithmic selection
- Standard-vs-random exposure is not a clean user-level randomized experiment for retention
- Declining activity and model risk signals remain predictive associations, not causal drivers
- The correct next step for causal evidence is the Phase 8 A/B test

Phase 10 synthesized the product recommendation:

- Test segment-aware personalization rather than immediately shipping it as a retention fix
- Use collaborative filtering for users with enough interaction history
- Use exploration, popularity, or content-based approaches for low-history users
- Target users with elevated disengagement risk and sufficient history for the first experiment
- Judge success with retention, engagement quality, and guardrail metrics

Phase 11 prepared portfolio and interview materials:

- final product report
- resume bullets
- 60-second explanation
- 3-minute technical explanation
- mock interview questions and answer guide
- consolidated interview-prep study document

## Methodology Roadmap

Planned phases:

1. Data understanding and product metrics
2. Cohort and retention analysis
3. Behavioral feature engineering
4. Churn/disengagement modeling
5. Model interpretation and product insights
6. Recommendation and personalization
7. Connecting personalization to disengagement
8. Experiment design
9. Causal inference, only if appropriate
10. Final product recommendation
11. Portfolio and interview preparation

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_retention_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_churn_modeling.ipynb
│   ├── 05_model_interpretation.ipynb
│   ├── 06_recommendation_system.ipynb
│   ├── 07_personalization_disengagement.ipynb
│   ├── 08_experiment_design.ipynb
│   ├── 09_causal_assessment.ipynb
│   ├── 10_final_recommendation.ipynb
│   └── 11_portfolio_interview_prep.ipynb
├── docs/
│   ├── interview_prep_so_far.md
│   ├── limitations.md
│   ├── methodology.md
│   ├── metrics.md
│   ├── modeling_decisions.md
│   └── resume_and_interview_assets.md
├── reports/
│   ├── final_report.md
│   └── figures/
└── src/
    ├── causal/
    │   └── phase9_causal_assessment.py
    ├── data/
    │   ├── phase1_data_understanding.py
    │   └── phase2_retention_analysis.py
    ├── experiments/
    │   └── phase8_experiment_design.py
    ├── features/
    │   └── phase3_behavioral_features.py
    └── models/
        ├── phase4_churn_modeling.py
        ├── phase5_model_interpretation.py
        ├── phase6_recommendation_system.py
        └── phase7_personalization_disengagement.py
```

## Reproducing Phase 1

Install dependencies:

```bash
pip install -r requirements.txt
```

Download KuaiRand-Pure from the official dataset page and extract it so the files are under:

```text
data/raw/KuaiRand-Pure/data/
```

Then run:

```bash
python src/data/phase1_data_understanding.py
```

The script generates summary tables in `reports/` and figures in `reports/figures/`.

Run Phase 2 cohort analysis:

```bash
python src/data/phase2_retention_analysis.py
```

Run Phase 3 feature engineering:

```bash
python src/features/phase3_behavioral_features.py
```

Run Phase 4 churn/disengagement modeling:

```bash
python src/models/phase4_churn_modeling.py
```

Run Phase 5 model interpretation:

```bash
python src/models/phase5_model_interpretation.py
```

Run Phase 6 recommendation modeling:

```bash
python src/models/phase6_recommendation_system.py
```

Run Phase 7 personalization-disengagement analysis:

```bash
python src/models/phase7_personalization_disengagement.py
```

Run Phase 8 experiment design:

```bash
python src/experiments/phase8_experiment_design.py
```

Run Phase 9 causal assessment:

```bash
python src/causal/phase9_causal_assessment.py
```

Phase 10 and Phase 11 are written deliverables:

- `reports/final_report.md`
- `docs/resume_and_interview_assets.md`
- `docs/interview_prep_so_far.md`

## Key Findings

- Early activity is strongly associated with later observed return.
- Behavioral churn models can identify elevated-risk users substantially better than a naive baseline.
- Random forest improved PR-AUC from a 0.046 prevalence baseline to 0.233.
- SVD matrix factorization improved recommendation Hit Rate@10 from 17.6% to 22.7%.
- Personalization gains were strongest for higher-activity users and weaker for low-history users.
- No causal retention effect was estimated from the historical dataset.

## Product Recommendation

Test segment-aware personalization in a user-level A/B experiment. Use collaborative filtering for users with enough interaction history, and use cold-start strategies such as popularity, exploration, or content-based recommendations for low-history users.

The launch decision should depend on measured retention lift, engagement quality, and guardrail metrics. Offline model gains should not be treated as proven retention impact.

## Important Limitations

This project uses observed behavioral logs, not complete user lifetime data. Unless later analysis supports stronger claims, "churn" means observed disengagement inside the dataset window, not confirmed permanent churn.

Random exposure logs help us reason about exposure bias, but offline behavioral data alone does not prove that a personalization intervention will improve retention in production.
