# Data

This project uses the public KuaiRand-Pure dataset.

Official dataset page: https://kuairand.com/

## Expected Local Layout

```text
data/
├── raw/
│   └── KuaiRand-Pure/
│       └── data/
│           ├── log_standard_4_08_to_4_21_pure.csv
│           ├── log_standard_4_22_to_5_08_pure.csv
│           ├── log_random_4_22_to_5_08_pure.csv
│           ├── user_features_pure.csv
│           ├── video_features_basic_pure.csv
│           └── video_features_statistic_pure.csv
└── processed/
```

Raw data is not committed to the repository. Download it from the official source and place it in the expected layout above.

## Files Used In Phase 1

- Standard early log: production recommendation interactions from April 9-April 21, 2022
- Standard late log: production recommendation interactions from April 21-May 8, 2022
- Random late log: interactions from randomized video exposure during the later period
- User features: anonymized user attributes and activity descriptors
- Video features: basic item metadata and aggregate video statistics

## Data Caveats

KuaiRand-Pure is a filtered version of KuaiRand that keeps videos from a candidate pool. It is manageable for portfolio analysis, but it is not complete user lifetime history.

The random and standard logs should not be casually combined because they come from different exposure mechanisms.
