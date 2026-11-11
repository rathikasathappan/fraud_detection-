# Measured Model Comparison Results

Run folder: `run_20261106_135425_287449`.
Main comparison seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51.
Held-out MAML seeds: 42, 43, 44, 45, 46; categories: Travel, Fashion.
Each main split uses disjoint ring-grouped train, validation, and test partitions. All main models use the same split, select a threshold by maximizing validation F1, and are evaluated only on test rows. Graph features are recomputed per seed using training labels only.

## Main Comparison

| Model | Test group | PR-AUC mean +/- SD | Precision | Recall | F1 | Fraud cases mean +/- SD |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | all_test | 0.4475 +/- 0.1388 | 0.5804 +/- 0.2077 | 0.3909 +/- 0.1174 | 0.4549 +/- 0.1354 | 47.80 +/- 10.32 |
| logistic_regression | cold_start | 0.4619 +/- 0.1635 | 0.5303 +/- 0.2198 | 0.4399 +/- 0.1461 | 0.4663 +/- 0.1653 | 29.50 +/- 7.15 |
| xgboost_graph | all_test | 0.6497 +/- 0.0923 | 0.8594 +/- 0.1034 | 0.5871 +/- 0.1055 | 0.6942 +/- 0.0966 | 47.80 +/- 10.32 |
| xgboost_graph | cold_start | 0.6746 +/- 0.1546 | 0.8476 +/- 0.1525 | 0.6158 +/- 0.1525 | 0.7074 +/- 0.1419 | 29.50 +/- 7.15 |
| xgboost_tabular | all_test | 0.5029 +/- 0.1198 | 0.5929 +/- 0.1594 | 0.4694 +/- 0.1001 | 0.5122 +/- 0.1021 | 47.80 +/- 10.32 |
| xgboost_tabular | cold_start | 0.4409 +/- 0.1436 | 0.4949 +/- 0.1893 | 0.4485 +/- 0.1169 | 0.4546 +/- 0.1284 | 29.50 +/- 7.15 |
| xgboost_tabular_no_business_age | all_test | 0.5035 +/- 0.1183 | 0.6502 +/- 0.1725 | 0.4564 +/- 0.0851 | 0.5334 +/- 0.1116 | 47.80 +/- 10.32 |
| xgboost_tabular_no_business_age | cold_start | 0.4193 +/- 0.1549 | 0.5700 +/- 0.2105 | 0.4144 +/- 0.1018 | 0.4733 +/- 0.1365 | 29.50 +/- 7.15 |

## Paired Differences

| Comparison | Test group | PR-AUC difference mean +/- SD | Positive seeds | Seeds |
| --- | --- | --- | --- | --- |
| graph_minus_tabular | all_test | 0.1468 +/- 0.0386 | 10 | 10.0 |
| graph_minus_tabular | cold_start | 0.2337 +/- 0.0682 | 10 | 10.0 |
| tabular_with_age_minus_without | all_test | -0.0007 +/- 0.0128 | 5 | 10.0 |
| tabular_with_age_minus_without | cold_start | 0.0216 +/- 0.0348 | 8 | 10.0 |

- graph_minus_tabular on all_test: graph XGBoost has the higher mean PR-AUC (mean paired difference 0.1468, SD 0.0386).
- graph_minus_tabular on cold_start: graph XGBoost has the higher mean PR-AUC (mean paired difference 0.2337, SD 0.0682).
- tabular_with_age_minus_without on all_test: no clear difference (mean paired difference -0.0007, SD 0.0128).
- tabular_with_age_minus_without on cold_start: no clear difference (mean paired difference 0.0216, SD 0.0348).

## Ring-Link Check

| Test ring-member occurrences | Any training entity link | Share | Training fraud link | Share | Per-seed fraud-link share mean +/- SD |
| --- | --- | --- | --- | --- | --- |
| 412 | 262 | 63.59% | 251 | 60.92% | 0.6056 +/- 0.1152 |
The share of test ring members with a training fraud link was 60.92%. 

## MAML Fairness

The standard tabular XGBoost baseline trained on the full training split, so it did see Travel merchants. The `xgboost_zero_shot` baseline and MAML meta-training exclude the held-out category; both receive only the listed training support examples for category adaptation. The support-fine-tuned XGBoost and MAML runs use the same sampled support rows. Query rows are disjoint from support rows. Zero-shot thresholds are chosen on validation rows from seen categories; fine-tuned XGBoost and MAML thresholds are chosen on support rows.

| Category | Model | Shots | Query group | PR-AUC mean +/- SD | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Fashion | maml | 5 | cold_start_query | 0.5883 +/- 0.2218 | 0.3538 +/- 0.1939 | 0.3169 +/- 0.2241 |
| Fashion | maml | 5 | held_out_category_query | 0.5511 +/- 0.2050 | 0.3767 +/- 0.2732 | 0.3015 +/- 0.2479 |
| Fashion | maml | 10 | cold_start_query | 0.6174 +/- 0.1506 | 0.3105 +/- 0.3179 | 0.2022 +/- 0.2126 |
| Fashion | maml | 10 | held_out_category_query | 0.5699 +/- 0.1573 | 0.3033 +/- 0.3189 | 0.1958 +/- 0.2208 |
| Fashion | maml | 20 | cold_start_query | 0.6168 +/- 0.1513 | 0.5795 +/- 0.2375 | 0.4457 +/- 0.2400 |
| Fashion | maml | 20 | held_out_category_query | 0.5619 +/- 0.1697 | 0.5450 +/- 0.2134 | 0.4285 +/- 0.2453 |
| Fashion | xgboost_seen_category | 0 | cold_start_query | 0.2402 +/- 0.0997 | 0.1567 +/- 0.1507 | 0.2032 +/- 0.1965 |
| Fashion | xgboost_seen_category | 0 | held_out_category_query | 0.2827 +/- 0.1390 | 0.1667 +/- 0.1667 | 0.2222 +/- 0.2222 |
| Fashion | xgboost_support_finetuned | 5 | cold_start_query | 0.2249 +/- 0.0734 | 0.4405 +/- 0.3932 | 0.1795 +/- 0.1778 |
| Fashion | xgboost_support_finetuned | 5 | held_out_category_query | 0.2106 +/- 0.0766 | 0.4433 +/- 0.2753 | 0.2057 +/- 0.1571 |
| Fashion | xgboost_support_finetuned | 10 | cold_start_query | 0.2277 +/- 0.0701 | 0.1500 +/- 0.1491 | 0.1816 +/- 0.1776 |
| Fashion | xgboost_support_finetuned | 10 | held_out_category_query | 0.2078 +/- 0.0830 | 0.2133 +/- 0.1574 | 0.2380 +/- 0.1729 |
| Fashion | xgboost_support_finetuned | 20 | cold_start_query | 0.2620 +/- 0.1120 | 0.5281 +/- 0.2879 | 0.1988 +/- 0.0800 |
| Fashion | xgboost_support_finetuned | 20 | held_out_category_query | 0.2383 +/- 0.1056 | 0.5033 +/- 0.2445 | 0.1752 +/- 0.0836 |
| Fashion | xgboost_zero_shot | 0 | cold_start_query | 0.2249 +/- 0.0734 | 0.5229 +/- 0.1650 | 0.3018 +/- 0.0968 |
| Fashion | xgboost_zero_shot | 0 | held_out_category_query | 0.2106 +/- 0.0766 | 0.5117 +/- 0.1148 | 0.2506 +/- 0.1148 |
| Travel | maml | 5 | cold_start_query | 0.3611 +/- 0.4158 | 0.1000 +/- 0.2236 | 0.1333 +/- 0.2982 |
| Travel | maml | 5 | held_out_category_query | 0.4565 +/- 0.2482 | 0.2333 +/- 0.2236 | 0.3333 +/- 0.3118 |
| Travel | maml | 10 | cold_start_query | 0.3611 +/- 0.4158 | 0.1000 +/- 0.2236 | 0.1333 +/- 0.2982 |
| Travel | maml | 10 | held_out_category_query | 0.4444 +/- 0.2202 | 0.2333 +/- 0.2236 | 0.2833 +/- 0.2982 |
| Travel | maml | 20 | cold_start_query | 0.3807 +/- 0.4035 | 0.1000 +/- 0.2236 | 0.1000 +/- 0.2236 |
| Travel | maml | 20 | held_out_category_query | 0.4432 +/- 0.2262 | 0.4200 +/- 0.2663 | 0.2616 +/- 0.3010 |
| Travel | xgboost_seen_category | 0 | cold_start_query | 0.2764 +/- 0.3212 | 0.1000 +/- 0.2236 | 0.1333 +/- 0.2982 |
| Travel | xgboost_seen_category | 0 | held_out_category_query | 0.5276 +/- 0.2832 | 0.3533 +/- 0.2280 | 0.4467 +/- 0.2745 |
| Travel | xgboost_support_finetuned | 5 | cold_start_query | 0.3214 +/- 0.4437 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Travel | xgboost_support_finetuned | 5 | held_out_category_query | 0.4855 +/- 0.2983 | 0.3867 +/- 0.4253 | 0.2864 +/- 0.3300 |
| Travel | xgboost_support_finetuned | 10 | cold_start_query | 0.3214 +/- 0.4437 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Travel | xgboost_support_finetuned | 10 | held_out_category_query | 0.4855 +/- 0.2983 | 0.2367 +/- 0.2518 | 0.3300 +/- 0.3271 |
| Travel | xgboost_support_finetuned | 20 | cold_start_query | 0.2423 +/- 0.3098 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Travel | xgboost_support_finetuned | 20 | held_out_category_query | 0.4393 +/- 0.2191 | 0.1733 +/- 0.1673 | 0.1833 +/- 0.1708 |
| Travel | xgboost_zero_shot | 0 | cold_start_query | 0.3214 +/- 0.4437 | 0.1000 +/- 0.2236 | 0.1333 +/- 0.2982 |
| Travel | xgboost_zero_shot | 0 | held_out_category_query | 0.4855 +/- 0.2983 | 0.6033 +/- 0.2849 | 0.5381 +/- 0.2050 |
- Fashion, 5-shot MAML vs xgboost_zero_shot: MAML has the higher mean PR-AUC (mean gap 0.3405; combined observed SD 0.2188).
- Fashion, 5-shot MAML vs xgboost_support_finetuned: MAML has the higher mean PR-AUC (mean gap 0.3405; combined observed SD 0.2188).
- Fashion, 10-shot MAML vs xgboost_zero_shot: MAML has the higher mean PR-AUC (mean gap 0.3593; combined observed SD 0.1750).
- Fashion, 10-shot MAML vs xgboost_support_finetuned: MAML has the higher mean PR-AUC (mean gap 0.3621; combined observed SD 0.1778).
- Fashion, 20-shot MAML vs xgboost_zero_shot: MAML has the higher mean PR-AUC (mean gap 0.3513; combined observed SD 0.1862).
- Fashion, 20-shot MAML vs xgboost_support_finetuned: MAML has the higher mean PR-AUC (mean gap 0.3236; combined observed SD 0.1999).
- Travel, 5-shot MAML vs xgboost_zero_shot: no clear difference (mean gap -0.0290; combined observed SD 0.3880).
- Travel, 5-shot MAML vs xgboost_support_finetuned: no clear difference (mean gap -0.0290; combined observed SD 0.3880).
- Travel, 10-shot MAML vs xgboost_zero_shot: no clear difference (mean gap -0.0410; combined observed SD 0.3708).
- Travel, 10-shot MAML vs xgboost_support_finetuned: no clear difference (mean gap -0.0410; combined observed SD 0.3708).
- Travel, 20-shot MAML vs xgboost_zero_shot: no clear difference (mean gap -0.0422; combined observed SD 0.3744).
- Travel, 20-shot MAML vs xgboost_support_finetuned: no clear difference (mean gap 0.0039; combined observed SD 0.3149).

## Business-Age Ablation

On all test merchants, tabular XGBoost PR-AUC was 0.5029 with business age and 0.5035 without it. The paired change was -0.0007 (SD 0.0128); no clear difference is visible within the seed-to-seed spread.

## Feature Importance

Importance is averaged across main-comparison seeds (normalized absolute coefficients for Logistic Regression; XGBoost gain-based `feature_importances_` for tree models).

| Model | Feature | Mean importance | SD |
| --- | --- | --- | --- |
| logistic_regression | cat__business_category_Grocery | 0.241494 | 0.039577 |
| logistic_regression | cat__business_category_Fashion | 0.221215 | 0.023081 |
| logistic_regression | num__transaction_count | 0.157485 | 0.015898 |
| logistic_regression | cat__business_category_Travel | 0.091212 | 0.043619 |
| logistic_regression | cat__business_category_Electronics | 0.087634 | 0.021371 |
| xgboost_graph | num__neighbours_flagged_fraud | 0.256618 | 0.090414 |
| xgboost_graph | num__louvain_community_fraud_rate | 0.254891 | 0.040680 |
| xgboost_graph | num__email_verified | 0.052851 | 0.012536 |
| xgboost_graph | num__transaction_count | 0.052077 | 0.010918 |
| xgboost_graph | num__phone_verified | 0.049567 | 0.017152 |
| xgboost_tabular | num__document_verified | 0.221135 | 0.043886 |
| xgboost_tabular | num__transaction_count | 0.152614 | 0.016680 |
| xgboost_tabular | num__phone_verified | 0.133999 | 0.018652 |
| xgboost_tabular | num__email_verified | 0.127429 | 0.018927 |
| xgboost_tabular | cat__business_category_Grocery | 0.096863 | 0.014816 |
| xgboost_tabular_no_business_age | num__document_verified | 0.229230 | 0.052896 |
| xgboost_tabular_no_business_age | num__transaction_count | 0.189959 | 0.024889 |
| xgboost_tabular_no_business_age | num__phone_verified | 0.125302 | 0.017312 |
| xgboost_tabular_no_business_age | cat__business_category_Grocery | 0.117256 | 0.019144 |
| xgboost_tabular_no_business_age | cat__business_category_Fashion | 0.110712 | 0.022806 |

The dashboard CSV was refreshed from this run: `data\powerbi_exports\model_metrics.csv`.
