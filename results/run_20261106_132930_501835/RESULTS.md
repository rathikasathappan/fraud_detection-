# Measured Model Comparison Results

Run folder: `run_20261106_132930_501835`.
Main comparison seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51.
Held-out MAML seeds: 42, 43, 44, 45, 46; categories: Travel, Fashion.
Each main split uses disjoint ring-grouped train, validation, and test partitions. All main models use the same split, select a threshold by maximizing validation F1, and are evaluated only on test rows. Graph features are recomputed per seed using training labels only.

## Main Comparison

| Model | Test group | PR-AUC mean +/- SD | Precision | Recall | F1 | Fraud cases mean +/- SD |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | all_test | 0.6098 +/- 0.0626 | 0.6540 +/- 0.1110 | 0.5364 +/- 0.0788 | 0.5842 +/- 0.0717 | 49.40 +/- 8.87 |
| logistic_regression | cold_start | 0.6756 +/- 0.0607 | 0.6401 +/- 0.1112 | 0.6048 +/- 0.0910 | 0.6156 +/- 0.0756 | 35.60 +/- 6.75 |
| xgboost_graph | all_test | 0.6893 +/- 0.0568 | 0.8477 +/- 0.0732 | 0.6104 +/- 0.0725 | 0.7046 +/- 0.0371 | 49.40 +/- 8.87 |
| xgboost_graph | cold_start | 0.7618 +/- 0.0599 | 0.8402 +/- 0.0725 | 0.7060 +/- 0.0699 | 0.7627 +/- 0.0328 | 35.60 +/- 6.75 |
| xgboost_tabular | all_test | 0.6729 +/- 0.0476 | 0.8271 +/- 0.0633 | 0.6191 +/- 0.0592 | 0.7042 +/- 0.0252 | 49.40 +/- 8.87 |
| xgboost_tabular | cold_start | 0.7364 +/- 0.0429 | 0.8179 +/- 0.0619 | 0.7118 +/- 0.0703 | 0.7570 +/- 0.0333 | 35.60 +/- 6.75 |
| xgboost_tabular_no_business_age | all_test | 0.4692 +/- 0.0610 | 0.5561 +/- 0.1727 | 0.4098 +/- 0.0624 | 0.4556 +/- 0.0776 | 49.40 +/- 8.87 |
| xgboost_tabular_no_business_age | cold_start | 0.4557 +/- 0.0706 | 0.5002 +/- 0.1724 | 0.4205 +/- 0.0686 | 0.4385 +/- 0.0802 | 35.60 +/- 6.75 |

## Paired Differences

| Comparison | Test group | PR-AUC difference mean +/- SD | Positive seeds | Seeds |
| --- | --- | --- | --- | --- |
| graph_minus_tabular | all_test | 0.0163 +/- 0.0262 | 7 | 10.0 |
| graph_minus_tabular | cold_start | 0.0254 +/- 0.0294 | 9 | 10.0 |
| tabular_with_age_minus_without | all_test | 0.2038 +/- 0.0443 | 10 | 10.0 |
| tabular_with_age_minus_without | cold_start | 0.2807 +/- 0.0669 | 10 | 10.0 |

- graph_minus_tabular on all_test: no clear difference (mean paired difference 0.0163, SD 0.0262).
- graph_minus_tabular on cold_start: no clear difference (mean paired difference 0.0254, SD 0.0294).
- tabular_with_age_minus_without on all_test: business age improves mean PR-AUC (mean paired difference 0.2038, SD 0.0443).
- tabular_with_age_minus_without on cold_start: business age improves mean PR-AUC (mean paired difference 0.2807, SD 0.0669).

## Ring-Link Check

| Test ring-member occurrences | Any training entity link | Share | Training fraud link | Share | Per-seed fraud-link share mean +/- SD |
| --- | --- | --- | --- | --- | --- |
| 412 | 74 | 17.96% | 15 | 3.64% | 0.0365 +/- 0.0180 |
The share of test ring members with a training fraud link was 3.64%. 

## MAML Fairness

The standard tabular XGBoost baseline trained on the full training split, so it did see Travel merchants. The `xgboost_zero_shot` baseline and MAML meta-training exclude the held-out category; both receive only the listed training support examples for category adaptation. The support-fine-tuned XGBoost and MAML runs use the same sampled support rows. Query rows are disjoint from support rows. Zero-shot thresholds are chosen on validation rows from seen categories; fine-tuned XGBoost and MAML thresholds are chosen on support rows.

| Category | Model | Shots | Query group | PR-AUC mean +/- SD | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Fashion | maml | 5 | cold_start_query | 0.5311 +/- 0.1797 | 0.3486 +/- 0.1923 | 0.3307 +/- 0.2180 |
| Fashion | maml | 5 | held_out_category_query | 0.4051 +/- 0.1632 | 0.2661 +/- 0.1628 | 0.2376 +/- 0.1982 |
| Fashion | maml | 10 | cold_start_query | 0.5421 +/- 0.1607 | 0.2986 +/- 0.3526 | 0.3167 +/- 0.3354 |
| Fashion | maml | 10 | held_out_category_query | 0.4136 +/- 0.1546 | 0.2150 +/- 0.2643 | 0.2455 +/- 0.2643 |
| Fashion | maml | 20 | cold_start_query | 0.5746 +/- 0.1202 | 0.5943 +/- 0.1546 | 0.4131 +/- 0.1879 |
| Fashion | maml | 20 | held_out_category_query | 0.4172 +/- 0.1425 | 0.4409 +/- 0.1295 | 0.3098 +/- 0.1376 |
| Fashion | xgboost_seen_category | 0 | all_test | 0.5148 +/- 0.0532 | 0.4527 +/- 0.0660 | 0.5842 +/- 0.0839 |
| Fashion | xgboost_seen_category | 0 | cold_start | 0.7016 +/- 0.1360 | 0.6279 +/- 0.1819 | 0.7034 +/- 0.1677 |
| Fashion | xgboost_support_finetuned | 5 | cold_start_query | 0.7186 +/- 0.1641 | 0.4500 +/- 0.5123 | 0.1298 +/- 0.1657 |
| Fashion | xgboost_support_finetuned | 5 | held_out_category_query | 0.5293 +/- 0.0746 | 0.3900 +/- 0.4588 | 0.1088 +/- 0.1394 |
| Fashion | xgboost_support_finetuned | 10 | cold_start_query | 0.7186 +/- 0.1641 | 0.5100 +/- 0.4037 | 0.4131 +/- 0.3411 |
| Fashion | xgboost_support_finetuned | 10 | held_out_category_query | 0.5293 +/- 0.0746 | 0.3703 +/- 0.3102 | 0.3272 +/- 0.2522 |
| Fashion | xgboost_support_finetuned | 20 | cold_start_query | 0.7205 +/- 0.1613 | 0.5307 +/- 0.1487 | 0.5567 +/- 0.1862 |
| Fashion | xgboost_support_finetuned | 20 | held_out_category_query | 0.5307 +/- 0.0719 | 0.3877 +/- 0.0728 | 0.4605 +/- 0.1228 |
| Fashion | xgboost_zero_shot | 0 | cold_start_query | 0.7186 +/- 0.1641 | 0.6279 +/- 0.1819 | 0.7034 +/- 0.1677 |
| Fashion | xgboost_zero_shot | 0 | held_out_category_query | 0.5293 +/- 0.0746 | 0.4527 +/- 0.0660 | 0.5842 +/- 0.0839 |
| Travel | maml | 5 | cold_start_query | 0.7784 +/- 0.0601 | 0.1709 +/- 0.2348 | 0.2254 +/- 0.3087 |
| Travel | maml | 5 | held_out_category_query | 0.6982 +/- 0.1372 | 0.1945 +/- 0.2214 | 0.2686 +/- 0.2883 |
| Travel | maml | 10 | cold_start_query | 0.7841 +/- 0.0869 | 0.5606 +/- 0.3633 | 0.4940 +/- 0.2846 |
| Travel | maml | 10 | held_out_category_query | 0.7200 +/- 0.1723 | 0.5707 +/- 0.3610 | 0.4919 +/- 0.3107 |
| Travel | maml | 20 | cold_start_query | 0.7798 +/- 0.0663 | 0.4376 +/- 0.2462 | 0.4943 +/- 0.1218 |
| Travel | maml | 20 | held_out_category_query | 0.7019 +/- 0.1390 | 0.4433 +/- 0.2720 | 0.5023 +/- 0.1431 |
| Travel | xgboost_seen_category | 0 | all_test | 0.8223 +/- 0.0637 | 0.7637 +/- 0.1034 | 0.8321 +/- 0.1039 |
| Travel | xgboost_seen_category | 0 | cold_start | 0.8812 +/- 0.0805 | 0.8206 +/- 0.1293 | 0.8608 +/- 0.1075 |
| Travel | xgboost_support_finetuned | 5 | cold_start_query | 0.9216 +/- 0.0764 | 0.6200 +/- 0.4147 | 0.5696 +/- 0.3777 |
| Travel | xgboost_support_finetuned | 5 | held_out_category_query | 0.8693 +/- 0.0899 | 0.6102 +/- 0.3048 | 0.5880 +/- 0.2398 |
| Travel | xgboost_support_finetuned | 10 | cold_start_query | 0.9216 +/- 0.0764 | 0.6273 +/- 0.4146 | 0.6625 +/- 0.3934 |
| Travel | xgboost_support_finetuned | 10 | held_out_category_query | 0.8693 +/- 0.0899 | 0.6132 +/- 0.3047 | 0.6915 +/- 0.2248 |
| Travel | xgboost_support_finetuned | 20 | cold_start_query | 0.9296 +/- 0.0651 | 0.7473 +/- 0.2360 | 0.6440 +/- 0.1418 |
| Travel | xgboost_support_finetuned | 20 | held_out_category_query | 0.8796 +/- 0.0893 | 0.6888 +/- 0.2181 | 0.6245 +/- 0.1145 |
| Travel | xgboost_zero_shot | 0 | cold_start_query | 0.9216 +/- 0.0764 | 0.7539 +/- 0.1919 | 0.8485 +/- 0.1288 |
| Travel | xgboost_zero_shot | 0 | held_out_category_query | 0.8693 +/- 0.0899 | 0.7066 +/- 0.1846 | 0.8162 +/- 0.1377 |

## Business-Age Ablation

On all test merchants, the paired PR-AUC change from including business age was 0.2038 (SD 0.0443); this indicates material sensitivity to business age, but does not establish that it is the only fraud signal.

## Feature Importance

Importance is averaged across main-comparison seeds (normalized absolute coefficients for Logistic Regression; XGBoost gain-based `feature_importances_` for tree models).

| Model | Feature | Mean importance | SD |
| --- | --- | --- | --- |
| logistic_regression | num__business_age_days | 0.245080 | 0.010576 |
| logistic_regression | num__transaction_count | 0.162591 | 0.029254 |
| logistic_regression | cat__business_category_Fashion | 0.127129 | 0.062986 |
| logistic_regression | cat__business_category_Electronics | 0.097077 | 0.040856 |
| logistic_regression | num__document_verified | 0.083809 | 0.012580 |
| xgboost_graph | num__louvain_community_fraud_rate | 0.218197 | 0.117081 |
| xgboost_graph | num__business_age_days | 0.152059 | 0.039625 |
| xgboost_graph | num__neighbours_flagged_fraud | 0.117843 | 0.061134 |
| xgboost_graph | num__transaction_count | 0.066163 | 0.014320 |
| xgboost_graph | cat__business_category_Fashion | 0.055896 | 0.025733 |
| xgboost_tabular | num__business_age_days | 0.246325 | 0.024455 |
| xgboost_tabular | num__document_verified | 0.132388 | 0.030771 |
| xgboost_tabular | num__transaction_count | 0.129180 | 0.015894 |
| xgboost_tabular | cat__business_category_Fashion | 0.092358 | 0.023313 |
| xgboost_tabular | num__phone_verified | 0.091783 | 0.021641 |
| xgboost_tabular_no_business_age | num__document_verified | 0.212993 | 0.028238 |
| xgboost_tabular_no_business_age | num__transaction_count | 0.189624 | 0.032262 |
| xgboost_tabular_no_business_age | num__phone_verified | 0.163402 | 0.042584 |
| xgboost_tabular_no_business_age | num__email_verified | 0.114045 | 0.019839 |
| xgboost_tabular_no_business_age | cat__business_category_Grocery | 0.103014 | 0.009698 |

The dashboard CSV was refreshed from this run: `data\powerbi_exports\model_metrics.csv`.
