# FraudGuard — Power BI Integration Guide

## Available Export Endpoints

All endpoints require an authenticated session (session cookie) and return UTF-8 CSV.
Export events are recorded in the Audit Log.

| Endpoint | Roles | Rows | Description |
|----------|-------|------|-------------|
| `GET /api/export/merchants.csv` | Admin, Investigator | ≤50 000 | Merchant master + latest risk assessment |
| `GET /api/export/alerts.csv` | Admin, Investigator | ≤50 000 | Alert queue with linked investigation cases |
| `GET /api/export/decisions.csv` | Admin, Investigator, Analyst | ≤50 000 | Full risk decision history |

> **Note:** All exports exclude `data_source = 'test'` rows.

---

## Connecting from Power BI Desktop

1. Open Power BI Desktop → **Get Data** → **Web**
2. Enter: `http://your-server:5000/api/export/merchants.csv`
3. Under **Advanced**, add HTTP header:
   - Name: `Cookie`
   - Value: `session=<your-session-cookie>`
4. Click **OK** → Power BI parses the CSV automatically

---

## Column Definitions

### merchants.csv

| Column | Type | Notes |
|--------|------|-------|
| `merchant_id` | String | Unique identifier |
| `business_category` | String | Electronics, Digital Services, Fashion, Grocery, Travel |
| `business_age_days` | Integer | NULL = not provided |
| `registration_date` | Datetime | ISO format |
| `phone_verified` | Boolean | 1=Yes, 0=No, NULL=not provided |
| `email_verified` | Boolean | 1=Yes, 0=No, NULL=not provided |
| `document_verified` | Boolean | 1=Yes, 0=No, NULL=not provided |
| `transaction_count` | Integer | 0 = cold-start |
| `fraud_label` | Integer | 1=Known fraud, 0=Legitimate, NULL=unlabelled |
| `data_source` | Enum | synthetic, company, api |
| `risk_score` | Float | 0.0–1.0, NULL if not assessed |
| `decision` | Enum | Approve, Review, Hold, NULL |
| `evaluated_at` | Datetime | Latest assessment timestamp |

### alerts.csv

| Column | Type | Notes |
|--------|------|-------|
| `alert_id` | String | Unique alert identifier |
| `merchant_id` | String | FK to merchants |
| `severity` | Enum | LOW, MEDIUM, HIGH, CRITICAL |
| `alert_type` | String | COLD_START_FRAUD_DETECTED, HIGH_RISK_ONBOARDING |
| `status` | Enum | OPEN, ACKNOWLEDGED, RESOLVED |
| `details` | String | Human-readable reason summary |
| `created_at` | Datetime | Alert creation time |
| `case_id` | String | Linked investigation case ID (may be empty) |
| `case_status` | Enum | Open, In review, Closed |
| `case_decision` | Enum | Confirm fraud, False positive, Needs more info |

### decisions.csv

| Column | Type | Notes |
|--------|------|-------|
| `assessment_id` | String | Unique assessment identifier |
| `merchant_id` | String | FK to merchants |
| `risk_score` | Float | Model output 0.0–1.0 |
| `decision` | Enum | Approve, Review, Hold |
| `evaluated_at` | Datetime | Assessment timestamp |
| `reasons` | JSON | Array of risk reason strings |
| `data_completeness` | String | e.g. "75.0%" |
| `business_category` | String | From merchants table |
| `data_source` | Enum | synthetic, company, api |

---

## Recommended Power BI Measures

```dax
// Fraud rate
Fraud Rate =
  DIVIDE(
    COUNTROWS(FILTER('decisions', 'decisions'[decision] = "Hold")),
    COUNTROWS('decisions')
  )

// Cold-start share
Cold Start % =
  DIVIDE(
    COUNTROWS(FILTER('merchants', 'merchants'[transaction_count] = 0)),
    COUNTROWS('merchants')
  )

// Average risk score
Avg Risk Score = AVERAGE('decisions'[risk_score])
```

