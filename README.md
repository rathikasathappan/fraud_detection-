# Real-Time Cold-Start Merchant Fraud Detection Engine
**Using Network Graph Priors, Explainable Machine Learning, and Meta-Learning**

---

## 🌟 Executive Summary
This production system scores newly registered merchants for fraud risk **at onboarding before they process a single transaction (cold-start)**. It leverages **temporal transaction & entity graph priors** (shared bank accounts, devices, and IP subnets) combined with gradient boosting (XGBoost) and Few-Shot Meta-Learning (MAML in PyTorch).

---

## 📁 Repository Structure
```text
merchant_fraud_detection/
├── data/
│   ├── synthetic/
│   │   ├── merchants.csv            # 5,000 synthetic onboarding merchants (~4.96% fraud)
│   │   ├── relationships.csv        # 15,000 entity link edges (Bank, Device, IP)
│   │   └── rings_metadata.json      # Ground-truth fraud ring tracking
│   ├── processed/
│   │   ├── train_tabular.csv        # Time + ring-isolated train set
│   │   ├── test_tabular.csv         # Time + ring-isolated test set
│   │   ├── merchants_with_graph.csv # Dataset enriched with leakage-free graph priors
│   │   ├── baseline_metrics.json    # Phase 2 metrics (Logistic Regression & XGBoost)
│   │   ├── graph_model_metrics.json # Phase 3 graph-enhanced XGBoost metrics
│   │   ├── model_comparison.json   # Full comparative benchmarking table
│   │   └── meta_learning_results.json# Phase 7 MAML few-shot adaptation results
│   ├── powerbi_exports/             # Ready-to-import Power BI tables
│   │   ├── merchants_scored.csv
│   │   ├── risk_assessments.csv
│   │   ├── graph_edges.csv
│   │   ├── model_metrics.csv
│   │   ├── daily_registrations.csv
│   │   └── alerts.csv
│   └── incoming/                    # Drop directory for company batch CSV files
├── src/
│   ├── db.py                        # MySQL connection manager & CRUD helpers
│   ├── generate_data.py             # Phase 1: Realistic data generator with multi-tenant noise
│   ├── graph_features.py            # Phase 3: Temporal graph feature engineering (Zero leakage)
│   ├── train_baseline.py            # Phase 2: Tabular baseline classifiers
│   ├── train_with_graph.py          # Phase 3: XGBoost with graph priors & cold-start testing
│   ├── risk_engine.py               # Phase 4: Calibrated scoring, decision tiers & SHAP explainability
│   ├── export_for_powerbi.py        # Phase 6: Power BI export pipeline
│   ├── process_new_data.py          # Batch ingestion, validation & real-time scoring
│   ├── retrain.py                   # Model retraining & versioning archive
│   └── meta_learning/
│       └── maml.py                  # Phase 7: PyTorch MAML few-shot domain adaptation
├── database/
│   └── schema.sql                   # MySQL DDL for merchants, relationships, risk, alerts
├── notebooks/
│   └── eda.ipynb                    # Jupyter notebook for exploratory data analysis
├── tests/
│   └── test_api_and_pipeline.py     # Automated pytest suite covering API & ingestion rules
├── app.py                           # Flask REST API & Interactive Web Dashboard
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## 🖥️ Web Platform & Page Architecture

The public entry flow is `/` (landing page) → `/login` (sign in) → `/dashboard` (authenticated workspace). The landing page remains public even when a session already exists; signed-in users can choose **Go to dashboard**. The hero uses original CSS/SVG motion graphics and does not depend on a remote background video.

The web platform also features 8 dedicated operational views with responsive layout, role-based access control (Admin, Investigator, Analyst), and real-time graph reasoning:

| Page | URL Route | Allowed Roles | Core Capabilities |
| :--- | :--- | :--- | :--- |
| **Public Landing** | `/` | Public | Product overview, responsive navigation, and original animated merchant-network artwork; authenticated users are still shown this page. |
| **Sign In** | `/login` | Public | Responsive sign-in experience; carries protected-route destinations through login. |
| **1. Dashboard** | `/dashboard` | Admin, Investigator, Analyst | KPI summary cards, decision breakdown, registration trends, category risk bubble map, score distribution profile, alert heatmap, connected-entity network, and high-risk merchants table. |
| **2. Merchant Onboarding** | `/onboarding` | Admin, Investigator | Interactive merchant registration with instantaneous graph prior scoring, SHAP explainability, and 1-click test fill presets (*Sample Clean* and *Fraud Ring*). |
| **3. Risk Assessment** | `/risk` & `/risk/<id>` | Admin, Investigator, Analyst | Merchant risk breakdown, decision badges (Approve, Review, Hold), plain-language graph evidence (ordered first), and 1-click presets to known fraud rings. |
| **4. Fraud Network Explorer** | `/network` & `/network/<id>` | Admin, Investigator, Analyst | Interactive NetworkX/Vis.js visualization of shared bank accounts, hardware devices, and IP addresses with physics toggle, zoom-fit controls, and preset ring quick-switching. |
| **5. Alerts & Investigation** | `/alerts` | Admin, Investigator | Operational queue for high and critical risk alerts, investigation case modal with investigator assignment, notes, and audit-logged decisions (*Confirm fraud*, *False positive*, *Needs more info*). |
| **6. Batch Ingestion & Upload** | `/upload` | Admin | Drag-and-drop CSV ingestion, downloadable sample CSV template (`/api/sample-csv`), field validation indicators, cohort scoring, and Power BI export refreshes. |
| **7. Model Performance** | `/models` | Admin, Investigator, Analyst | Comparative benchmark tables (Baseline vs XGBoost + Graph Priors), cold-start vs all cohort metrics, PyTorch MAML few-shot adaptation on held-out sectors, and model retraining pipeline trigger. |
| **8. Audit Log & Governance** | `/audit-log` | Admin, Investigator | Chronological event audit trail tracking user actions, onboarding evaluations, and case updates, plus an Admin 1-click purge button for temporary test registrations (`data_source = 'test'`). |

---

## 🗄️ MySQL Database Setup & Environment Settings
Database connection parameters are securely loaded at runtime from `.env` in the project root using `python-dotenv`:
```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=fraud_app
DB_PASSWORD=your_secure_password
DB_NAME=merchant_fraud_db
SECRET_KEY=generate_a_random_secret_key
```

`DB_USER` and `SECRET_KEY` are required in `.env`. Use a dedicated, least-privileged MySQL account; the application refuses a missing or root `DB_USER` and refuses to start without `SECRET_KEY`. Do not commit `.env`.

The database tables are automatically initialized from [`database/schema.sql`](file:///c:/Users/RATHIKA/Desktop/fraud_detection/database/schema.sql) on server startup:
1. `merchants`: Full merchant profile, verification statuses, and origin source (`synthetic`, `company`, `api`).
2. `relationships`: Entity graph edges (`bank_account`, `device`, `ip_address`).
3. `risk_assessments`: Calibrated risk scores, decisions (`Approve`, `Review`, `Hold`), SHAP reasons, and graph evidence paths.
4. `alerts`: Security alert notifications for high-risk and fraud-ring activity.

---

## 🚀 Running the Project

### 1. Start the Flask Backend & Web UI
```powershell
python app.py
```
Open **`http://localhost:5000`** in your browser to view the interactive dashboard, register new merchants, view live cold-start scores, and monitor alerts.

Authenticated JSON `POST` requests must include the CSRF token from the page's `csrf-token` meta tag in the `X-CSRFToken` header.

### 2. Run Ingestion on New Company Batch CSV
Place any new CSV file into `data/incoming/` or run directly:
```powershell
python src/process_new_data.py data/incoming/sample_merchants.csv
```

### 3. Run Automated Tests
```powershell
python -m pytest tests/
```

### 4. Run the Ten-Seed Model Study
```powershell
python -m src.repeated_comparison
```
This runs a ten-seed ring-grouped comparison of Logistic Regression, tabular XGBoost, graph XGBoost, and a tabular XGBoost business-age ablation. Thresholds for the main models are selected on validation splits. The held-out MAML comparison retains Travel and adds Fashion, with zero-shot and support-fine-tuned XGBoost baselines. Measured CSV tables and `RESULTS.md` are written under `results/run_<timestamp>/`; the run also refreshes `data/powerbi_exports/model_metrics.csv`, which the dashboard reads. The command requires PyTorch and does not create fallback benchmark scores.

---

## 📊 Power BI Dashboard Integration
All export tables are located in **`data/powerbi_exports/`**:
- **Get Data > Text/CSV** in Power BI Desktop to load:
  - `merchants_scored.csv` (Merchant profiles, decisions, and top risk reasons)
  - `daily_registrations.csv` (Registration trendline by day and approval status)
  - `graph_edges.csv` (Bipartite network relationships for Network Navigator visual)
  - `model_metrics.csv` (Baseline vs Graph PR-AUC, Recall, Precision, and F1)
  - `alerts.csv` (Open critical/high alerts)
