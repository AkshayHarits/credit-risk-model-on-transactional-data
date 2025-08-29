# FinShield: Anomaly-Based Credit Risk Assessment

## Project Overview
FinShield is an unsupervised machine learning solution designed to assess individual credit risk using alternative data, specifically financial transaction histories. Built for financial inclusion, it enables robust scoring for "credit invisible" populations without a traditional credit record. FinShield's approach is fully anomaly-based: it uses raw user and transaction data to engineer a "financial fingerprint," applies Isolation Forest models to quantify behavioral deviations, and outputs an intuitive 0-100 risk score. Every score is fully explainable on a feature-by-feature basis using SHAP (SHapley Additive exPlanations).

## Methodological Journey
- **Attempt 1: Supervised Learning (Abandoned):**
  - Flawed "fraud" labels: Over 98% of active users were incorrectly tagged as high-risk, making the target variable unsuitable.
- **Attempt 2: Unsupervised Clustering (Abandoned):**
  - GMM identified behavioral personas, but required subjective human interpretation, violating objectivity.
- **Final Approach: Anomaly Detection:**
  - Measures behavioral deviation directly, yielding objective, granular scores per user, and aligns with the transparency and inclusivity goals.

## Project Structure
finshield_credit_risk/
│
├── data/
│ ├── transactions_data.csv
│ ├── users_data.csv
│ ├── mcc_codes.json
│ ├── train_features.csv # Generated
│ └── test_features.csv # Generated
│
├── models/
│ ├── anomaly_model.joblib # Generated
│ ├── scaler.joblib # Generated
│ └── geo_clusterer.joblib # Generated
│
├── src/
│ ├── config.py # Stores file paths/settings
│ ├── feature_engineering.py # Cleans data & creates features
│ ├── anomaly_model.py # Trains Isolation Forest model
│ └── dashboard.py # Streamlit dashboard
│
└── README.md

text

## Installation & Setup

Clone the repository:
git clone <your-repo-url>
cd finshield_credit_risk

text

Create a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

text

Install dependencies:
pip install pandas scikit-learn streamlit shap matplotlib plotly scipy

Or, if requirements.txt is provided:
pip install -r requirements.txt

text

## How to Run

### 1. Data Preparation & Feature Engineering
In the `src` directory, execute:
python feature_engineering.py

text
This script cleans raw data, creates advanced behavioral features (e.g., `age_risk_factor`, `behavioral_volatility_score`), partitions users into training (80%) and test (20%) sets, and saves features under `data/`.

### 2. Model Training
Train the anomaly detection model:
python anomaly_model.py

text
This loads `train_features.csv`, fits the IsolationForest model, and saves both the trained model and feature scaler to `models/`.

### 3. Launch the Dashboard
Start the interactive Streamlit dashboard:
streamlit run dashboard.py

text
Open a browser to the displayed local URL (usually `http://localhost:8501`) to interact with FinShield.

## Dashboard Features

- **Dynamic Risk Score:** Interactive gauge displays the user’s 0-100 credit risk score.
- **Business Controls:** Sidebar slider lets loan officers set risk thresholds, instantly generating "Accept" or "Reject" recommendations.
- **User Profile Summary:** Displays key demographic and financial details for context.
- **Individual Risk Drivers:** SHAP waterfall plot provides a feature-by-feature breakdown explaining the user’s score.
- **Global Feature Importance:** Summary plot highlights which financial behaviors are most weighted by the model across the full user base.

## Troubleshooting

- If encountering dependency errors, verify your `requirements.txt` lists all needed packages.
- Ensure that raw data files are present in the `data/` directory prior to running scripts.
- For issues with Streamlit, confirm the model and scaler files exist in `models/` after running model training.

## Contribution

Please open an issue or pull request to suggest improvements, bug fixes, or add new features. The modular structure ensures changes in one module do not break the entire pipeline.