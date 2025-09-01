#  Anomaly-Based Credit Risk Assessment

> **A sophisticated credit risk assessment model designed for underbanked populations using alternative financial data and advanced anomaly detection**

## Abstract

The model tackles the critical challenge of evaluating creditworthiness in the absence of traditional financial data by leveraging alternative data, specifically customer transaction histories. Through an iterative process, the project evolved from initial supervised and clustering-based approaches to a robust unsupervised anomaly detection system.

The final model employs an **Isolation Forest algorithm** to identify users with anomalous financial behaviors, translating mathematical deviations into risk scores refined by a logical adjustment engine. A key contribution is its emphasis on **transparency** - model predictions are fully explained using **SHAP (SHapley Additive exPlanations)**, providing feature-level justifications for each user's score.

## Key Features

### **Problem Reframing**
- **From**: "Is this user fraudulent?"
- **To**: "How unusual is this user's financial behavior?"

### **Advanced Feature Engineering**
- **Financial Fingerprint**: Comprehensive behavioral stability metrics
- **Financial Health Ratios**: DTI, savings rates, spending patterns
- **Semantic Signals**: Transaction categorization and risk behavior flags

### **Logical Score Adjustment**
- Rule-based engine refines raw model output
- Applies discounts for positive "Golden Features"
- Ensures "good outliers" (debt-free individuals) receive appropriate low-risk scores

### **Full Transparency**
- Every prediction explained using SHAP waterfall plots
- Individual feature contributions visible
- Global feature importance analysis

## The Challenge of Financial Inclusion

In the global financial landscape, a significant barrier to economic growth is the lack of access to credit for "credit invisible" individuals. Traditional financial institutions rely heavily on historical credit data, creating a cyclical problem:

- **Without credit history** → Cannot access credit
- **Without credit access** → Cannot build credit history

This model breaks this cycle using alternative data sources.

##  Dataset

### Data Source
This project utilizes a comprehensive financial dataset from the **Caixabank Tech Challenge 2024**.

** Download**: [Kaggle - Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data)

### Data Components

| File | Description | Key Fields |
|------|-------------|------------|
| `transactions_data.csv` | Core transaction records | `client_id`, `amount`, `date`, `mcc` |
| `users_data.csv` | User demographics & financials | `current_age`, `yearly_income`, `total_debt`, `city`, `state` |
| `mcc_codes.json` | Merchant category mappings | MCC code to human-readable descriptions |

## Project Architecture

```
anomaly-based-credit-risk/
│
├── data/
│   ├── transactions_data.csv     # Main dataset (download required)
│   ├── users_data.csv            # User demographics
│   ├── mcc_codes.json           # Merchant category codes
│   ├── train_features.csv       # Generated training features
│   └── test_features.csv        # Generated test features
│
├── models/
│   ├── anomaly_model.joblib     # Trained Isolation Forest
│   ├── scaler.joblib            # Feature scaler
│   └── geo_clusterer.joblib     # Geographic clustering model
│
├── src/
│   ├── config.py                # Configuration & file paths
│   ├── feature_engineering.py   # Data cleaning & feature creation
│   ├── anomaly_model.py         # Model training pipeline
│   └── dashboard.py             # Interactive Streamlit dashboard
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── project_report.pdf           # Comprehensive technical report
```

## Methodological Evolution

### Attempt 1: Supervised Learning
- **Approach**: XGBoost classifier on fraud labels
- **Problem**: 98% of users labeled as "high-risk" - essentially meaningless
- **Lesson**: Traditional labels don't align with credit risk

### Attempt 2: Clustering Analysis  
- **Approach**: Gaussian Mixture Models to discover behavioral personas
- **Problem**: Required subjective human interpretation of clusters
- **Lesson**: Violated core objective of data-driven objectivity

### Final Solution: Anomaly Detection
- **Approach**: Isolation Forest for behavioral deviation measurement
- **Advantage**: Objective, granular, explainable
- **Core Insight**: *"Normalcy is a proxy for low risk"*

## Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd anomaly-based-credit-risk
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Setup
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data)
2. Extract `transactions_data.csv`
3. Place in `anomaly-based-credit-risk/data/` directory

## Usage

### Step 1: Feature Engineering
```bash
cd src
python feature_engineering.py
```

**Processes:**
- Data cleaning and validation
- Advanced behavioral feature creation
- Train/test split (80/20)
- Feature persistence

### Step 2: Model Training
```bash
python anomaly_model.py
```

**Processes:**
- Isolation Forest training
- Feature scaling and normalization
- Model persistence and validation

### Step 3: Interactive Dashboard
```bash
streamlit run dashboard.py
```

**Access:** Open `http://localhost:8501` in your browser

## Dashboard Features

### Dynamic Risk Scoring
- **Real-time Assessment**: 0-100 risk score with interactive gauge
- **Logical Adjustments**: Transparent discount system for "good outliers"
- **Business Controls**: Configurable risk thresholds
- **Decision Engine**: Automated Accept/Reject recommendations

### User Analysis
- **Financial Profile**: Demographics and key metrics
- **Behavioral Patterns**: Transaction analysis and spending habits
- **Risk Factors**: Detailed breakdown of contributing elements

### Explainable AI
- **SHAP Waterfall Plots**: Feature-level contribution analysis
- **Global Importance**: Model-wide feature impact assessment
- **Complete Transparency**: Every decision fully justified

## Advanced Feature Engineering

The model creates a sophisticated "financial fingerprint" using:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `spending_cv` | σ_monthly / μ_monthly | Spending stability measurement |
| `dti_ratio` | Total Debt / Yearly Income | Leverage assessment |
| `high_risk_behavior_flag` | Binary indicator | Distress signal detection |
| `financial_discipline_score` | 1 - Avg(Rank(cv), Rank(wants_ratio)) | Behavioral responsibility |
| `age_cushion_interaction` | Age Risk / (Financial Cushion + 1) | Compound risk modeling |
| `savings_x_discipline` | Savings Rate × Discipline Score | Positive behavior amplification |

## Model Deep Dive

### Isolation Forest Algorithm

**Core Principle**: Anomalies are easier to isolate than normal points.

**Mathematical Foundation**:
```
Anomaly Score: s(x,n) = 2^(-E(h(x))/c(n))

Where:
- E(h(x)) = Expected path length for point x
- c(n) = Average path length in BST with n points
- Score ∈ [0,1]: 0 = normal, 1 = anomaly
```

**Business Translation**:
- Raw anomaly scores → Percentile-based 0-100 risk scores
- Logical adjustments for "good outliers"
- SHAP explanations for transparency

## Case Studies

### High-Risk Profile (User 80)
- **Final Score**: 72 (High Risk)
- **Key Drivers**: Age-cushion interaction, high DTI ratio
- **Insight**: Advanced age + weak financial cushion = compound risk

### Low-Risk "Good Outlier" (User 460)
- **Final Score**: 18 (Low Risk)
- **Profile**: Statistically anomalous but financially healthy
- **Adjustment**: Logical engine applied discount for strong savings behavior

## Future Enhancements

### Enhanced Feature Engineering
- **Time-Series Analysis**: Monthly spending slope via linear regression
- **Financial Trajectory**: Detect lifestyle inflation patterns
- **Seasonal Behavior**: Capture spending seasonality

### Advanced Modeling
- **Behavioral Clustering**: Pre-model K-Means for automatic persona discovery
- **Two-Stage Architecture**:
  - Stage 1: Isolation Forest (unsupervised discovery)
  - Stage 2: XGBoost (expert-guided refinement)
  - Combined approach leveraging both paradigms

### Production Features
- **Real-time Scoring**: API endpoint for live assessments
- **Model Monitoring**: Drift detection and performance tracking
- **A/B Testing**: Framework for model comparison and optimization

## Project Achievements

- **Objective Assessment**: Data-driven scoring without biased historical labels
- **Full Explainability**: SHAP-powered transparency for every prediction
- **Smart Adjustments**: Logic engine prevents good outlier penalization
- **Production Ready**: Scalable architecture with proper train/test methodology
- **Interactive Interface**: Real-time dashboard for business users
- **Financial Inclusion**: Designed for underserved populations

## Dependencies

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
shap>=0.42.0
plotly>=5.15.0
joblib>=1.3.0
```

See `requirements.txt` for complete dependency list.

## Technical Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ for dataset and models
- **Browser**: Modern browser for Streamlit dashboard

## Documentation

- **Technical Report**: See `project_report.pdf` for comprehensive methodology
- **Code Documentation**: Inline comments throughout source files
- **Feature Definitions**: Detailed mathematical formulations included

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- **Dataset**: Caixabank Tech Challenge 2024
- **Platform**: Kaggle for data hosting
- **Libraries**: Scikit-learn, SHAP, Streamlit communities


<div align="center">

**Built with ❤️ for financial inclusion and responsible lending practices**

*Empowering the credit invisible through innovative data science*

</div>