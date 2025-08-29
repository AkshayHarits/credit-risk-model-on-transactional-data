# Anomaly-Based Credit Risk Assessment

## 🎯 Project Overview

This data science project addresses a critical challenge in modern finance: **assessing the creditworthiness of the "credit invisible" population**. Traditional credit scoring relies on historical data that millions of individuals lack. Our solution uses unsupervised anomaly detection on alternative financial transaction data to bridge this gap.

### Core Innovation
- **Reframes the problem**: From "is this user fraudulent?" to "how unusual is this user's financial behavior?"
- **Multi-dimensional approach**: Engineers a comprehensive "financial fingerprint" for each user
- **Transparency-first**: Every prediction is fully explainable using SHAP (SHapley Additive exPlanations)
- **Objective scoring**: Produces intuitive 0-100 credit risk scores without human bias

## 🔬 Methodological Journey

Our approach evolved through rigorous experimentation:

### ❌ Attempt 1: Supervised Learning (Abandoned)
- **Goal**: Predict pre-existing "fraud" labels
- **Issue**: Over 98% of active users incorrectly tagged as high-risk
- **Result**: Target variable proved useless for meaningful credit assessment

### ❌ Attempt 2: Unsupervised Clustering (Abandoned)  
- **Method**: Gaussian Mixture Models (GMM) for behavioral personas
- **Issue**: Required subjective human interpretation of clusters
- **Result**: Violated core objective of maintaining objectivity

### ✅ Final Approach: Anomaly Detection (Successful)
- **Method**: Isolation Forest algorithm
- **Advantage**: Measures behavioral deviation directly
- **Benefits**: Objective, granular, individually-scored, and natively explainable

## 📁 Project Structure

```
anomaly-based-credit-risk/
│
├── data/
│   ├── transactions_data.csv      # Main dataset (download required)
│   ├── users_data.csv            # User demographics
│   ├── mcc_codes.json           # Merchant category codes
│   ├── train_features.csv       # Generated training features
│   └── test_features.csv        # Generated test features
│
├── models/
│   ├── anomaly_model.joblib     # Trained Isolation Forest
│   ├── scaler.joblib           # Feature scaler
│   └── geo_clusterer.joblib    # Geographic clustering model
│
├── src/
│   ├── config.py               # Configuration & file paths
│   ├── feature_engineering.py  # Data cleaning & feature creation
│   ├── anomaly_model.py        # Model training pipeline
│   └── dashboard.py            # Interactive Streamlit dashboard
│
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd anomaly-based-credit-risk
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Data Download & Setup

> **Important**: Due to file size constraints, the primary dataset is not included in the repository.

1. **Download the dataset** from Kaggle: [transactions-datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data)
2. **Extract** the `transactions_data.csv` file
3. **Place** it in the `anomaly-based-credit-risk/data/` directory

## 🏃‍♂️ How to Run

### Step 1: Data Preparation & Feature Engineering
```bash
cd src
python feature_engineering.py
```
**What this does:**
- Cleans raw transaction data
- Creates advanced behavioral features (e.g., `age_risk_factor`, `behavioral_volatility_score`)
- Partitions users into training (80%) and test (20%) sets
- Saves processed features to `data/` directory

### Step 2: Model Training
```bash
python anomaly_model.py
```
**What this does:**
- Loads `train_features.csv`
- Trains the Isolation Forest anomaly detection model
- Saves trained model and feature scaler to `models/` directory

### Step 3: Launch Interactive Dashboard
```bash
streamlit run dashboard.py
```
**Then:** Open your browser to the displayed URL (typically `http://localhost:8501`)

## 📈 Dashboard Features

### 🎯 Dynamic Risk Scoring
- **Interactive gauge**: Displays user's 0-100 credit risk score in real-time
- **Business controls**: Sidebar slider for loan officers to set risk thresholds
- **Instant decisions**: Generate "Accept" or "Reject" recommendations

### 👤 User Insights
- **Profile summary**: Key demographic and financial details
- **Behavioral analysis**: Transaction patterns and spending habits

### 🔍 Explainable AI
- **Individual risk drivers**: SHAP waterfall plots break down feature contributions
- **Global feature importance**: Model-wide behavioral weightings
- **Full transparency**: Every prediction is completely explainable

## 🔮 Future Enhancements

### 🎛️ Model Optimization
- **Hyperparameter tuning**: Implement Bayesian optimization (e.g., Optuna)
- **Algorithm exploration**: Experiment with Variational Autoencoders (VAEs)

### 🛠️ Feature Engineering
- **Time-series features**: Transaction velocity and cyclical patterns
- **Advanced behavioral metrics**: Payroll alignment and seasonal spending

### 🚀 Production Deployment
- **REST API**: Wrap scoring logic in FastAPI for programmatic access
- **Scalable architecture**: Cloud deployment for high-volume scoring
- **Real-time processing**: Stream processing for live risk assessment

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Dependency errors** | Verify all packages in `requirements.txt` are installed |
| **Missing data files** | Ensure `transactions_data.csv` is in `data/` directory |
| **Streamlit won't start** | Confirm model files exist in `models/` after training |
| **Import errors** | Check Python path and virtual environment activation |

### Getting Help
- **Check logs**: Most scripts provide detailed error messages
- **Verify file structure**: Ensure all directories match the project structure
- **Environment issues**: Try recreating the virtual environment

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

1. **🐛 Report bugs**: Open an issue with detailed error descriptions
2. **💡 Suggest features**: Submit enhancement requests via GitHub issues  
3. **🔧 Submit code**: Fork, create a feature branch, and submit a pull request
4. **📚 Improve docs**: Help us make the documentation even better

### Development Guidelines
- **Modular design**: Changes in one module shouldn't break others
- **Code quality**: Follow PEP 8 style guidelines
- **Testing**: Include unit tests for new features
- **Documentation**: Update README for significant changes

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🏆 Key Achievements

- ✅ **Objective risk assessment** without relying on flawed historical labels
- ✅ **Fully explainable predictions** using state-of-the-art SHAP technology  
- ✅ **Scalable architecture** ready for production deployment
- ✅ **Interactive dashboard** for real-time risk evaluation
- ✅ **Robust methodology** validated through iterative experimentation

---

*Built with ❤️ for financial inclusion and responsible lending practices.*