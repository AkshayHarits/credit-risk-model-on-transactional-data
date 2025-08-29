from pathlib import Path

# --- Project Directories ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# --- Raw Data File Paths ---
TRANSACTIONS_PATH = DATA_DIR / "transactions_data.csv"
USERS_PATH = DATA_DIR / "users_data.csv"
MCC_CODES_PATH = DATA_DIR / "mcc_codes.json"

# --- Processed Data File Paths ---
TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
TEST_USER_IDS_PATH = DATA_DIR / "test_user_ids.json"

# --- Model & Asset Paths ---
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.joblib"
GEO_CLUSTERER_PATH = MODELS_DIR / "geo_clusterer.joblib"

# --- Model Training Settings ---
RANDOM_STATE = 42
TEST_SET_SIZE = 0.2
NUM_GEO_CLUSTERS = 10

USERS_PATH = DATA_DIR / "users_data.csv"