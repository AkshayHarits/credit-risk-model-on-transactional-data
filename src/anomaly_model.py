import pandas as pd
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from config import (
    TRAIN_FEATURES_PATH, SCALER_PATH, ANOMALY_MODEL_PATH, RANDOM_STATE, MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Trains an Isolation Forest model to detect anomalous user behavior."""
    logging.info("Starting anomaly model training pipeline...")
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Load the TRAINING feature set
    features_df = pd.read_csv(TRAIN_FEATURES_PATH).set_index('client_id')
    logging.info(f"Loaded {len(features_df)} users for training.")
    
    # 2. Scale the features
    logging.info("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    joblib.dump(scaler, SCALER_PATH)

    # 3. Train the Isolation Forest model
    logging.info("Training Isolation Forest model...")
    iso_forest = IsolationForest(
        contamination='auto',
        random_state=RANDOM_STATE
    )
    iso_forest.fit(scaled_features)
    joblib.dump(iso_forest, ANOMALY_MODEL_PATH)
    logging.info(f"Anomaly model training complete and saved to {ANOMALY_MODEL_PATH}")

if __name__ == "__main__":
    main()