import pandas as pd
import numpy as np
import json
import logging
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib

from config import (
    TRANSACTIONS_PATH, USERS_PATH, MCC_CODES_PATH, TRAIN_FEATURES_PATH,
    TEST_FEATURES_PATH, GEO_CLUSTERER_PATH, NUM_GEO_CLUSTERS,
    RANDOM_STATE, MODELS_DIR, TEST_SET_SIZE, TEST_USER_IDS_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prep_data():
    """Loads and cleans raw data for all active users."""
    logging.info("Loading and preparing raw data...")
    transactions_df = pd.read_csv(TRANSACTIONS_PATH)
    users_df = pd.read_csv(USERS_PATH)

    active_user_ids = transactions_df['client_id'].unique()
    users_df = users_df[users_df['id'].isin(active_user_ids)]
    logging.info(f"Processing {len(users_df)} active users.")

    users_df.rename(columns={'id': 'client_id'}, inplace=True)
    for col in ['total_debt', 'yearly_income', 'per_capita_income']:
        users_df[col] = pd.to_numeric(users_df[col].astype(str).str.replace('$', '', regex=False), errors='coerce')
    users_df['yearly_income'] = users_df['yearly_income'].fillna(users_df['yearly_income'].median())
    users_df['per_capita_income'] = users_df['per_capita_income'].fillna(users_df['per_capita_income'].median())
    users_df['total_debt'] = users_df['total_debt'].fillna(0)
    
    transactions_df['amount'] = pd.to_numeric(transactions_df['amount'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
    
    with open(MCC_CODES_PATH, 'r') as f:
        mcc_data = json.load(f)
    mcc_df = pd.DataFrame(list(mcc_data.items()), columns=['mcc', 'mcc_description'])
    mcc_df['mcc'] = pd.to_numeric(mcc_df['mcc'])

    master_df = pd.merge(transactions_df, mcc_df, on='mcc', how='left')
    master_df['transaction_time'] = pd.to_datetime(master_df['date'], utc=True)

    return master_df, users_df

def create_behavioral_features(transactions, users):
    """Engineers a rich set of behavioral and demographic features for each user."""
    logging.info(f"Starting feature engineering for {users['client_id'].nunique()} users...")
    users = users.set_index('client_id')
    features = pd.DataFrame(index=users.index)
    
    # Age Risk Feature
    def calculate_age_risk(age):
        prime_age_start = 30; prime_age_end = 60; SENIOR_RISK_MULTIPLIER = 1.5
        if age < prime_age_start: return prime_age_start - age
        elif age > prime_age_end: return (age - prime_age_end) * SENIOR_RISK_MULTIPLIER
        else: return 0
    features['age_risk_factor'] = users['current_age'].apply(calculate_age_risk)
    
    # Transaction Aggregates
    agg_features = transactions.groupby('client_id')['amount'].agg(['sum', 'mean', 'std', 'count', 'max']).add_prefix('txn_')
    features = features.join(agg_features)
    
    # Time-Based Features
    transactions['month'] = transactions['transaction_time'].dt.to_period('M')
    monthly_spend = transactions.groupby(['client_id', 'month'])['amount'].sum().unstack().fillna(0)
    features['monthly_spending_volatility'] = monthly_spend.std(axis=1)
    
    latest_date_in_dataset = transactions['transaction_time'].max()
    user_last_txn = transactions.groupby('client_id')['transaction_time'].max()
    features['days_since_last_txn'] = (latest_date_in_dataset - user_last_txn).dt.days

    # Spending Acceleration
    all_time_avg_spend = monthly_spend.mean(axis=1)
    last_3_months_avg_spend = monthly_spend.iloc[:, -3:].mean(axis=1)
    features['spending_acceleration'] = last_3_months_avg_spend / (all_time_avg_spend + 1e-6)
    
    # --- NEW: Conditional Financial Health Features ---
    significant_debt_threshold = 100  # Define what counts as significant debt
    
    # Create masks for users with and without significant debt
    has_significant_debt = users['total_debt'] > significant_debt_threshold
    no_significant_debt = ~has_significant_debt

    # 1. Financial Cushion Score (Safety Metric)
    income_percentile = users['yearly_income'].rank(pct=True)
    debt_percentile = users['total_debt'].rank(pct=True)
    
    # For users with debt, the score is income rank minus debt rank
    features.loc[has_significant_debt, 'financial_cushion_score'] = income_percentile - debt_percentile
    # For users with no debt, their cushion is simply based on their income rank
    features.loc[no_significant_debt, 'financial_cushion_score'] = income_percentile

    # 2. Debt Burden Score (Risk Metric)
    monthly_income = users['yearly_income'] / 12
    estimated_monthly_debt_service = users['total_debt'] / 60
    
    # For users with debt, calculate their burden
    debt_burden = estimated_monthly_debt_service / (monthly_income + 1e-6)
    features.loc[has_significant_debt, 'debt_burden_score'] = debt_burden
    # For users with no debt, their burden is 0 (the safest value)
    features.loc[no_significant_debt, 'debt_burden_score'] = 0
    # --- END NEW ---

    # Behavioral Volatility Score
    volatility_percentile = features['monthly_spending_volatility'].rank(pct=True)
    txn_std_percentile = features['txn_std'].rank(pct=True)
    acceleration_percentile = features['spending_acceleration'].rank(pct=True)
    features['behavioral_volatility_score'] = (volatility_percentile + txn_std_percentile + acceleration_percentile) / 3
    
    # Spending Ratios
    total_spend = features['txn_sum'].replace(0, 1e-6)
    WANTS_MCCS = [5812, 5814, 7996, 7832, 5813, 7011, 4722]
    VICE_MCCS = [7995, 5921]
    PLANNING_MCCS = [6300, 8931, 7276]

    def calculate_spend_ratio(mcc_list):
        spend = transactions[transactions['mcc'].isin(mcc_list)].groupby('client_id')['amount'].sum()
        return spend / total_spend

    features['wants_spend_ratio'] = calculate_spend_ratio(WANTS_MCCS)
    features['vice_spend_ratio'] = calculate_spend_ratio(VICE_MCCS)
    features['planning_spend_ratio'] = calculate_spend_ratio(PLANNING_MCCS)
    
    return features.fillna(0)

def main():
    """Main pipeline to generate train and test feature sets."""
    logging.info("Starting full feature engineering pipeline...")
    MODELS_DIR.mkdir(exist_ok=True)
    
    transactions, users = load_and_prep_data()

    train_users, test_users = train_test_split(
        users, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )
    test_user_ids = test_users['client_id'].tolist()
    with open(TEST_USER_IDS_PATH, 'w') as f:
        json.dump(test_user_ids, f)
    logging.info(f"Split data: {len(train_users)} training users, {len(test_users)} test users.")
    
    train_transactions = transactions[transactions['client_id'].isin(train_users['client_id'])].copy()
    test_transactions = transactions[transactions['client_id'].isin(test_users['client_id'])].copy()

    geo_coords_train = train_users[['latitude', 'longitude']].dropna()
    clusterer = KMeans(n_clusters=NUM_GEO_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    train_users['geo_cluster'] = clusterer.fit_predict(geo_coords_train)
    joblib.dump(clusterer, GEO_CLUSTERER_PATH)

    logging.info("Processing the training set...")
    train_features = create_behavioral_features(train_transactions, train_users)
    train_features = train_features.join(train_users.set_index('client_id')['geo_cluster'])
    train_features.to_csv(TRAIN_FEATURES_PATH)
    logging.info(f"Training features saved to {TRAIN_FEATURES_PATH}")
    
    logging.info("Processing the test set...")
    geo_coords_test = test_users[['latitude', 'longitude']].dropna()
    if not geo_coords_test.empty:
        test_users['geo_cluster'] = clusterer.predict(geo_coords_test)
    test_features = create_behavioral_features(test_transactions, test_users)
    test_features = test_features.join(test_users.set_index('client_id')['geo_cluster'])
    test_features.to_csv(TEST_FEATURES_PATH)
    logging.info(f"Test features saved to {TEST_FEATURES_PATH}")

if __name__ == "__main__":
    main()