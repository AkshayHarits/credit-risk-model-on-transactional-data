# feature_engineering.py
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
    # This function remains unchanged.
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
    with open(MCC_CODES_PATH, 'r') as f: mcc_data = json.load(f)
    mcc_df = pd.DataFrame(list(mcc_data.items()), columns=['mcc', 'mcc_description'])
    mcc_df['mcc'] = pd.to_numeric(mcc_df['mcc'])
    master_df = pd.merge(transactions_df, mcc_df, on='mcc', how='left')
    master_df['transaction_time'] = pd.to_datetime(master_df['date'], utc=True)
    return master_df, users_df

def create_behavioral_features(transactions, users):
    """Engineers a streamlined, powerful, and non-redundant feature set."""
    logging.info(f"Starting streamlined feature engineering for {users['client_id'].nunique()} users...")
    users = users.set_index('client_id')
    features = pd.DataFrame(index=users.index)
    
    # --- Section 1: Foundational Aggregates (Streamlined) ---
    agg_features = transactions.groupby('client_id')['amount'].agg(['sum', 'mean', 'count', 'max']).add_prefix('txn_')
    features = features.join(agg_features)
    
    # --- Section 2: Advanced Stability & Consistency Metrics ---
    transactions['month'] = transactions['transaction_time'].dt.to_period('M')
    monthly_spend = transactions.groupby(['client_id', 'month'])['amount'].sum().unstack().fillna(0)
    monthly_mean_spend = monthly_spend.mean(axis=1).replace(0, 1e-6)
    features['spending_cv'] = monthly_spend.std(axis=1) / monthly_mean_spend
    time_deltas = transactions.sort_values('transaction_time').groupby('client_id')['transaction_time'].diff().dt.days
    features['txn_regularity_days'] = time_deltas.groupby(transactions['client_id']).mean()
    features['days_since_last_txn'] = (transactions['transaction_time'].max() - transactions.groupby('client_id')['transaction_time'].max()).dt.days

    # --- Section 3: Sophisticated Financial Health Ratios ---
    features['dti_ratio'] = users['total_debt'] / (users['yearly_income'] + 1e-6)
    user_activity_duration_yrs = (transactions.groupby('client_id')['transaction_time'].max() - transactions.groupby('client_id')['transaction_time'].min()).dt.days / 365.25
    total_inferred_income = users['yearly_income'] * user_activity_duration_yrs
    features['estimated_savings_rate'] = (total_inferred_income - features['txn_sum']) / (total_inferred_income + 1e-6)
    
    # --- Section 4: Semantic Transaction Analysis (Path 2 Implementation) ---
    logging.info("Adding semantic features for financial distress...")
    DISTRESS_MCCS = [6010, 6011, 6051, 7321, 5933] # Cash Advance, Debt Collectors, Pawn Shops
    GAMBLING_MCCS = [7995, 7801]
    features['distress_txn_count'] = transactions[transactions['mcc'].isin(DISTRESS_MCCS)].groupby('client_id')['amount'].count()
    gambling_spend = transactions[transactions['mcc'].isin(GAMBLING_MCCS)].groupby('client_id')['amount'].sum()
    features['gambling_spend_ratio'] = gambling_spend / features['txn_sum'].replace(0, 1e-6)

    # --- Section 5: "Golden Features" & Final Interaction Term ---
    features['is_debt_free'] = (users['total_debt'] == 0).astype(int)
    features['high_savings_behavior'] = (features['estimated_savings_rate'] > 0.25).astype(int)
    wants_spend_ratio = transactions[transactions['mcc'].isin([5812, 5814, 7996, 7832, 5813, 7011, 4722])].groupby('client_id')['amount'].sum() / features['txn_sum'].replace(0, 1e-6)
    features['financial_discipline_score'] = 1 - (features['spending_cv'].rank(pct=True) + wants_spend_ratio.rank(pct=True)) / 2

    # Using the more powerful interaction term directly, making raw age_risk redundant
    age_risk_factor = users['current_age'].apply(lambda age: max(0, 30 - age) + max(0, age - 60) * 1.5)
    financial_cushion_score = users['yearly_income'].rank(pct=True) - users['total_debt'].rank(pct=True)
    features['age_cushion_interaction'] = age_risk_factor / (financial_cushion_score + 1)
    
    features.fillna(0, inplace=True)
    
    # --- Section 6: Final Transformations ---
    for col in ['txn_sum', 'txn_mean', 'txn_max', 'dti_ratio', 'spending_cv']:
        features[col] = np.log1p(features[col])

    for col in ['financial_discipline_score', 'age_cushion_interaction', 'txn_regularity_days']:
        lower_bound, upper_bound = features[col].quantile(0.01), features[col].quantile(0.99)
        features[col] = features[col].clip(lower_bound, upper_bound)
        
    return features.fillna(0)

# The 'main' function remains unchanged and will work with the new feature set.
def main():
    logging.info("Starting full feature engineering pipeline...")
    MODELS_DIR.mkdir(exist_ok=True)
    transactions, users = load_and_prep_data()
    train_users, test_users = train_test_split(users, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE)
    test_user_ids = test_users['client_id'].tolist()
    with open(TEST_USER_IDS_PATH, 'w') as f: json.dump(test_user_ids, f)
    logging.info(f"Split data: {len(train_users)} training users, {len(test_users)} test users.")
    train_transactions = transactions[transactions['client_id'].isin(train_users['client_id'])].copy()
    test_transactions = transactions[transactions['client_id'].isin(test_users['client_id'])].copy()
    geo_coords_train = train_users[['latitude', 'longitude']].dropna()
    clusterer = KMeans(n_clusters=NUM_GEO_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    train_users.loc[geo_coords_train.index, 'geo_cluster'] = clusterer.fit_predict(geo_coords_train)
    joblib.dump(clusterer, GEO_CLUSTERER_PATH)
    logging.info("Processing the training set...")
    train_features = create_behavioral_features(train_transactions, train_users)
    train_features = train_features.join(train_users.set_index('client_id')['geo_cluster'])
    train_features.to_csv(TRAIN_FEATURES_PATH)
    logging.info(f"Training features saved to {TRAIN_FEATURES_PATH}")
    logging.info("Processing the test set...")
    geo_coords_test = test_users[['latitude', 'longitude']].dropna()
    if not geo_coords_test.empty:
        test_users.loc[geo_coords_test.index, 'geo_cluster'] = clusterer.predict(geo_coords_test)
    test_features = create_behavioral_features(test_transactions, test_users)
    test_features = test_features.join(test_users.set_index('client_id')['geo_cluster'])
    test_features.to_csv(TEST_FEATURES_PATH)
    logging.info(f"Test features saved to {TEST_FEATURES_PATH}")

if __name__ == "__main__":
    main()