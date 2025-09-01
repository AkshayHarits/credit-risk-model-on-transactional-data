# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import base64
from scipy.stats import percentileofscore

from config import (
    TEST_FEATURES_PATH, SCALER_PATH, ANOMALY_MODEL_PATH, 
    TEST_USER_IDS_PATH, USERS_PATH
)

# custom_css and render_svg functions remain unchanged.
custom_css = """<style> .st-emotion-cache-10trblm {font-size: 28px;} h2 {font-size: 22px;} h3 {font-size: 18px;} [data-testid="stMetricLabel"] {font-size: 14px;} [data-testid="stMetricValue"] {font-size: 24px;} .st-emotion-cache-1629p8f h1{font-size: 18px;} </style>"""
def render_svg(svg_file):
    with open(svg_file, "r") as f: svg_string = f.read()
    b64 = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")
    return f"<div style='text-align: center;'><img src='data:image/svg+xml;base64,{b64}' width='150'/></div>"

@st.cache_resource
def load_assets():
    """Loads all assets and pre-calculates values for the dashboard."""
    assets = {
        "model": joblib.load(ANOMALY_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "features": pd.read_csv(TEST_FEATURES_PATH).set_index('client_id'),
    }
    users_df = pd.read_csv(USERS_PATH)
    for col in ['yearly_income', 'total_debt', 'per_capita_income']:
        users_df[col] = pd.to_numeric(users_df[col].astype(str).str.replace('$', '', regex=False),errors='coerce').fillna(0)
    assets["users"] = users_df.set_index('id')
    with open(TEST_USER_IDS_PATH, 'r') as f: assets["test_user_ids"] = json.load(f)
    test_features_scaled = assets["scaler"].transform(assets["features"])
    assets["raw_scores"] = pd.Series(assets["model"].decision_function(test_features_scaled), index=assets["features"].index)
    assets["explainer"] = shap.TreeExplainer(assets["model"])
    assets["shap_values_test"] = assets["explainer"].shap_values(test_features_scaled)
    # Pre-calculate discipline score threshold for adjustments
    assets["discipline_q3"] = assets["features"]['financial_discipline_score'].quantile(0.75)
    return assets

def get_user_risk_profile(user_id, assets):
    """Generates a risk score with logical adjustments, and SHAP explanation."""
    user_features = assets["features"].loc[[user_id]]
    user_demographics = assets["users"].loc[user_id]
    
    # --- START: LOGICAL SCORE ADJUSTMENT ---
    raw_score = assets["raw_scores"].loc[user_id]
    risk_score = 100 - percentileofscore(assets["raw_scores"], raw_score, kind='rank')
    
    discount_factor = 0.0
    discount_reasons = []

    # Apply discounts for strong positive financial signals
    if user_features['is_debt_free'].iloc[0] == 1:
        discount_factor += 0.40 # 40% discount for being debt-free
        discount_reasons.append("Debt-Free")
    if user_features['high_savings_behavior'].iloc[0] == 1:
        discount_factor += 0.25 # 25% discount for high savings
        discount_reasons.append("High Savings")
    if user_features['financial_discipline_score'].iloc[0] >= assets["discipline_q3"]:
        discount_factor += 0.15 # 15% for top-quartile financial discipline
        discount_reasons.append("High Discipline")

    # Apply the discount to the anomaly-based score
    adjusted_risk_score = risk_score * (1 - min(discount_factor, 0.75)) # Cap discount at 75%
    
    if discount_factor > 0:
        st.info(f"**Score Adjustment Applied!** Base Score: {int(round(risk_score))}, Final Score: {int(round(adjusted_risk_score))}. Reasons: {', '.join(discount_reasons)}.")
    
    final_score = int(round(np.clip(adjusted_risk_score, 5, 95)))
    # --- END: LOGICAL SCORE ADJUSTMENT ---

    user_index = assets["features"].index.get_loc(user_id)
    explanation = shap.Explanation(
        values=assets["shap_values_test"][user_index], 
        base_values=assets["explainer"].expected_value, 
        data=user_features.iloc[0],
        feature_names=user_features.columns
    )
    
    return final_score, explanation, user_demographics

st.set_page_config(layout="wide", page_title="Anomaly Risk Assessment")
st.markdown(custom_css, unsafe_allow_html=True)

assets = load_assets()

st.sidebar.markdown(render_svg("logo.svg"), unsafe_allow_html=True)
st.sidebar.header("RISK APPETITE SLIDER")
risk_threshold = st.sidebar.slider("Set Loan Approval Risk Threshold", 0, 100, 70, 1)
st.sidebar.header("Customer Selection")
selected_user = st.sidebar.selectbox("Choose a user from the hold-out test set:", options=assets["test_user_ids"])

st.title("Credit Risk Assessment Dashboard")

if selected_user:
    risk_score, explanation, demographics = get_user_risk_profile(selected_user, assets)

    st.header(f"Risk Profile for User: {selected_user}")
    
    st.subheader("Loan Recommendation")
    if risk_score >= risk_threshold:
        st.error(f"**Decision: REJECT** (User risk score of {risk_score} is above the threshold of {risk_threshold})")
    else:
        st.success(f"**Decision: ACCEPT** (User risk score of {risk_score} is below the threshold of {risk_threshold})")
    st.markdown("---")

    col_score, col_profile = st.columns([1, 2])
    with col_score:
        st.subheader("Credit Risk Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score,
            title={'text': "Risk Score (0-100)"},
            gauge={
                'axis': {'range': [None, 100]}, 'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ca02c"},
                    {'range': [40, 70], 'color': "#ff7f0e"},
                    {'range': [70, 100], 'color': "#d62728"}],
            }))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.info("A higher score indicates the user's financial behavior is more anomalous.")

    with col_profile:
        st.subheader("User Profile Summary")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("##### Demographic Profile")
            st.metric(label="Age", value=f"{demographics['current_age']} years")
            st.metric(label="Yearly Income", value=f"${demographics.get('yearly_income', 0):,.0f}")
        with sub_col2:
            st.markdown("##### Financial Snapshot")
            st.metric(label="Total Debt", value=f"${demographics.get('total_debt', 0):,.0f}")
            lat = demographics.get('latitude', 0.0); lon = demographics.get('longitude', 0.0)
            st.metric(label="Coordinates (Lat, Lon)", value=f"{lat:.2f}, {lon:.2f}")
            
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Individual Risk Drivers", "Global Model Behavior"])
    with tab1:
        st.subheader(f"Risk Factor Analysis for User {selected_user}")
        st.write("This chart shows which behaviors had the biggest impact on this user's score. Red bars are features that made the user seem more normal (less risky); blue bars are features that made them seem more anomalous (riskier).")
        fig, ax = plt.subplots()
        # Cleaner plot with fewer features
        shap.plots.waterfall(explanation, max_display=8, show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)
    with tab2:
        st.subheader("Global Feature Importance")
        st.write("This chart shows the top features that have the largest impact on risk scores across all users.")
        fig, ax = plt.subplots()
        # Cleaner plot with fewer features
        shap.summary_plot(assets["shap_values_test"], assets["features"], plot_type="bar", max_display=12, show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)
else:
    st.info("Select a user from the sidebar to see their risk profile.")