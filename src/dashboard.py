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

# --- CUSTOM CSS FOR FONT SIZES ---
custom_css = """
<style>
    /* Main title */
    .st-emotion-cache-10trblm { 
        font-size: 28px; 
    }
    /* Headers */
    h2 {
        font-size: 22px;
    }
    h3 {
        font-size: 18px;
    }
    /* Metric label */
    [data-testid="stMetricLabel"] {
        font-size: 14px;
    }
    /* Metric value */
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    /* Sidebar headers */
    .st-emotion-cache-1629p8f h1{
        font-size: 18px;
    }
</style>
"""

def render_svg(svg_file):
    """Reads an SVG file and returns an HTML img tag with Base64 encoding."""
    with open(svg_file, "r") as f:
        svg_string = f.read()
    b64 = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")
    html_string = f"<div style='text-align: center;'><img src='data:image/svg+xml;base64,{b64}' width='150'/></div>"
    return html_string

@st.cache_resource
def load_assets():
    """Loads all assets and cleans the raw user data for display."""
    assets = {
        "model": joblib.load(ANOMALY_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "features": pd.read_csv(TEST_FEATURES_PATH).set_index('client_id'),
    }
    
    users_df = pd.read_csv(USERS_PATH)
    for col in ['yearly_income', 'total_debt', 'per_capita_income']:
        users_df[col] = pd.to_numeric(
            users_df[col].astype(str).str.replace('$', '', regex=False),
            errors='coerce'
        ).fillna(0)
    assets["users"] = users_df.set_index('id')

    with open(TEST_USER_IDS_PATH, 'r') as f:
        assets["test_user_ids"] = json.load(f)

    test_features_scaled = assets["scaler"].transform(assets["features"])
    raw_scores = assets["model"].decision_function(test_features_scaled)
    assets["raw_scores"] = pd.Series(raw_scores, index=assets["features"].index)
    
    assets["explainer"] = shap.TreeExplainer(assets["model"])
    assets["shap_values_test"] = assets["explainer"].shap_values(test_features_scaled)
    return assets

def get_user_risk_profile(user_id, assets):
    """Generates a risk score, explanation, and demographic data for a single user."""
    user_features = assets["features"].loc[[user_id]]
    user_demographics = assets["users"].loc[user_id]
    # --- START: NEW PERCENTILE-BASED SCORING LOGIC ---
    
    # Get the user's raw anomaly score
    raw_score = assets["raw_scores"].loc[user_id]
    
    # Calculate the percentile rank of this user's score compared to all test users.
    # 'kind=rank' handles ties. Lower raw score = more anomalous = lower percentile.
    percentile = percentileofscore(assets["raw_scores"], raw_score, kind='rank')
    
    # We invert the percentile because a low percentile (very anomalous) should be a high risk score.
    risk_score = 100 - percentile
    
    # Clip to our realistic 5-95 range
    risk_score = int(round(np.clip(risk_score, 5, 95)))

    # --- END: NEW SCORING LOGIC ---
    user_index = assets["features"].index.get_loc(user_id)
    user_shap_values = assets["shap_values_test"][user_index]
    
    explanation = shap.Explanation(
        values=user_shap_values, 
        base_values=assets["explainer"].expected_value, 
        data=user_features.iloc[0],
        feature_names=user_features.columns
    )
    
    return risk_score, explanation, user_demographics

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Anomaly Risk Assessment")
st.markdown(custom_css, unsafe_allow_html=True) # Inject CSS

assets = load_assets()

# --- Sidebar ---
st.sidebar.markdown(render_svg("logo.svg"), unsafe_allow_html=True)


st.sidebar.header("RISK APPETITE SLIDER")
risk_threshold = st.sidebar.slider(
    "Set Loan Approval Risk Threshold", 
    min_value=0, max_value=100, value=70, step=1
)
st.sidebar.header("Customer Selection")
selected_user = st.sidebar.selectbox(
    "Choose a user from the hold-out test set:",
    options=assets["test_user_ids"]
)

# --- Main Page ---
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
            loc_parts = [part for part in [demographics.get('city', ''), demographics.get('state', '')] if isinstance(part, str) and part.strip()]
            location_str = ", ".join(loc_parts)
            st.metric(label="Location", value=location_str if location_str else "N/A")
        with sub_col2:
            st.markdown("##### Financial Snapshot")
            st.metric(label="Total Debt", value=f"${demographics.get('total_debt', 0):,.0f}")
            st.metric(label="Number of Credit Cards", value=demographics.get('num_credit_cards', 'N/A'))

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Individual Risk Drivers", "Global Model Behavior"])
    with tab1:
        st.subheader(f"Risk Factor Analysis for User {selected_user}")
        st.write("This chart shows which behaviors had the biggest impact on this user's score. Red bars are features that made the user seem more normal (less risky); blue bars are features that made them seem more anomalous (riskier).")
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)
    with tab2:
        st.subheader("Global Feature Importance")
        st.write("This chart shows the top features that have the largest impact on risk scores across all users.")
        fig, ax = plt.subplots()
        shap.summary_plot(assets["shap_values_test"], assets["features"], plot_type="bar", show=False)
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)

else:
    st.info("Select a user from the sidebar to see their risk profile.")