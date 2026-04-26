import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Mobile Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    model     = joblib.load('models/xgb_model.pkl')
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )
    return model, explainer

model, explainer = load_models()

# Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
MODEL  = "llama-3.3-70b-versatile"

# Title
st.title("🔍 Mobile Money Fraud Detection")
st.markdown("Powered by **XGBoost + SHAP + Groq LLaMA 3**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    Detects fraudulent mobile money transactions 
    using Machine Learning and explains WHY in 
    plain English using Gen AI.
    
    **Model:** XGBoost (99.8% detection rate)
    **Explainability:** SHAP values
    **Gen AI:** Groq LLaMA 3
    **Dataset:** PaySim Mobile Money
    """)
    st.markdown("---")
    st.markdown("Built by **Ishwari Mankari**")
    st.markdown("[GitHub](https://github.com/Ishwarimankari/mobile-fraud-detection)")

# Transaction type mapping
type_mapping = {
    "CASH_IN":    0,
    "CASH_OUT":   1,
    "DEBIT":      2,
    "PAYMENT":    3,
    "TRANSFER":   4
}

# Input form
st.subheader("Enter Transaction Details")
col1, col2 = st.columns(2)

with col1:
    step             = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
    transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount           = st.number_input("Transaction Amount", min_value=0.0, value=50000.0, step=1000.0)
    old_balance_org  = st.number_input("Sender Old Balance", min_value=0.0, value=50000.0, step=1000.0)
    new_balance_org  = st.number_input("Sender New Balance", min_value=0.0, value=0.0, step=1000.0)

with col2:
    old_balance_dest = st.number_input("Receiver Old Balance", min_value=0.0, value=10000.0, step=1000.0)
    new_balance_dest = st.number_input("Receiver New Balance", min_value=0.0, value=10000.0, step=1000.0)

st.markdown("---")

# Helper functions
def engineer_features(step, type_val, amount, old_bal_org,
                       new_bal_org, old_bal_dest, new_bal_dest):
    data = {
        'step':                    step,
        'type':                    type_val,
        'amount':                  amount,
        'oldbalanceOrg':           old_bal_org,
        'newbalanceOrig':          new_bal_org,
        'oldbalanceDest':          old_bal_dest,
        'newbalanceDest':          new_bal_dest,
        'balance_change_orig':     old_bal_org - new_bal_org,
        'orig_balance_zero':       1 if new_bal_org == 0 else 0,
        'dest_balance_unchanged':  1 if old_bal_dest == new_bal_dest else 0,
        'amount_to_balance_ratio': amount / (old_bal_org + 1)
    }
    return pd.DataFrame([data])


def get_explanation(df, fraud_prob, is_fraud):
    shap_vals = explainer.shap_values(df)[0]
    shap_df   = pd.DataFrame({
        'feature':    df.columns,
        'value':      df.iloc[0].values,
        'shap_value': shap_vals
    }).sort_values('shap_value', key=abs, ascending=False).head(5)

    features_text = ""
    for _, row in shap_df.iterrows():
        direction      = "increases" if row['shap_value'] > 0 else "decreases"
        features_text += f"- {row['feature']}: value={row['value']:.2f}, {direction} fraud risk\n"

    prompt = f"""
You are a fraud analyst at a bank.

Transaction Details:
- Fraud Probability: {fraud_prob*100:.1f}%
- Prediction: {"FRAUD" if is_fraud else "GENUINE"}
- Amount: {df['amount'].values[0]:,.2f}
- Sender Old Balance: {df['oldbalanceOrg'].values[0]:,.2f}
- Sender New Balance: {df['newbalanceOrig'].values[0]:,.2f}

Top factors:
{features_text}

Write 3 sentences in simple English explaining why this 
transaction is {"suspicious" if is_fraud else "legitimate"}.
Use actual numbers. No ML technical terms.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a fraud analyst."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content, shap_df


# Predict button
if st.button("🔍 Check Transaction", type="primary", use_container_width=True):
    df = engineer_features(
        step,
        float(type_mapping[transaction_type]),
        amount,
        old_balance_org,
        new_balance_org,
        old_balance_dest,
        new_balance_dest
    )

    with st.spinner("Analyzing transaction..."):
        fraud_prob  = float(model.predict_proba(df)[0][1])
        is_fraud    = fraud_prob > 0.3
        explanation, shap_df = get_explanation(df, fraud_prob, is_fraud)

    st.markdown("---")
    st.subheader("Analysis Result")

    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
    with col4:
        risk  = "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.3 else "LOW"
        color = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        st.metric("Risk Level", f"{color} {risk}")
    with col5:
        st.metric("Verdict", "🚨 FRAUD" if is_fraud else "✅ GENUINE")

    st.markdown("---")
    st.subheader("AI Explanation")
    if is_fraud:
        st.error(explanation)
    else:
        st.success(explanation)

    st.markdown("---")
    st.subheader("Top Factors")
    for _, row in shap_df.iterrows():
        direction = "⬆️ increases" if row['shap_value'] > 0 else "⬇️ decreases"
        st.write(
            f"**{row['feature']}** = {row['value']:.2f} "
            f"→ {direction} fraud risk "
            f"(impact: {abs(row['shap_value']):.3f})"
        )