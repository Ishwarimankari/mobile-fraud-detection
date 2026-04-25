import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Mobile Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Mobile Money Fraud Detection")
st.markdown("Powered by **XGBoost + SHAP + Groq LLaMA 3**")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app detects fraudulent mobile money 
    transactions using Machine Learning and 
    explains WHY in plain English using Gen AI.
    
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
    step            = st.number_input("Step (Hour of transaction)", min_value=1, max_value=744, value=1)
    transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount          = st.number_input("Transaction Amount", min_value=0.0, value=50000.0, step=1000.0)
    old_balance_org = st.number_input("Sender Old Balance", min_value=0.0, value=50000.0, step=1000.0)
    new_balance_org = st.number_input("Sender New Balance", min_value=0.0, value=0.0, step=1000.0)

with col2:
    old_balance_dest = st.number_input("Receiver Old Balance", min_value=0.0, value=10000.0, step=1000.0)
    new_balance_dest = st.number_input("Receiver New Balance", min_value=0.0, value=10000.0, step=1000.0)

st.markdown("---")

# Prefill buttons
st.subheader("Quick Test")
col3, col4 = st.columns(2)

with col3:
    fraud_test = st.button("Load Fraud Example 🚨")
with col4:
    genuine_test = st.button("Load Genuine Example ✅")

# Session state for prefill
if 'prefill' not in st.session_state:
    st.session_state.prefill = None

if fraud_test:
    st.session_state.prefill = 'fraud'
    st.rerun()

if genuine_test:
    st.session_state.prefill = 'genuine'
    st.rerun()

st.markdown("---")

# Predict button
predict_button = st.button("🔍 Check Transaction", type="primary", use_container_width=True)

if predict_button:
    # Build payload
    payload = {
        "step":           float(step),
        "type":           float(type_mapping[transaction_type]),
        "amount":         float(amount),
        "oldbalanceOrg":  float(old_balance_org),
        "newbalanceOrig": float(new_balance_org),
        "oldbalanceDest": float(old_balance_dest),
        "newbalanceDest": float(new_balance_dest)
    }

    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=30
            )
            result = response.json()

            st.markdown("---")
            st.subheader("Analysis Result")

            # Show fraud probability
            col5, col6, col7 = st.columns(3)

            with col5:
                prob = result['fraud_probability']
                st.metric(
                    label="Fraud Probability",
                    value=f"{prob * 100:.1f}%"
                )

            with col6:
                risk = result['risk_level']
                color = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
                st.metric(
                    label="Risk Level",
                    value=f"{color} {risk}"
                )

            with col7:
                verdict = "🚨 FRAUD" if result['is_fraud'] else "✅ GENUINE"
                st.metric(
                    label="Verdict",
                    value=verdict
                )

            st.markdown("---")

            # Show explanation
            st.subheader("AI Explanation")
            if result['is_fraud']:
                st.error(result['explanation'])
            else:
                st.success(result['explanation'])

            st.markdown("---")

            # Show top features
            st.subheader("Top Factors")
            for f in result['top_features']:
                direction = "⬆️ increases" if f['shap_value'] > 0 else "⬇️ decreases"
                impact    = abs(f['shap_value'])
                st.write(
                    f"**{f['feature']}** = {f['value']:.2f} "
                    f"→ {direction} fraud risk "
                    f"(impact: {impact:.3f})"
                )

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {str(e)}")