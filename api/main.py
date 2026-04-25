from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap
import os
from dotenv import load_dotenv
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

load_dotenv('.env')

app = FastAPI(
    title="Mobile Fraud Detection API",
    description="Detects fraudulent mobile money transactions and explains why",
    version="1.0.0"
)

# Load all models once when API starts
print("Loading models...")
model     = joblib.load('models/xgb_model.pkl')
explainer = joblib.load('models/shap_explainer.pkl')
X_sample  = joblib.load('models/X_sample.pkl')
client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL     = "llama-3.3-70b-versatile"
print("Models loaded successfully")


# Input schema — what the API expects
class Transaction(BaseModel):
    step:              float
    type:              float
    amount:            float
    oldbalanceOrg:     float
    newbalanceOrig:    float
    oldbalanceDest:    float
    newbalanceDest:    float


# Helper — engineer features same as preprocessing
def engineer_features(t: Transaction):
    data = {
        'step':                    t.step,
        'type':                    t.type,
        'amount':                  t.amount,
        'oldbalanceOrg':           t.oldbalanceOrg,
        'newbalanceOrig':          t.newbalanceOrig,
        'oldbalanceDest':          t.oldbalanceDest,
        'newbalanceDest':          t.newbalanceDest,
        'balance_change_orig':     t.oldbalanceOrg - t.newbalanceOrig,
        'orig_balance_zero':       1 if t.newbalanceOrig == 0 else 0,
        'dest_balance_unchanged':  1 if t.oldbalanceDest == t.newbalanceDest else 0,
        'amount_to_balance_ratio': t.amount / (t.oldbalanceOrg + 1)
    }
    return pd.DataFrame([data])


# Helper — get SHAP explanation
def get_shap_text(df):
    shap_vals = explainer.shap_values(df)[0]
    shap_df   = pd.DataFrame({
        'feature':    df.columns,
        'value':      df.iloc[0].values,
        'shap_value': shap_vals
    }).sort_values('shap_value', key=abs, ascending=False)
    return shap_df.head(5)


# Helper — get Groq explanation
def get_groq_explanation(df, fraud_prob, is_fraud, top_features):
    features_text = ""
    for _, row in top_features.iterrows():
        direction      = "increases" if row['shap_value'] > 0 else "decreases"
        features_text += f"- {row['feature']}: value={row['value']:.2f}, {direction} fraud risk (impact={abs(row['shap_value']):.3f})\n"

    prompt = f"""
You are a fraud analyst at a bank. Analyze this mobile money transaction.

Transaction Details:
- Fraud Probability: {fraud_prob:.4f} ({fraud_prob*100:.1f}%)
- Prediction: {"FRAUD" if is_fraud else "GENUINE"}
- Transaction Amount: {df['amount'].values[0]:,.2f}
- Sender Old Balance: {df['oldbalanceOrg'].values[0]:,.2f}
- Sender New Balance: {df['newbalanceOrig'].values[0]:,.2f}

Top factors driving this prediction:
{features_text}

Write a 3 sentence explanation in simple English for a bank employee.
Explain WHY this transaction is {"suspicious" if is_fraud else "legitimate"}.
Use the actual numbers from above.
Do not use technical ML terms like SHAP or XGBoost.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a fraud analyst. Give clear concise explanations based only on the evidence provided."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=200
    )
    return response.choices[0].message.content


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "llm": "Groq Llama3"}


# Main prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    # Step 1: Engineer features
    df = engineer_features(transaction)

    # Step 2: Predict
    fraud_prob = float(model.predict_proba(df)[0][1])
    is_fraud   = fraud_prob > 0.3

    # Step 3: SHAP explanation
    top_features = get_shap_text(df)

    # Step 4: Groq plain English explanation
    explanation = get_groq_explanation(
        df, fraud_prob, is_fraud, top_features
    )

    # Step 5: Return response
    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud":          is_fraud,
        "risk_level":        "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.3 else "LOW",
        "explanation":       explanation,
        "top_features": [
            {
                "feature":    row['feature'],
                "value":      round(float(row['value']), 4),
                "shap_value": round(float(row['shap_value']), 4)
            }
            for _, row in top_features.iterrows()
        ]
    }