\# Mobile Money Fraud Detection + Gen AI



Detects fraudulent mobile money transactions and explains

WHY it is fraud in plain English using GPT-4o.



\## Tech Stack

\- Dataset   : PaySim Mobile Money (Kaggle)

\- ML Model  : XGBoost + LightGBM

\- Imbalance : SMOTE

\- Explain   : SHAP values

\- Gen AI    : GPT-4o

\- API       : FastAPI

\- UI        : Streamlit

\- Tracking  : MLflow



\## How to Run

1\. pip install -r requirements.txt

2\. jupyter notebook

3\. uvicorn api.main:app --reload

4\. streamlit run app/streamlit\_app.py

