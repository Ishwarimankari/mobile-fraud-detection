\# 🔍 Mobile Money Fraud Detection + Gen AI



A complete end-to-end Machine Learning project that detects 

fraudulent mobile money transactions and explains WHY in 

plain English using Generative AI.



\---



\## 🎯 Project Overview



Financial fraud costs billions every year. This project builds 

an intelligent fraud detection system that:

\- Detects fraudulent transactions with 99.8% accuracy

\- Explains every decision in plain English using LLM

\- Serves predictions via a REST API

\- Provides an interactive web UI for analysts



\---



\## 🏗️ Architecture



Raw Transaction Data

↓

Feature Engineering (SMOTE for imbalance)

↓

XGBoost ML Model (99.8% detection rate)

↓

SHAP Values (explainability)

↓

Groq LLaMA 3 (plain English explanation)

↓

FastAPI (REST API) → Streamlit (Web UI)

↓

Azure App Service (Live Deployment)



\---



\## 💻 Tech Stack



| Category | Technology |

|---|---|

| Dataset | PaySim Mobile Money (Kaggle) |

| Language | Python 3.12 |

| ML Models | XGBoost + LightGBM |

| Imbalance | SMOTE |

| Explainability | SHAP values |

| Gen AI | Groq LLaMA 3 (llama-3.3-70b-versatile) |

| API | FastAPI + Uvicorn |

| UI | Streamlit |

| Experiment Tracking | MLflow |

| Deployment | Azure App Service |

| Version Control | GitHub |



\---



\## 📊 Dataset



\*\*PaySim Mobile Money Dataset\*\*

\- Source: Kaggle

\- Transactions: 6,362,620

\- Fraud cases: 8,213 (0.13%)

\- Features: 11 (7 original + 4 engineered)

\- Fraud types: TRANSFER and CASH\_OUT only



\*\*Key Insight from EDA:\*\*

Fraud transactions drain the entire account balance in 

one transaction. Average fraud amount is 8x higher than 

genuine transactions.



\---



\## 🔧 Feature Engineering



4 new features created to capture fraud patterns:



| Feature | Description |

|---|---|

| balance\_change\_orig | How much sender balance changed |

| orig\_balance\_zero | Did sender balance go to zero |

| dest\_balance\_unchanged | Did receiver balance not change |

| amount\_to\_balance\_ratio | Amount as ratio of old balance |



\---



\## 📈 Model Performance



| Metric | Score |

|---|---|

| ROC-AUC | 0.9995 |

| Average Precision | 0.9984 |

| Detection Rate | 99.8% |

| Fraud Detected | 1639 / 1643 |



\---



\## 🤖 How Gen AI Works



1\. XGBoost predicts fraud probability

2\. SHAP computes top 5 feature contributions

3\. Features sent to Groq LLaMA 3 as prompt

4\. LLM writes plain English explanation

5\. Explanation shown to bank analyst



Example output:



"This transaction is suspicious because the sender's

entire balance of 6,273,660 was transferred in one

transaction, leaving their account at zero. The amount

equals 100% of the sender's balance which is highly

unusual for legitimate transactions."



\---



\## 🚀 How to Run Locally



\### 1. Clone the repository

```bash

git clone https://github.com/Ishwarimankari/mobile-fraud-detection.git

cd mobile-fraud-detection

```



\### 2. Create virtual environment

```bash

python -m venv venv

venv\\Scripts\\activate

```



\### 3. Install dependencies

```bash

pip install -r requirements.txt

```



\### 4. Set up environment variables

Create a `.env` file:



\### 5. Download dataset

Download PaySim dataset from Kaggle and place in `data/paysim.csv`



\### 6. Run notebooks in order

notebooks/01\_eda.ipynb

notebooks/02\_preprocessing.ipynb

notebooks/03\_model\_training.ipynb

notebooks/04\_shap\_analysis.ipynb

notebooks/05\_genai\_explanation.ipynb



\### 7. Start FastAPI

```bash

uvicorn api.main:app --reload

```



\### 8. Start Streamlit

```bash

streamlit run app/streamlit\_app.py

```



\---



\## ☁️ Azure Deployment



This project is deployed on Microsoft Azure using:



\- \*\*Azure App Service\*\* — hosts the Streamlit web application

\- \*\*Azure Container Registry\*\* — stores Docker image

\- \*\*Docker\*\* — containerizes the entire application



\### Live URL



\### Deployment Steps

1\. Dockerize the application

2\. Push image to Azure Container Registry

3\. Deploy to Azure App Service

4\. Configure environment variables on Azure



\---



\## 📁 Project Structure



mobile-fraud-detection/

├── data/                    ← dataset (not uploaded to GitHub)

├── models/                  ← trained model files

├── notebooks/

│   ├── 01\_eda.ipynb         ← exploratory data analysis

│   ├── 02\_preprocessing.ipynb  ← data cleaning and features

│   ├── 03\_model\_training.ipynb ← XGBoost + LightGBM

│   ├── 04\_shap\_analysis.ipynb  ← explainability

│   └── 05\_genai\_explanation.ipynb ← Groq LLM integration

├── api/

│   └── main.py              ← FastAPI REST API

├── app/

│   └── streamlit\_app.py     ← Streamlit web UI

├── reports/                 ← saved charts and plots

├── .env                     ← API keys (not on GitHub)

├── .gitignore

├── requirements.txt

└── README.md



\---



\## 🎓 Key Learnings



\- Handling extreme class imbalance with SMOTE

\- SHAP values for ML model explainability

\- Integrating LLM APIs for Gen AI features

\- Building REST APIs with FastAPI

\- Deploying ML applications on Azure

\- End to end ML project structure



\---



\## 👩‍💻 Author



\*\*Ishwari Mankari\*\*



\- GitHub: \[Ishwarimankari](https://github.com/Ishwarimankari)





