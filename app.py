import streamlit as st
import pandas as pd
import joblib

from src.feature_engineering import create_features

# Load model
model = joblib.load(
    "models/logistic_regression_model.pkl"
)

# Load feature list
features = joblib.load(
    "models/model_features.pkl"
)


# Page title
st.title("Banking Customer Churn Predictor")

st.write(
    """
    Adjust the customer's characteristics below to see
    how their predicted churn risk changes.
    """
)


# Customer inputs
age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

income = st.number_input(
    "Annual income (£)",
    min_value=0,
    max_value=200000,
    value=40000
)

tenure_years = st.slider(
    "Tenure (years)",
    min_value=0,
    max_value=50,
    value=5
)

account_balance = st.number_input(
    "Account balance (£)",
    min_value=0.0,
    max_value=200000.0,
    value=10000.0
)

savings_balance = st.number_input(
    "Savings balance (£)",
    min_value=0.0,
    max_value=200000.0,
    value=20000.0
)

monthly_app_logins = st.slider(
    "Monthly app logins",
    min_value=0,
    max_value=50,
    value=8
)

monthly_transactions = st.slider(
    "Monthly transactions",
    min_value=0,
    max_value=100,
    value=20
)

complaints = st.number_input(
    "Number of complaints",
    min_value=0,
    max_value=10,
    value=0
)

customer_satisfaction = st.slider(
    "Customer satisfaction",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

balance_change_pct = st.slider(
    "Balance change (%)",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=0.1
)

# Create a DataFrame from the customer inputs
customer_data = pd.DataFrame({
    "age": [age],
    "income": [income],
    "tenure_years": [tenure_years],
    "account_balance": [account_balance],
    "savings_balance": [savings_balance],
    "monthly_app_logins": [monthly_app_logins],
    "monthly_transactions": [monthly_transactions],
    "complaints": [complaints],
    "customer_satisfaction": [customer_satisfaction],
    "balance_change_pct": [balance_change_pct]
})

# Create engineered features
customer_data = create_features(customer_data)

# Select the features used by the model
customer_features = customer_data[features]

features = joblib.load(
    "models/model_features.pkl"
)

# Predict churn probability
churn_probability = model.predict_proba(
    customer_features
)[0, 1]

st.subheader("Churn prediction")

st.metric(
    "Predicted churn probability",
    f"{churn_probability:.1%}"
)

if churn_probability >= 0.5:
    st.warning("This customer is currently classified as at risk of churn.")
else:
    st.success("This customer is currently classified as lower risk of churn.")