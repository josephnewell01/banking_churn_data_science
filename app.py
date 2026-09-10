import streamlit as st
import pandas as pd
import joblib

from src.feature_engineering import create_features


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Banking Retention Simulator",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# Load model and feature list
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/logistic_regression_model.pkl"
    )


@st.cache_data
def load_features():
    return joblib.load(
        "models/model_features.pkl"
    )


model = load_model()
features = load_features()


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def predict_churn(customer_inputs):

    customer_data = pd.DataFrame({
        "age": [customer_inputs["age"]],
        "income": [customer_inputs["income"]],
        "tenure_years": [customer_inputs["tenure_years"]],
        "account_balance": [customer_inputs["account_balance"]],
        "savings_balance": [customer_inputs["savings_balance"]],
        "monthly_app_logins": [
            customer_inputs["monthly_app_logins"]
        ],
        "monthly_transactions": [
            customer_inputs["monthly_transactions"]
        ],
        "complaints": [customer_inputs["complaints"]],
        "customer_satisfaction": [
            customer_inputs["customer_satisfaction"]
        ],
        "balance_change_pct": [
            customer_inputs["balance_change_pct"]
        ]
    })

    # Apply the same feature engineering used during modelling
    customer_data = create_features(customer_data)

    # Select exactly the features used by the model
    customer_features = customer_data[features]

    # Predict probability
    probability = model.predict_proba(
        customer_features
    )[0, 1]

    return probability, customer_features


def get_classification(probability, threshold):

    if probability >= threshold:
        return "At risk"
    else:
        return "Lower risk"

def generate_recommendations(customer_inputs):

    recommendations = []

    if customer_inputs["complaints"] > 0:
        recommendations.append(
            (
                "Customer complaints",
                "Review recent service issues and consider "
                "proactive customer support."
            )
        )

    if customer_inputs["customer_satisfaction"] < 6:
        recommendations.append(
            (
                "Low satisfaction",
                "Consider a customer-service follow-up to "
                "understand and address sources of dissatisfaction."
            )
        )

    if customer_inputs["monthly_app_logins"] < 5:
        recommendations.append(
            (
                "Low app engagement",
                "Consider targeted digital engagement to "
                "encourage useful app activity."
            )
        )

    if customer_inputs["monthly_transactions"] < 10:
        recommendations.append(
            (
                "Low transaction activity",
                "Consider reviewing whether relevant banking "
                "products or services could better meet the "
                "customer's needs."
            )
        )

    if customer_inputs["balance_change_pct"] < 0:
        recommendations.append(
            (
                "Declining balance",
                "Consider reviewing whether the customer is "
                "moving funds elsewhere or changing their banking needs."
            )
        )

    if customer_inputs["tenure_years"] < 2:
        recommendations.append(
            (
                "Short tenure",
                "Consider additional engagement during the "
                "early customer relationship."
            )
        )

    return recommendations

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 Banking Customer Retention Simulator")

st.write(
    """
    Explore how changes in customer behaviour and financial
    characteristics affect predicted churn risk.
    """
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Customer scenario")

st.sidebar.write(
    "Adjust the inputs below to simulate a customer."
)


age = st.sidebar.slider(
    "Age",
    18,
    80,
    40
)

income = st.sidebar.number_input(
    "Annual income (£)",
    min_value=0,
    max_value=200000,
    value=40000
)

tenure_years = st.sidebar.slider(
    "Tenure (years)",
    0,
    50,
    5
)

account_balance = st.sidebar.number_input(
    "Account balance (£)",
    min_value=0.0,
    max_value=200000.0,
    value=10000.0
)

savings_balance = st.sidebar.number_input(
    "Savings balance (£)",
    min_value=0.0,
    max_value=200000.0,
    value=20000.0
)

monthly_app_logins = st.sidebar.slider(
    "Monthly app logins",
    0,
    50,
    8
)

monthly_transactions = st.sidebar.slider(
    "Monthly transactions",
    0,
    100,
    20
)

complaints = st.sidebar.number_input(
    "Number of complaints",
    min_value=0,
    max_value=10,
    value=0
)

customer_satisfaction = st.sidebar.slider(
    "Customer satisfaction",
    0.0,
    10.0,
    7.0,
    step=0.1
)

balance_change_pct = st.sidebar.slider(
    "Balance change (%)",
    -100.0,
    100.0,
    0.0,
    step=0.1
)


# --------------------------------------------------
# Threshold
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.header("Decision threshold")

threshold = st.sidebar.slider(
    "Churn classification threshold",
    0.10,
    0.70,
    0.50,
    step=0.05
)

st.sidebar.caption(
    "A lower threshold flags more customers as potentially "
    "at risk, increasing recall but also false positives."
)


# --------------------------------------------------
# Current customer inputs
# --------------------------------------------------

current_inputs = {
    "age": age,
    "income": income,
    "tenure_years": tenure_years,
    "account_balance": account_balance,
    "savings_balance": savings_balance,
    "monthly_app_logins": monthly_app_logins,
    "monthly_transactions": monthly_transactions,
    "complaints": complaints,
    "customer_satisfaction": customer_satisfaction,
    "balance_change_pct": balance_change_pct
}


# --------------------------------------------------
# Store baseline scenario
# --------------------------------------------------

if "baseline_inputs" not in st.session_state:

    st.session_state.baseline_inputs = (
        current_inputs.copy()
    )


if st.sidebar.button("Set current scenario as baseline"):

    st.session_state.baseline_inputs = (
        current_inputs.copy()
    )

    st.rerun()


baseline_inputs = st.session_state.baseline_inputs


# --------------------------------------------------
# Predictions
# --------------------------------------------------

baseline_probability, baseline_features = predict_churn(
    baseline_inputs
)

what_if_probability, what_if_features = predict_churn(
    current_inputs
)


baseline_classification = get_classification(
    baseline_probability,
    threshold
)

what_if_classification = get_classification(
    what_if_probability,
    threshold
)


# --------------------------------------------------
# Calculate change
# --------------------------------------------------

risk_change = (
    what_if_probability - baseline_probability
)

risk_change_percentage_points = (
    risk_change * 100
)


# --------------------------------------------------
# Scenario comparison
# --------------------------------------------------

st.header("Retention scenario")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Baseline churn risk",
        f"{baseline_probability:.1%}"
    )


with col2:

    st.metric(
        "What-if churn risk",
        f"{what_if_probability:.1%}",
        delta=f"{risk_change_percentage_points:+.1f} percentage points",
        delta_color="inverse"
    )


with col3:

    st.metric(
        "Classification",
        what_if_classification
    )


# --------------------------------------------------
# Interpretation
# --------------------------------------------------

if risk_change < 0:

    st.success(
        f"""
        The simulated scenario reduces predicted churn risk by
        **{abs(risk_change_percentage_points):.1f} percentage points**.
        """
    )

elif risk_change > 0:

    st.warning(
        f"""
        The simulated scenario increases predicted churn risk by
        **{risk_change_percentage_points:.1f} percentage points**.
        """
    )

else:

    st.info(
        "The simulated scenario does not change the predicted churn risk."
    )


# --------------------------------------------------
# Classification comparison
# --------------------------------------------------

st.subheader("Risk classification")

classification_col1, classification_col2 = st.columns(2)


with classification_col1:

    st.write("**Baseline scenario**")

    if baseline_classification == "At risk":
        st.warning("At risk")
    else:
        st.success("Lower risk")


with classification_col2:

    st.write("**What-if scenario**")

    if what_if_classification == "At risk":
        st.warning("At risk")
    else:
        st.success("Lower risk")


# --------------------------------------------------
# Prediction explanation
# --------------------------------------------------

st.header("Why does the model make this prediction?")

st.write(
    """
    The bars below show how each feature contributes to the
    simulated customer's prediction. Positive values push the
    prediction towards churn; negative values push it away.
    """
)


# Get model components
scaler = model.named_steps["scaler"]
logistic_model = model.named_steps["model"]

# Standardise what-if customer
customer_scaled = scaler.transform(
    what_if_features
)

coefficients = logistic_model.coef_[0]

contributions = (
    customer_scaled[0] * coefficients
)

contribution_df = pd.DataFrame({
    "Feature": features,
    "Contribution": contributions
})


contribution_df = contribution_df.sort_values(
    "Contribution"
)


# --------------------------------------------------
# Contribution chart
# --------------------------------------------------

st.bar_chart(
    contribution_df.set_index("Feature")[
        "Contribution"
    ]
)


# --------------------------------------------------
# Strongest factors
# --------------------------------------------------

st.subheader("Strongest factors")

positive = contribution_df[
    contribution_df["Contribution"] > 0
].sort_values(
    "Contribution",
    ascending=False
).head(5)


negative = contribution_df[
    contribution_df["Contribution"] < 0
].sort_values(
    "Contribution"
).head(5)


factor_col1, factor_col2 = st.columns(2)


with factor_col1:

    st.write("### Increasing predicted risk")

    if len(positive) == 0:

        st.write("No major factors are increasing risk.")

    else:

        for _, row in positive.iterrows():

            st.write(
                f"**{row['Feature']}**"
            )


with factor_col2:

    st.write("### Reducing predicted risk")

    if len(negative) == 0:

        st.write("No major factors are reducing risk.")

    else:

        for _, row in negative.iterrows():

            st.write(
                f"**{row['Feature']}**"
            )

# --------------------------------------------------
# Retention recommendations
# --------------------------------------------------

st.header("Potential retention actions")

st.write(
    """
    These suggestions are based on observable customer
    characteristics. They are potential actions to investigate,
    not guaranteed interventions.
    """
)

recommendations = generate_recommendations(
    current_inputs
)

if what_if_probability < 0.30:

    st.success(
        "The customer's predicted churn risk is relatively low. "
        "The suggestions below should be viewed as optional "
        "engagement opportunities rather than urgent retention actions."
    )

if recommendations:

    for factor, recommendation in recommendations:

        st.info(
            f"**{factor}**\n\n"
            f"{recommendation}"
        )

else:

    st.success(
        "No obvious retention actions were identified "
        "from the current customer characteristics."
    )

# --------------------------------------------------
# Calculated features
# --------------------------------------------------

with st.expander("View model inputs and calculated features"):

    st.write("Features supplied to the model:")

    st.dataframe(
        what_if_features.T.rename(
            columns={0: "Value"}
        )
    )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.caption(
    """
    This simulator uses a synthetic dataset and is intended
    for demonstration purposes. Model predictions represent
    statistical associations, not causal relationships or
    guaranteed customer behaviour.
    """
)