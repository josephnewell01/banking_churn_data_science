from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Settings
# -----------------------------

RANDOM_SEED = 42
NUMBER_OF_CUSTOMERS = 20_000
NUMBER_OF_TRANSACTIONS = 300_000
NUMBER_OF_SERVICE_CONTACTS = 30_000

rng = np.random.default_rng(RANDOM_SEED)


# -----------------------------
# Create output directory
# -----------------------------

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Generate customers
# -----------------------------

customer_ids = np.arange(100001, 100001 + NUMBER_OF_CUSTOMERS)

age = rng.integers(18, 81, NUMBER_OF_CUSTOMERS)

income = np.clip(
    rng.normal(38_000, 15_000, NUMBER_OF_CUSTOMERS),
    18_000,
    150_000
).round(0)

tenure = rng.integers(1, 16, NUMBER_OF_CUSTOMERS)

account_balance = np.maximum(
    rng.normal(8_000, 7_000, NUMBER_OF_CUSTOMERS),
    0
).round(2)

savings_balance = np.maximum(
    rng.normal(15_000, 20_000, NUMBER_OF_CUSTOMERS),
    0
).round(2)

monthly_app_logins = np.maximum(
    rng.poisson(12, NUMBER_OF_CUSTOMERS),
    0
)

monthly_transactions = np.maximum(
    rng.poisson(18, NUMBER_OF_CUSTOMERS),
    0
)

complaints = rng.poisson(0.4, NUMBER_OF_CUSTOMERS)

customer_satisfaction = np.clip(
    rng.normal(7.5, 1.5, NUMBER_OF_CUSTOMERS),
    1,
    10
).round(1)

balance_change_pct = np.clip(
    rng.normal(0, 15, NUMBER_OF_CUSTOMERS),
    -70,
    70
).round(2)


# -----------------------------
# Create churn probability
# -----------------------------

# We deliberately create realistic relationships
# between customer behaviour and churn.

churn_score = (
    -0.8
    - 0.08 * monthly_app_logins
    - 0.04 * monthly_transactions
    - 0.35 * customer_satisfaction
    - 0.05 * tenure
    + 0.45 * complaints
    - 0.00002 * account_balance
    - 0.01 * balance_change_pct
)

churn_probability = 1 / (1 + np.exp(-churn_score))

churned = rng.binomial(1, churn_probability)


customers = pd.DataFrame({
    "customer_id": customer_ids,
    "age": age,
    "income": income,
    "tenure_years": tenure,
    "account_balance": account_balance,
    "savings_balance": savings_balance,
    "monthly_app_logins": monthly_app_logins,
    "monthly_transactions": monthly_transactions,
    "complaints": complaints,
    "customer_satisfaction": customer_satisfaction,
    "balance_change_pct": balance_change_pct,
    "churned": churned
})


# -----------------------------
# Generate transactions
# -----------------------------

transaction_customer_ids = rng.choice(
    customer_ids,
    size=NUMBER_OF_TRANSACTIONS
)

transaction_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(
    rng.integers(0, 365, NUMBER_OF_TRANSACTIONS),
    unit="D"
)

transaction_types = rng.choice(
    [
        "Card Payment",
        "Cash Withdrawal",
        "Direct Debit",
        "Bank Transfer",
        "Standing Order"
    ],
    size=NUMBER_OF_TRANSACTIONS,
    p=[0.45, 0.10, 0.20, 0.15, 0.10]
)

transaction_amounts = np.round(
    np.exp(rng.normal(3.5, 1.0, NUMBER_OF_TRANSACTIONS)),
    2
)

transactions = pd.DataFrame({
    "transaction_id": np.arange(
        1,
        NUMBER_OF_TRANSACTIONS + 1
    ),
    "customer_id": transaction_customer_ids,
    "transaction_date": transaction_dates,
    "transaction_type": transaction_types,
    "amount": transaction_amounts
})


# -----------------------------
# Generate customer service data
# -----------------------------

service_customer_ids = rng.choice(
    customer_ids,
    size=NUMBER_OF_SERVICE_CONTACTS
)

service_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(
    rng.integers(0, 365, NUMBER_OF_SERVICE_CONTACTS),
    unit="D"
)

contact_types = rng.choice(
    [
        "General Enquiry",
        "Complaint",
        "Technical Support",
        "Account Query",
        "Fraud Concern"
    ],
    size=NUMBER_OF_SERVICE_CONTACTS,
    p=[0.40, 0.15, 0.20, 0.20, 0.05]
)

service_contacts = pd.DataFrame({
    "contact_id": np.arange(
        1,
        NUMBER_OF_SERVICE_CONTACTS + 1
    ),
    "customer_id": service_customer_ids,
    "contact_date": service_dates,
    "contact_type": contact_types
})


# -----------------------------
# Save datasets
# -----------------------------

customers.to_csv(
    OUTPUT_DIR / "customers.csv",
    index=False
)

transactions.to_csv(
    OUTPUT_DIR / "transactions.csv",
    index=False
)

service_contacts.to_csv(
    OUTPUT_DIR / "service_contacts.csv",
    index=False
)


# -----------------------------
# Print summary
# -----------------------------

print("Data generation complete!")
print()
print(f"Customers: {len(customers):,}")
print(f"Transactions: {len(transactions):,}")
print(f"Service contacts: {len(service_contacts):,}")
print()

print(
    f"Churn rate: "
    f"{customers['churned'].mean() * 100:.1f}%"
)

print()
print("Files created:")

for file in OUTPUT_DIR.iterdir():
    print(f" - {file}")