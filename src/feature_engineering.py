import pandas as pd


def create_features(df):
    df = df.copy()

    # Total money held with the bank
    df["total_balance"] = (
        df["account_balance"] + df["savings_balance"]
    )

    # Proportion of total balance held in savings
    df["savings_ratio"] = (
        df["savings_balance"] 
        / df["total_balance"].replace(0, pd.NA)
    )

    # Number of transactions per app login
    df["transactions_per_login"] = (
        df["monthly_transactions"]
        / df["monthly_app_logins"].replace(0, pd.NA)
    )

    # Whether the customer has made at least one complaint
    df["has_complaint"] = (
        df["complaints"] > 0
    ).astype(int)

    # Simple low-engagement indicator
    df["low_engagement"] = (
        (df["monthly_app_logins"] < 5)
        & (df["monthly_transactions"] < 10)
    ).astype(int)

    # Whether the customer's balance is declining
    df["balance_declining"] = (
        df["balance_change_pct"] < 0
    ).astype(int)

    return df