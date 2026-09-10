import pandas as pd


def create_features(df):
    df = df.copy()

    df["total_balance"] = (
        df["account_balance"] + df["savings_balance"]
    )

    df["savings_ratio"] = (
        df["savings_balance"]
        / df["total_balance"].replace(0, pd.NA)
    )

    df["transactions_per_login"] = (
        df["monthly_transactions"]
        / df["monthly_app_logins"].replace(0, pd.NA)
    ).fillna(0)

    df["has_complaint"] = (
        df["complaints"] > 0
    ).astype(int)

    df["low_engagement"] = (
        (df["monthly_app_logins"] < 5)
        & (df["monthly_transactions"] < 10)
    ).astype(int)

    df["balance_declining"] = (
        df["balance_change_pct"] < 0
    ).astype(int)

    return df