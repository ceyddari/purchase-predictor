import pandas as pd
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "purchase_ml"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

users_df = pd.read_sql("SELECT * FROM users", engine)


transactions_df = pd.read_sql("SELECT * FROM transactions", engine)


df = users_df.merge(transactions_df, left_on="id", right_on="user_id", how="left")

df["created_at"] = pd.to_datetime(df["created_at"])
today = pd.Timestamp("2025-12-31")


features = (
    df.groupby("id_x")
    .agg(
        total_spent=("amount", "sum"),
        transaction_count=("amount", "count"),
        avg_spent=("amount", "mean"),
        last_transaction=("created_at", "max"),
    )
    .reset_index()
)

features.rename(columns={"id_x": "user_id"}, inplace=True)

features["recency_days"] = (today - features["last_transaction"]).dt.days

features.fillna(
    {"total_spent": 0, "transaction_count": 0, "avg_spent": 0, "recency_days": 999},
    inplace=True,
)

print(features)
