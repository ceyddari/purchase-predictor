import pandas as pd
import numpy as np
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


future_window_end = today + pd.Timedelta(days=30)

future_tx = transactions_df[
    (transactions_df["created_at"] > today)
    & (transactions_df["created_at"] <= future_window_end)
]

labels = future_tx.groupby("user_id").size().reset_index(name="tx_count_future")

dataset = features.merge(labels, on="user_id", how="left")

dataset["will_buy"] = (dataset["tx_count_future"].fillna(0) > 0).astype(int)

dataset.drop(columns=["tx_count_future", "last_transaction"], inplace=True)

feature_cols = ["total_spent", "transaction_count", "avg_spent", "recency_days"]

X = dataset[feature_cols].to_numpy(dtype=float)
y = dataset["will_buy"].to_numpy(dtype=int)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0

X = (X - X_mean) / X_std

# train/test
rng = np.random.default_rng(42)
idx = np.arange(len(X))
rng.shuffle(idx)

split = int(len(X) * 0.8)
train_idx = idx[:split]
test_idx = idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]


print(dataset)
