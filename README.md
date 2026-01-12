# Purchase Prediction System

An end-to-end machine learning project that predicts whether a user will make a purchase in the next 30 days based on historical transaction data.

The project reads data from PostgreSQL, performs feature engineering and label generation, trains a neural network model, and writes prediction results back to the database.

---

## Project Overview

This system answers the following business question:

> "Which users are likely to make a purchase in the next 30 days?"

Each user is assigned:
- a purchase probability
- a binary prediction (`will_buy`)

These outputs can be used for targeted marketing, campaign planning, or customer segmentation.

---

## Tech Stack

- Python  
- pandas, numpy  
- TensorFlow (Keras)  
- PostgreSQL  
- SQLAlchemy  

---

## Features

- Reads user and transaction data from PostgreSQL
- Performs feature engineering (total spend, transaction count, average spend, recency)
- Generates time-based labels using a 30-day future window
- Trains a binary classification neural network
- Writes prediction results back to PostgreSQL (`predictions` table)

---

## Database Tables

**Input tables**
- `users`
- `transactions`

**Output table**
- `predictions`
  - `user_id`
  - `as_of_date`
  - `probability`
  - `will_buy`

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Ensure PostgreSQL is running and database credentials are correct.

3. Run the pipeline:
python app/db_read.py

4. View predictions in PostgreSQL:
SELECT * FROM predictions ORDER BY user_id;
