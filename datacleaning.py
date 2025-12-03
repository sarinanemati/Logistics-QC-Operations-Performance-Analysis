import pandas as pd
import numpy as np

df = pd.read_csv(
    "/mnt/data/synthetic_datasets/fastmove_synthetic_deliveries.csv",
    parse_dates=["created_at", "pickup_time", "estimated_arrival", "delivery_time"]
)

clean_df = df.copy()

clean_df = clean_df[clean_df["status"] != "Cancelled"]

clean_df = clean_df[(clean_df["distance_km"] > 0) & (clean_df["est_duration_min"] > 0)]


clean_df = clean_df.dropna()

clean_df["qc_issue_type"] = clean_df["qc_issue_type"].fillna("None")

clean_df["hour"] = clean_df["pickup_time"].dt.hour
clean_df["day_of_week"] = clean_df["pickup_time"].dt.dayofweek
clean_df["is_delayed"] = (clean_df["delay_minutes"] > 5).astype(int)

clean_path = "/mnt/data/synthetic_datasets/fastmove_cleaned.csv"
clean_df.to_csv(clean_path, index=False)

print("Data cleaning completed.")
print("Cleaned dataset saved to:", clean_path)
print(clean_df.head())
