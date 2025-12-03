import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

df = pd.read_csv("/mnt/data/synthetic_datasets/fastmove_cleaned.csv")

y = df["is_delayed"]

features = [
    "distance_km",
    "est_duration_min",
    "courier_experience_years",
    "courier_rating",
    "items_count",
    "attempt_count",
    "hour",
    "day_of_week"
]

cat_cols = ["vehicle_type", "weather", "traffic_level", "zone", "root_cause"]

df_encoded = df.copy()
encoders = {}

for col in cat_cols:
    enc = LabelEncoder()
    df_encoded[col] = enc.fit_transform(df_encoded[col])
    encoders[col] = enc
    features.append(col)


X = df_encoded[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=220,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

importance = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n", feature_importance_df)

feature_importance_df.to_csv(
    "/mnt/data/synthetic_datasets/feature_importance.csv",
    index=False
)
