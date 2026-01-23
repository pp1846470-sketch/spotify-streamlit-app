import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
final_df = pd.read_csv("spotify_preprocessed_dataset.csv")

# 🔥 SELECT FINAL FEATURES (VERY IMPORTANT)
FEATURES = [
    "artist_popularity",
    "artist_followers",
    "track_duration_min",
    "album_type",
    "explicit"
]

X = final_df[FEATURES]
y = final_df["popular"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model & Scaler saved successfully")
print("Features:", FEATURES)
