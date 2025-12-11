import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

CLEAN_PATH = "data/processed/cleaned_data.csv"
ENCODED_PATH = "data/processed/encoded_data.csv"
CITY_ENCODER_PATH = "data/processed/city_encoder.pkl"
CUISINE_ENCODER_PATH = "data/processed/cuisine_encoder.pkl"
MODEL_PATH = "data/processed/kmeans.pkl"

print("📥 Loading all data...")
clean_df = pd.read_csv(CLEAN_PATH)
encoded_df = pd.read_csv(ENCODED_PATH)

city_enc = pickle.load(open(CITY_ENCODER_PATH, "rb"))
all_cuisines = pickle.load(open(CUISINE_ENCODER_PATH, "rb"))
kmeans = pickle.load(open(MODEL_PATH, "rb"))

print("✅ Loaded successfully.\n")

# -------------------------------------------------------------------

def recommend(city, cuisine, min_rating=3.0, max_cost=600, top_n=10):

    # Build user vector
    city_vec = city_enc.transform([[city]])[0]

    cuisine_vec = [1 if c == cuisine.lower() else 0 for c in all_cuisines]

    user_vector = list(city_vec) + cuisine_vec + [min_rating, 10]  # rating_count dummy

    # Predict cluster
    cluster_id = kmeans.predict([user_vector])[0]
    print(f"📌 User assigned to cluster: {cluster_id}")

    cluster_data = clean_df[clean_df["cluster"] == cluster_id].copy()

    # Apply filters
    results = cluster_data[
        (cluster_data["city"].str.lower() == city.lower()) &
        (cluster_data["cuisine"].str.contains(cuisine, case=False)) &
        (cluster_data["rating"] >= min_rating) &
        (cluster_data["cost"] <= max_cost)
    ]

    if results.empty:
        print("\n⚠ No exact match! Showing nearest items from cluster.\n")
        return cluster_data.head(top_n)[["name", "city", "cuisine", "rating", "cost"]]

    return results.head(top_n)[["name", "city", "cuisine", "rating", "cost"]]

# -------------------------------------------------------------------

print("Testing K-Means ONLY recommendation...\n")

res = recommend(
    city="Chennai",
    cuisine="biryani",
    min_rating=4.0,
    max_cost=500,
    top_n=5
)

print("\nRESULTS:\n")
print(res)
