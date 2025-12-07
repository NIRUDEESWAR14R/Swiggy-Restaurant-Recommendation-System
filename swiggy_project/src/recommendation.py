import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROCESSED_DIR = Path("data/processed/")

# Load data
encoded_df = pd.read_csv(PROCESSED_DIR / "encoded_data.csv")
clean_df = pd.read_csv(PROCESSED_DIR / "cleaned_data.csv")

# Load encoders
with open(PROCESSED_DIR / "city_encoder.pkl", "rb") as f:
    city_encoder = pickle.load(f)

with open(PROCESSED_DIR / "cuisine_encoder.pkl", "rb") as f:
    cuisine_encoder = pickle.load(f)


def build_user_vector(city, cuisine_list, rating, cost):
    """Convert user input to model feature vector."""
    
    # --- City Encoding ---
    city_vec = city_encoder.transform([[city]])

    # --- Cuisine Encoding ---
    cuisine_key = ", ".join(sorted(set(cuisine_list)))
    cuisine_vec = cuisine_encoder.transform([[cuisine_key]])

    # Numeric part
    numeric_vec = np.array([[rating, 0, cost]])  # rating_count always 0 for user

    # Final combined vector
    final_vec = np.hstack([numeric_vec, city_vec, cuisine_vec])
    return final_vec


def recommend(city, cuisine_text, min_rating, max_cost, top_n=5):
    """Return top-N recommendations using cosine similarity."""
    
    cuisine_list = [c.strip() for c in cuisine_text.split(",")]
    
    user_vec = build_user_vector(city, cuisine_list, min_rating, max_cost)

    # Compute cosine similarity
    similarity = cosine_similarity(user_vec, encoded_df.values)[0]

    clean_df_copy = clean_df.copy()
    clean_df_copy["similarity"] = similarity

    # Filters
    filtered = clean_df_copy[
        (clean_df_copy["city"] == city) &
        (clean_df_copy["rating"] >= min_rating) &
        (clean_df_copy["cost"] <= max_cost)
    ]

    if filtered.empty:
        return pd.DataFrame()

    # Sort by similarity
    results = filtered.sort_values("similarity", ascending=False).head(top_n)

    return results[
        ["name", "city", "cuisine", "rating", "rating_count", "cost", "address", "link"]
    ]


# Test run
if __name__ == "__main__":
    print("Testing recommendation...\n")

    test_output = recommend(
        city="Ambattur,Chennai",
        cuisine_text="South Indian",
        min_rating=3.5,
        max_cost=300,
        top_n=5
    )

    print(test_output)
