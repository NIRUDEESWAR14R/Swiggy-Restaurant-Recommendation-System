import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

PROCESSED_DIR = Path("data/processed/")
INPUT_FILE = PROCESSED_DIR / "cleaned_data.csv"

ENCODED_FILE = PROCESSED_DIR / "encoded_data.csv"
CITY_ENCODER_FILE = PROCESSED_DIR / "city_encoder.pkl"
CUISINE_ENCODER_FILE = PROCESSED_DIR / "cuisine_encoder.pkl"


def preprocess():
    print("🔧 Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    print("Cleaned data shape:", df.shape)

    # ----------------------------
    # 1. One-Hot Encode City
    # ----------------------------
    print("🏙️ Encoding city...")
    city_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    city_encoded = city_encoder.fit_transform(df[["city"]])
    city_df = pd.DataFrame(
        city_encoded,
        columns=city_encoder.get_feature_names_out(["city"])
    )

    # ----------------------------
    # 2. One-Hot Encode Cuisine
    # ----------------------------
    print("🍽️ Encoding cuisine...")

    # Convert "A, B, C" → ["A", "B", "C"]
    df["cuisine_list"] = df["cuisine"].apply(
        lambda x: [c.strip() for c in str(x).split(",")]
    )

    # Cuisine encoder (multi-label one-hot)
    cuisine_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    cuisine_encoded = cuisine_encoder.fit_transform(
        df["cuisine_list"].apply(lambda x: ", ".join(sorted(set(x)))).values.reshape(-1, 1)
    )

    cuisine_df = pd.DataFrame(
        cuisine_encoded,
        columns=cuisine_encoder.get_feature_names_out(["cuisine"])
    )

    # ----------------------------
    # 3. Combine Encoded + Numeric Features
    # ----------------------------
    print("🔗 Combining features...")

    numeric_df = df[["rating", "rating_count", "cost"]].reset_index(drop=True)

    final_df = pd.concat([numeric_df, city_df, cuisine_df], axis=1)

    # ----------------------------
    # 4. Save encoded dataset
    # ----------------------------
    final_df.to_csv(ENCODED_FILE, index=False)
    print("💾 encoded_data.csv saved.")

    # ----------------------------
    # 5. Save encoders for Streamlit
    # ----------------------------
    with open(CITY_ENCODER_FILE, "wb") as f:
        pickle.dump(city_encoder, f)

    with open(CUISINE_ENCODER_FILE, "wb") as f:
        pickle.dump(cuisine_encoder, f)

    print("\n🎉 Preprocessing completed successfully!")
    print(f"Saved:\n - {ENCODED_FILE}\n - {CITY_ENCODER_FILE}\n - {CUISINE_ENCODER_FILE}")


if __name__ == "__main__":
    preprocess()
