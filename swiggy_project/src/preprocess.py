import pandas as pd
import pickle
from pathlib import Path

RAW_PATH = Path("data/raw/swiggy.csv")
CLEAN_PATH = Path("data/processed/cleaned_data.csv")
ENCODED_PATH = Path("data/processed/encoded_data.csv")

CITY_ENCODER_PATH = Path("data/processed/city_encoder.pkl")
CUISINE_LIST_PATH = Path("data/processed/cuisine_list.pkl")

print("\n🔧 Loading raw data...")
df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)

# ---------------------- CLEANING ----------------------

# Replace invalid ratings (“--”) & convert to float
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating"].fillna(df["rating"].median(), inplace=True)

# Replace missing rating_count
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)

# Fix cost — convert & fill missing with 300
df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
df["cost"].fillna(300, inplace=True)

# Standardize city & cuisine
df["city"] = df["city"].astype(str).str.strip().fillna("unknown")
df["cuisine"] = df["cuisine"].astype(str).str.lower().str.replace("&", ",")

# Convert cuisine string → list
df["cuisine_list"] = df["cuisine"].str.split(",")

# Save cleaned data
df.to_csv(CLEAN_PATH, index=False)
print("✅ cleaned_data.csv saved.")

# ---------------------- ENCODING ----------------------

print("🏙️ Encoding city names...")
from sklearn.preprocessing import OneHotEncoder

city_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
city_matrix = city_enc.fit_transform(df[["city"]])

print("🍽️ Extracting cuisine categories...")

# Unique cuisines
all_cuisines = sorted({
    c.strip()
    for lst in df["cuisine_list"]
    for c in lst
    if c.strip()
})

# SAVE CUISINE LIST for Streamlit
with open(CUISINE_LIST_PATH, "wb") as f:
    pickle.dump(all_cuisines, f)

# Create binary multi-hot cuisine matrix
df_cuisine = pd.DataFrame(0, index=df.index, columns=all_cuisines)

for idx, lst in enumerate(df["cuisine_list"]):
    for c in lst:
        c = c.strip()
        if c in df_cuisine.columns:
            df_cuisine.at[idx, c] = 1

# ---------------------- FINAL ENCODED DATA ----------------------

encoded_df = pd.concat(
    [
        pd.DataFrame(city_matrix, index=df.index, columns=city_enc.get_feature_names_out(["city"])),
        df_cuisine,
        df[["rating", "rating_count", "cost"]],
    ],
    axis=1
)

encoded_df.to_csv(ENCODED_PATH, index=False)
print("💾 encoded_data.csv saved.")

# Save city encoder
with open(CITY_ENCODER_PATH, "wb") as f:
    pickle.dump(city_enc, f)

print("\n🎉 Preprocessing completed successfully!")
print("Saved files:")
print(" - cleaned_data.csv")
print(" - encoded_data.csv")
print(" - city_encoder.pkl")
print(" - cuisine_list.pkl")
