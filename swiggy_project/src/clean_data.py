import pandas as pd
from pathlib import Path

# Folder paths
RAW_PATH = Path("data/raw/swiggy.csv")
PROCESSED_DIR = Path("data/processed/")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_data():
    print("📥 Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Clean cost column (remove ₹)
    df["cost"] = (
        df["cost"]
        .astype(str)
        .str.replace("₹", "")
        .str.replace(",", "")
        .str.strip()
    )
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["cost"].fillna(df["cost"].median(), inplace=True)

    # Fix cuisine column
    df["cuisine"] = df["cuisine"].fillna("").astype(str)

    # Clean city column
    df["city"] = df["city"].fillna("Unknown")

    # Fill missing ratings
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)

    # Save cleaned file
    output_path = PROCESSED_DIR / "cleaned_data.csv"
    df.to_csv(output_path, index=False)

    print("✅ cleaned_data.csv saved at:", output_path)

if __name__ == "__main__":
    clean_data()
