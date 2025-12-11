import pandas as pd
import pickle
from sklearn.cluster import MiniBatchKMeans

ENCODED_PATH = "data/processed/encoded_data.csv"
CLEAN_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "data/processed/kmeans.pkl"

print("\n🤖 Loading encoded data...")
df = pd.read_csv(ENCODED_PATH)
print("Encoded shape:", df.shape)

# Ensure no NaN in encoded data
df.fillna(0, inplace=True)

k = 25  # Good for 1.5L rows

print(f"⚡ Training MiniBatchKMeans ({k} clusters)...")

kmeans = MiniBatchKMeans(
    n_clusters=k,
    batch_size=2048,
    max_iter=120,
    random_state=42
)

cluster_labels = kmeans.fit_predict(df)

pickle.dump(kmeans, open(MODEL_PATH, "wb"))
print(f"✅ K-Means model saved at: {MODEL_PATH}")

# Save cluster labels to cleaned_data.csv
clean_df = pd.read_csv(CLEAN_PATH)
clean_df["cluster"] = cluster_labels
clean_df.to_csv(CLEAN_PATH, index=False)

print("✅ Cluster labels added to cleaned_data.csv")
print("🎉 Training completed successfully!")
