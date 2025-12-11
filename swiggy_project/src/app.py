import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ------------------ FILE PATHS ------------------
BASE = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE / "data" / "processed" / "cleaned_data.csv"
ENCODED_PATH = BASE / "data" / "processed" / "encoded_data.csv"
CITY_ENCODER_PATH = BASE / "data" / "processed" / "city_encoder.pkl"
KMEANS_PATH = BASE / "data" / "processed" / "kmeans.pkl"


# ============================================================
#                LOAD ALL NECESSARY FILES
# ============================================================

@st.cache_resource
def load_all_data():
    clean_df = pd.read_csv(CLEAN_PATH)
    encoded_df = pd.read_csv(ENCODED_PATH)

    # Load encoders & model
    with open(CITY_ENCODER_PATH, "rb") as f:
        city_enc = pickle.load(f)
    with open(KMEANS_PATH, "rb") as f:
        kmeans = pickle.load(f)

    # Extract all cuisines from cleaned df
    cuisine_set = set()
    for row in clean_df["cuisine"]:
        if isinstance(row, str):
            for c in row.split(","):
                cuisine_set.add(c.strip().lower())

    cuisine_list = sorted(list(cuisine_set))

    # Determine cuisine columns in encoded_df
    city_feature_names = list(city_enc.get_feature_names_out(["city"]))
    non_cuisine_cols = set(city_feature_names + ["rating", "rating_count", "cost"])
    cuisine_cols = [c for c in encoded_df.columns if c not in non_cuisine_cols]

    return clean_df, encoded_df, city_enc, cuisine_list, cuisine_cols, kmeans


clean_df, encoded_df, city_enc, cuisine_unique, cuisine_cols, kmeans = load_all_data()


# ============================================================
#            BUILD USER VECTOR (MUST MATCH ENCODED ORDER)
# ============================================================

def build_user_vector(city: str, cuisine: str,
                      min_rating: float, max_cost: float) -> np.ndarray:
    """
    Build user feature vector in the same order as encoded_data.csv:
    [ city one-hot | cuisine multi-hot | rating | rating_count | cost ]
    """

    # 1) City one-hot
    user_city_enc = city_enc.transform([[city]])  # shape: (1, n_city_features)
    user_city_enc = user_city_enc[0]  # → 1D

    # 2) Cuisine multi-hot aligned to cuisine_cols
    user_cuisine_vec = np.zeros(len(cuisine_cols))
    cuisine_lower = cuisine.lower().strip()

    for i, col in enumerate(cuisine_cols):
        if cuisine_lower == col.lower():
            user_cuisine_vec[i] = 1

    # 3) Numeric features
    # rating_count is not meaningful for user vector → set 0
    user_numeric = np.array([min_rating, 0.0, max_cost])

    # 4) Concatenate: [city | cuisines | rating | rating_count | cost]
    user_vec = np.concatenate([user_city_enc, user_cuisine_vec, user_numeric])
    return user_vec.reshape(1, -1)


# ============================================================
#                     RECOMMENDATION LOGIC
# ============================================================

def recommend(city: str, cuisine: str,
              min_rating: float, max_cost: float,
              top_n: int = 5):
    """
    1. Build user vector & get its cluster from KMeans
    2. Filter restaurants by user's city, cuisine, rating, cost
    3. Among filtered ones, rank by:
       - same cluster as user (from KMeans)
       - then by rating (desc)
    4. If no restaurant passes filters → fallback to top rated from the user's cluster
    """

    # ---------- 1) USER CLUSTER VIA KMEANS ----------
    user_vec = build_user_vector(city, cuisine, min_rating, max_cost)
    user_cluster = int(kmeans.predict(user_vec)[0])

    # ---------- 2) FILTER BY USER PREFERENCES ----------
    df = clean_df.copy()

    # Strong filters on real-world fields
    mask = (
        (df["city"].str.lower() == city.lower())
        & (df["cuisine"].str.contains(cuisine, case=False, na=False))
        & (df["rating"] >= min_rating)
        & (df["cost"] <= max_cost)
    )

    filtered = df[mask].copy()

    # ---------- 3) IF WE HAVE MATCHES → RANK USING CLUSTER ----------
    if not filtered.empty:
        # Mark if restaurant's cluster matches user's cluster
        if "cluster" in filtered.columns:
            filtered["same_cluster"] = (filtered["cluster"] == user_cluster).astype(int)
        else:
            # just in case cluster wasn't added (safety)
            filtered["same_cluster"] = 0

        # Sort: first same cluster, then by rating
        filtered = filtered.sort_values(
            ["same_cluster", "rating"],
            ascending=[False, False]
        )

        return filtered.head(top_n), False  # False = not a fallback

    # ---------- 4) FALLBACK: USE USER CLUSTER ----------
    if "cluster" in df.columns:
        fallback = df[df["cluster"] == user_cluster].copy()
    else:
        fallback = df.copy()

    if fallback.empty:
        return fallback, True  # nothing to show

    fallback = fallback.sort_values("rating", ascending=False).head(top_n)
    return fallback, True  # True = we used fallback


# ============================================================
#                        STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Swiggy KMeans Recommender", layout="wide")

# Reduce font size a bit for overall app
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main heading with new logo
st.markdown(
    "<h1 style='text-align: center;'>🍴 Swiggy Restaurant Recommendation System (K-Means)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h4 style='text-align: center;'>Find the best restaurants based on your city, cuisine, rating and budget.</h4>",
    unsafe_allow_html=True,
)

st.sidebar.header("🔍 Search Filters")

# ------------------ FILTERS ------------------
city_options = ["-- Select City --"] + sorted(clean_df["city"].unique().tolist())
city = st.sidebar.selectbox("Select City", city_options)

cuisine_options = ["-- Select Cuisine --"] + cuisine_unique
cuisine = st.sidebar.selectbox("Select Cuisine", cuisine_options)

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
max_cost = st.sidebar.slider("Maximum Cost for Two (₹)", 100, 2000, 2000, 50)

top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

submit = st.sidebar.button("🔎 Get Recommendations")


# ------------------ RESULTS AREA ------------------
col_center = st.container()

with col_center:
    if submit:
        if city == "-- Select City --":
            st.error("❌ Please select a city.")
        elif cuisine == "-- Select Cuisine --":
            st.error("❌ Please select a cuisine.")
        else:
            results, used_fallback = recommend(
                city=city,
                cuisine=cuisine,
                min_rating=min_rating,
                max_cost=max_cost,
                top_n=top_n,
            )

            st.markdown("## 🍴 Recommended Restaurants")

            if results.empty:
                st.warning("⚠ No restaurants found even with fallback. Try relaxing filters.")
            else:
                if used_fallback:
                    st.warning(
                        "⚠ No restaurant exactly matched all your filters. "
                        "Showing the best suggestions from the closest K-Means cluster."
                    )
                else:
                    st.success("✅ Showing restaurants that match your filters.")

                # Show cards
                for _, row in results.iterrows():
                    name = row.get("name", "Unknown")
                    city_val = row.get("city", "Unknown")
                    cuisine_val = row.get("cuisine", "Unknown")
                    rating_val = row.get("rating", "N/A")
                    cost_val = row.get("cost", "N/A")
                    address = row.get("address", "")
                    link = row.get("link", "")

                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin-bottom:10px;">
                            <h3>🍛 {name}</h3>
                            <p><b>City:</b> {city_val}</p>
                            <p><b>Cuisine:</b> {cuisine_val}</p>
                            <p><b>Rating:</b> ⭐ {rating_val}</p>
                            <p><b>Cost for Two:</b> ₹{cost_val}</p>
                            <p><b>Address:</b> {address}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if isinstance(link, str) and link.strip():
                        st.markdown(f"[🔗 View on Swiggy]({link})")
                    st.markdown("---")
    else:
        st.info("👈 Choose your city, cuisine, rating and budget from the left, then click **Get Recommendations**.")
