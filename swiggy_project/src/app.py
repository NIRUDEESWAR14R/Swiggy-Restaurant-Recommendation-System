import streamlit as st
st.set_page_config(page_title="Swiggy Recommendation System", layout="centered")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------
# PATHS
# ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"

# ---------------------------------------
# LOAD & PREPARE DATA
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN_PATH)

    df["city"] = df["city"].fillna("Unknown").astype(str)
    df["cuisine"] = df["cuisine"].fillna("Unknown").astype(str)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(0)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(df["cost"].median())

    features = df[["rating", "rating_count", "cost"]].copy()
    features["cost"] = -features["cost"]

    feature_matrix = features.to_numpy()
    return df, feature_matrix

df, feature_matrix = load_data()

# ---------------------------------------
# RECOMMENDATION LOGIC
# ---------------------------------------
def recommend(city, cuisine, min_rating, max_cost, top_n):
    cuisine_keyword = cuisine.split(",")[0].strip()

    mask = (
        (df["city"] == city) &
        (df["rating"] >= min_rating) &
        (df["cost"] <= max_cost) &
        (df["cuisine"].str.contains(cuisine_keyword, case=False, na=False))
    )

    filtered = df[mask]
    if filtered.empty:
        return filtered

    cand_matrix = feature_matrix[filtered.index]

    user_vec = np.array([[min_rating,
                          filtered["rating_count"].mean(),
                          -max_cost]])

    sim = cosine_similarity(user_vec, cand_matrix)[0]
    filtered = filtered.copy()
    filtered["similarity"] = sim

    return filtered.sort_values("similarity", ascending=False).head(top_n)

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.title("🍽️ Swiggy Restaurant Recommendation System")
st.write("Find restaurants based on your taste, location, and budget.")

st.sidebar.header("🔍 Filter Options")

cities = ["Select City"] + sorted(df["city"].unique())
cuisines = ["Select Cuisine"] + sorted(df["cuisine"].unique())

city = st.sidebar.selectbox("Select City", cities, index=0)
cuisine = st.sidebar.selectbox("Preferred Cuisine", cuisines, index=0)

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0, 0.1)
max_cost = st.sidebar.slider("Maximum Cost (₹)", 50, 2000, 500, 50)
top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 5)

clicked = st.sidebar.button("Get Recommendations 🚀")

if clicked:
    if city == "Select City" or cuisine == "Select Cuisine":
        st.warning("Please select a valid city and cuisine.")
    else:
        results = recommend(city, cuisine, min_rating, max_cost, top_n)

        if results.empty:
            st.error("No matching restaurants found. Try changing filters.")
        else:
            st.subheader(f"Top {len(results)} Recommendations")
            for _, row in results.iterrows():
                st.markdown(
                    f"""
                    ### 🍛 {row['name']}
                    **City:** {row['city']}  
                    **Cuisine:** {row['cuisine']}  
                    **Rating:** ⭐ {row['rating']} ({row['rating_count']} reviews)  
                    **Cost for Two:** ₹{row['cost']}  

                    **Address:** {row['address']}  

                    🔗 [View on Swiggy]({row['link']})
                    ---
                    """
                )
else:
    st.info("Use the filters on the left and click **Get Recommendations 🚀**.")
