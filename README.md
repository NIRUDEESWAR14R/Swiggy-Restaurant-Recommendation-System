# 🍽️ Swiggy Restaurant Recommendation System

This project is a **Restaurant Recommendation System** built using  
**Machine Learning (K-Means Clustering)** and deployed through a  
**Streamlit web application**.  

It recommends restaurants based on the user's selected **city**, **cuisine**, **rating**, and **budget**.  
K-Means is used to group similar restaurants and improve recommendation quality by ranking results using cluster similarity.

---

## 🚀 Features

### ✅ **ML-Based Recommendations**
- Uses **K-Means clustering** to group similar restaurants.
- Ranks restaurants based on:
  - City & cuisine match
  - Rating >= user threshold
  - Cost <= user budget
  - Cluster similarity (same-cluster restaurants prioritized)

### ✅ **Streamlit Web App**
- Clean UI with:
  - Sidebar filters
  - Restaurant cards
  - Direct Swiggy links
- No pre-filled selections; user fully controls filters.

### ✅ **Efficient Data Processing**
- One-hot encoded cities
- Multi-hot encoded cuisines
- Handles missing values automatically
- Optimized MiniBatchKMeans for fast training (148K+ restaurants)

---
## FLOW

### 🧹 Step 1: Preprocess Data

Creates:

 - cleaned_data.csv

 - encoded_data.csv

 - city_encoder.pkl

 - python src/preprocess.py

### ⚡ Step 2: Train K-Means

Creates:

 - kmeans.pkl

 - cleaned_data.csv (with cluster labels)

 - python src/train_kmeans.py

### 🤖 Step 3: Test Recommendation Logic (Optional)
python src/recommendation.py

### 🌐 Step 4: Run the Streamlit App
streamlit run src/app.py

---

## 🧠 Machine Learning Used

 - K-Means Clustering

 - Groups restaurants based on:

 - city encoding

 - cuisine encoding

 - rating

 - rating_count

 - cost

---

### Why K-Means?

 - No labelled data required

 - Fast & scalable for 100K+ rows

 - Ideal for grouping similar restaurants

---

### 📊 Dataset

The dataset contains 148,541 restaurants with fields:

 - Name

 - City

 - Cuisine

 - Rating

 - Rating Count

 - Cost for Two

 - Address

 - Swiggy Link

---

### 🎯 Future Enhancements

 - Add collaborative filtering

 - Hybrid recommender (content + clustering + cosine)

 - User login & preference history

---

# 👨‍💻 Developer
 
  Nirudeeswar R
 
 📍 Chennai
 
 🎓 B.Tech CSE
 
 📧 nirudeeswarr14@gmail.com
---

