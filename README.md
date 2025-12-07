# Swiggy Restaurant Recommendation System

A machine-learning–based restaurant recommendation system built using **Python, One-Hot Encoding, Cosine Similarity, and Streamlit**.

---

## 📌 Project Overview

This project recommends restaurants to users based on:

- City  
- Cuisine preference  
- Minimum rating  
- Maximum cost  
- Number of recommendations  

The system performs:

- Data Cleaning  
- Data Preprocessing (One-Hot Encoding)  
- Similarity-based Recommendation  
- Streamlit Web Application  

---

## 🧠 Skills Used

- Data Preprocessing  
- One-Hot Encoding  
- Cosine Similarity  
- Python  
- Streamlit  
- Pandas, Scikit-learn  

---

## 📂 Project Structure

![Untitled - Frame 1 (4)](https://github.com/user-attachments/assets/b306c2fe-d745-4185-b227-ba7c445bf9f3)


---

## 🧹 Data Cleaning

- Removes duplicates
- Converts cost to numeric
- Handles missing values
- Saves processed output to **cleaned_data.csv**

---

## 🔧 Data Preprocessing

- One-Hot Encodes **city**
- One-Hot Encodes **cuisine**
- Combines numerical features
- Saves outputs:
  - `encoded_data.csv`
  - `city_encoder.pkl`
  - `cuisine_encoder.pkl`

---

## 🤖 Recommendation Engine

Uses **Cosine Similarity** to:

1. Convert user input to encoded vector  
2. Calculate similarities with restaurants  
3. Filter by rating & cost  
4. Return top N recommendations  

---

## 🌐 Streamlit App

Provides:

- City selection  
- Cuisine selection  
- Rating and cost filters  
- Number of recommendations slider  
- Displays:
  - Restaurant name
  - City
  - Cuisine
  - Rating
  - Cost
  - Address
  - Swiggy link


---

## 📊 Results

- Clean, consistent dataset  
- Fully encoded dataset for ML  
- Accurate recommendations  
- Fully functional Streamlit UI  

---

## 📈 Evaluation Metrics

- Recommendation relevance  
- Speed & performance  
- UX quality  
- Clean–encoded dataset alignment  

---

## 📘 Report Summary

### Data Cleaning  
Removed duplicates, fixed missing values, normalized cost column.

### Preprocessing  
Applied One-Hot Encoding to city & cuisine; combined with numerical features.

### Recommendation Logic  
Cosine similarity used for preference-based ranking.

### Key Insights  
Cuisine similarity strongly impacts ranking; cost and rating filters enhance precision.

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  

---

# 👨‍💻 Developer
 
  Nirudeeswar R
 
 📍 Chennai
 
 🎓 B.Tech CSE
 
 📧 nirudeeswarr14@gmail.com
---

