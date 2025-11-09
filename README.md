# 🏠 Singapore Private Housing Transaction Prediction

This repository contains a complete **machine learning pipeline** for predicting **private housing transaction prices in Singapore**, developed as part of the **DSS5105 Quantitative Data Science Project** at NUS.  
The project focuses on generating interpretable, data-driven insights for property market analysis, policy decisions, and buyer awareness.

---

## 📘 Project Overview

### 🎯 Objective
The goal of this project is to build a **robust and explainable machine learning model** to analyze Singapore’s private housing transactions and uncover key factors driving property prices.

**Specifically, the project aims to:**
1. Identify the **Top 3 Key Factors** influencing private property prices  
2. Explore **Patterns & Trends** across locations, property types, and time  
3. Provide **Buyer Insights** for informed decision-making  

---

## 🧩 Project Structure

| Section | Description |
|----------|-------------|
| **1. Environment Setup and Data Loading** | Import required packages and load transaction datasets from URA / public APIs. |
| **2. Exploratory Data Analysis (EDA)** | Examine feature distributions, correlations, outliers, and geospatial trends. |
| **3. Data Preprocessing & Feature Engineering** | Handle missing values, encode categorical features, generate time-based and geospatial features. |
| **4. Model Development & Training** | Implement baseline models (Linear Regression, Random Forest, LightGBM) and tune hyperparameters. |
| **5. Model Evaluation & Explainability** | Evaluate models using RMSE, MAE, and R²; apply **SHAP** for interpretability. |
| **6. Visualization & Insights** | Visualize predicted vs actual prices, feature importance, and location-wise patterns. |

---

## ⚙️ Environment Setup

### Requirements
- Python ≥ 3.9  
- Jupyter Notebook  
- Common ML libraries:  

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn geopandas
```

---

## 📂 Data Description

The dataset used in this project consists of Singapore private property transactions, including:
- **Project Name / Street / Postal Code**
- **Property Type (Condo, Apartment, etc.)**
- **Floor Area, Price, PSF, Lease Period**
- **Transaction Date**
- **Latitude / Longitude (for mapping)**

Data sources include:
- Urban Redevelopment Authority (URA) open data
- Data.gov.sg public datasets

---

## 🧠 Models Implemented

| Model | Purpose |
|--------|----------|
| **Linear Regression** | Establish a baseline predictive model |
| **Random Forest Regressor** | Capture nonlinear relationships between price and features |
| **LightGBM Regressor** | Achieve strong predictive performance with optimized hyperparameters |
| **SHAP Analysis** | Explain model outputs and identify top drivers of price variation |

---

## 📈 Results Summary

- **Best Model:** LightGBM with cross-validation  
- **Top 3 Factors:** Location (postal district), Floor Area, Lease Remaining  
- **R² Score:** ~0.90 on validation data  
- **Interpretability:** SHAP summary plots reveal how each feature impacts price

---

## 🗺️ Visual Insights

The notebook provides visualizations such as:
- Price distribution by region and property type  
- Temporal trends in price per square foot (PSF)  
- SHAP feature importance summary and dependence plots  

Example:
```python
shap.summary_plot(shap_values, X_test)
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/housing-prediction.git
   cd housing-prediction
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open and run the `notebook.ipynb` file step by step.

---

## 📊 Future Improvements

- Integrate **geospatial clustering** (DBSCAN / HDBSCAN) for neighborhood segmentation  
- Add **temporal forecasting** using LSTM or Prophet  
- Develop a **Streamlit dashboard** for interactive visualization  

---

## 📜 License
This project is released under the [MIT License](LICENSE).
