---
title: "Used Car Price Prediction"
emoji: "🚗"
colorFrom: "blue"
colorTo: "green"
sdk: "streamlit"
sdk_version: "1.32.0"
app_file: "streamlit_app.py"
pinned: false
---

# 🚗 AI Used Car Price Prediction

This is a full pipeline project to **scrape used car listings** from [Mobil123.com](https://www.mobil123.com), preprocess the data, analyze **price depreciation**, and build a **generative AI price prediction model** with a simple **Streamlit interface**.

## 🧪 Features

✅ Scraping used car data (brand, model, year, engine, price, etc.)  
✅ Data cleaning and feature engineering  
✅ Depreciation analysis over the past 20 years  
✅ AI-powered price prediction with uncertainty estimation  
✅ Streamlit UI for interactive prediction & visualization  
✅ CLI mode for offline use

---

## 📁 Project Structure

```
.
├── data/
│   └── mobil123_data.csv          # Scraped car listing data
├── models/
│   └── car_price_model.pkl        # Trained ML model
├── scraper.py                     # Scraper for Mobil123
├── car_price_model.py             # Core model logic & pipeline
├── streamlit_app.py               # Streamlit app interface
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Scrape Data

```bash
python scraper.py
```

### 3. Train the Model

```bash
python car_price_model.py
```

### 4. Run the App

```bash
streamlit run streamlit_app.py
```

---

## 📊 App Features

- **Price Prediction** with confidence range and probable price samples
- **Depreciation Projection** for up to 15 years
- **Market Comparison** to similar listings
- **Data Insights** across brand, year, and historical value

---

## ⚠️ Notes

- Kilometer (odometer) was excluded due to inconsistent scraping
- Depreciation trends are computed from 20-year market median prices
- The prediction includes uncertainty estimation using probabilistic modeling