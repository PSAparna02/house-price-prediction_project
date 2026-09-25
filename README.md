# House Price Intelligence Platform

A business-intelligence project that turns raw, messy house-sale data into
clear KPIs, trends, drivers, risks, opportunities, and an interactive price
prediction model — built with Python and Streamlit.

## Problem Statement

Real estate businesses collect large volumes of raw sale data but rarely turn
it into decisions. This project cleans a raw house-sale dataset, surfaces the
handful of KPIs that matter most, explains *why* prices move, flags risks and
opportunities, and gives a working price-prediction tool — following a
business-intelligence flow of **Data → KPIs → Trends → Drivers → Risk/Opportunity → Action**.

## Dataset

`house_price_raw.csv` — 1,208 synthetic house-sale records including square
footage, bedrooms/bathrooms, age, lot size, location, school rating, distance
to city center, sale date, and sale price. The raw file intentionally contains
realistic messiness (missing values, duplicate rows, inconsistent text
casing, outliers, and a data-entry typo) so the cleaning step is meaningful.

> Data source: synthetically generated for this project (see
> `generate_data.py`). If you swap in a real dataset (e.g. from Kaggle), keep
> the same column names or update `app.py` accordingly, and note the dataset's
> source link in your project report.

## Project Structure

```
├── app.py                 # Single-file Streamlit app (cleaning, KPIs, EDA, model)
├── generate_data.py        # Script used to generate the sample dataset
├── house_price_raw.csv     # Raw sample dataset
├── requirements.txt        # Python dependencies
└── README.md
```

## How It Works

1. **Data Cleaning** — strips/standardizes text, removes duplicates, fixes a
   negative-value data-entry error, imputes missing values with the median,
   and removes extreme statistical outliers.
2. **Executive Overview (KPIs)** — average price, median price/sqft, and
   year-over-year growth, plus the headline price trend over time.
3. **Price & Property Analysis (Drivers)** — which features (location, sqft,
   school rating, etc.) actually move the price, backed by a trained model's
   feature importances.
4. **Risk & Opportunity** — aging inventory, distance-to-city discount,
   pool/renovation premiums, and concrete recommended actions.
5. **Predict a Price** — a Random Forest Regression model (R² ≈ 0.90 on this
   sample data) that gives an instant price estimate for a hypothetical
   property.

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Model

- **Algorithm:** Random Forest Regressor (scikit-learn)
- **Features:** square footage, bedrooms, bathrooms, age, garage spaces,
  pool, lot size, renovation status, school rating, distance to city,
  location (one-hot encoded)
- **Performance on sample data:** R² ≈ 0.90, average error ≈ $25,000

## Author's Note

This project was built as part of the BharatCares AI/Data Analytics
masterclass project assignment, following the business-intelligence workflow
(clean → explore → analyze → visualize → decide → act) covered in the course.
