"""
House Price Intelligence Platform
==================================
A single-file Python (Streamlit) application that turns raw, messy house-sale
data into a business-intelligence platform: it cleans the data, surfaces the
key KPIs, explains trends/drivers/risks/opportunities, and lets a user get an
instant price prediction from a trained regression model.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

Structure (mirrors the BI hierarchy: KPIs -> Trends -> Drivers -> Risk/Opportunity -> Action):
    1) Executive Overview      - top KPIs & headline trend
    2) Price & Property Analysis - what drives price (drivers)
    3) Risk & Opportunity      - where the business should be careful / lean in
    4) Predict a Price         - the trained model, usable interactively
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Intelligence Platform", layout="wide")

DATA_PATH = "house_price_raw.csv"


# ---------------------------------------------------------------------------
# 1. DATA CLEANING
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize location text (strip whitespace, fix casing)
    df["location"] = df["location"].str.strip().str.title()

    # Drop exact duplicate rows
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "house_id"])

    # Fix negative sqft entry errors (data-entry typos)
    df["sqft"] = df["sqft"].abs()

    # Impute missing numeric values with the median (robust to outliers)
    for col in ["bathrooms", "garage", "lot_size", "school_rating"]:
        df[col] = df[col].fillna(df[col].median())

    # Remove statistically extreme price outliers (beyond 3 std devs) so they
    # don't distort KPIs or the model
    price_mean, price_std = df["price"].mean(), df["price"].std()
    df = df[(df["price"] - price_mean).abs() <= 3 * price_std]

    # Derived fields used throughout the app
    df["price_per_sqft"] = (df["price"] / df["sqft"]).round(2)
    df["sale_date"] = pd.to_datetime(
        dict(year=df["sale_year"], month=df["sale_month"], day=1)
    )

    return df.reset_index(drop=True)


@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [
        "sqft", "bedrooms", "bathrooms", "age_years", "garage", "has_pool",
        "lot_size", "renovated", "school_rating", "distance_to_city_km",
        "location",
    ]
    X = pd.get_dummies(df[features], columns=["location"], drop_first=True)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, X.columns.tolist(), mae, r2


df = load_and_clean_data(DATA_PATH)
model, model_columns, mae, r2 = train_model(df)

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 House Price BI Platform")
page = st.sidebar.radio(
    "Go to",
    ["Executive Overview", "Price & Property Analysis", "Risk & Opportunity", "Predict a Price"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Cleaned dataset: {len(df):,} records")
st.sidebar.caption(f"Model accuracy: R² = {r2:.2f}, avg error = ${mae:,.0f}")


# ---------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW  (KPIs + headline trend -> "what is happening")
# ---------------------------------------------------------------------------
if page == "Executive Overview":
    st.title("Executive Overview")
    st.caption("The top-line numbers a business owner needs at a glance.")

    # --- Top 3 KPIs (the masterclass explicitly recommends picking a small,
    #     high-signal set rather than showing everything) ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Sale Price", f"${df['price'].mean():,.0f}")
    col2.metric("Median Price / Sqft", f"${df['price_per_sqft'].median():,.2f}")
    yearly = df.groupby("sale_year")["price"].mean()
    yoy_growth = (yearly.iloc[-1] / yearly.iloc[-2] - 1) * 100 if len(yearly) > 1 else 0
    col3.metric("YoY Price Growth", f"{yoy_growth:+.1f}%")

    st.markdown("### Price Trend Over Time")
    trend = df.groupby("sale_date")["price"].mean().reset_index()
    fig = px.line(trend, x="sale_date", y="price", markers=True,
                   labels={"sale_date": "Sale Month", "price": "Avg Price ($)"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sales Volume by Location")
    vol = df["location"].value_counts().reset_index()
    vol.columns = ["location", "sales_count"]
    fig2 = px.bar(vol, x="location", y="sales_count",
                   labels={"location": "Location", "sales_count": "# of Sales"})
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 2: PRICE & PROPERTY ANALYSIS  ("drivers" - why prices move)
# ---------------------------------------------------------------------------
elif page == "Price & Property Analysis":
    st.title("Price & Property Analysis")
    st.caption("What actually drives a higher sale price.")

    st.markdown("### Average Price by Location")
    loc_price = df.groupby("location")["price"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(loc_price, x="location", y="price",
                  labels={"location": "Location", "price": "Avg Price ($)"})
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Price vs. Square Footage")
        fig2 = px.scatter(df, x="sqft", y="price", color="location", opacity=0.6,
                            labels={"sqft": "Square Feet", "price": "Price ($)"})
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown("### Price by Bedroom Count")
        fig3 = px.box(df, x="bedrooms", y="price",
                       labels={"bedrooms": "Bedrooms", "price": "Price ($)"})
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### What Drives Price? (Model Feature Importance)")
    importances = pd.Series(model.feature_importances_, index=model_columns)
    importances = importances.sort_values(ascending=False).head(10).reset_index()
    importances.columns = ["feature", "importance"]
    fig4 = px.bar(importances, x="importance", y="feature", orientation="h",
                   labels={"importance": "Relative Importance", "feature": "Feature"})
    st.plotly_chart(fig4, use_container_width=True)
    st.info(
        "Square footage, location, and school rating are typically the strongest "
        "price drivers — useful for guiding renovation or marketing investment."
    )


# ---------------------------------------------------------------------------
# PAGE 3: RISK & OPPORTUNITY  (-> recommended actions)
# ---------------------------------------------------------------------------
elif page == "Risk & Opportunity":
    st.title("Risk & Opportunity")
    st.caption("Where the business should be cautious, and where it should lean in.")

    old_homes = df[df["age_years"] > 50]
    far_homes = df[df["distance_to_city_km"] > 20]
    cheapest_loc = df.groupby("location")["price"].mean().idxmin()
    priciest_loc = df.groupby("location")["price"].mean().idxmax()

    st.markdown("### ⚠️ Risks")
    st.write(
        f"- **Aging inventory**: {len(old_homes):,} homes ({len(old_homes)/len(df):.0%} of "
        f"listings) are over 50 years old and average **${old_homes['price'].mean():,.0f}**, "
        f"below the portfolio average — renovation risk if left unaddressed."
    )
    st.write(
        f"- **Distance discount**: Homes more than 20km from the city center "
        f"average **${far_homes['price'].mean():,.0f}**, notably lower than closer homes "
        f"(**${df[df['distance_to_city_km'] <= 20]['price'].mean():,.0f}**)."
    )
    st.write(
        f"- **{cheapest_loc}** is consistently the lowest-priced area — a soft spot "
        f"in the portfolio worth investigating."
    )

    st.markdown("### 🌱 Opportunities")
    pool_premium = df[df["has_pool"] == 1]["price"].mean() - df[df["has_pool"] == 0]["price"].mean()
    reno_premium = df[df["renovated"] == 1]["price"].mean() - df[df["renovated"] == 0]["price"].mean()
    st.write(
        f"- **{priciest_loc}** commands the highest average price — a strong area "
        f"for targeted marketing or acquisition."
    )
    st.write(f"- Homes with a **pool** sell for **${pool_premium:,.0f} more** on average.")
    st.write(f"- **Renovated** homes sell for **${reno_premium:,.0f} more** on average — "
             f"renovation spend appears to pay back.")

    st.markdown("### ✅ Recommended Actions")
    st.success(
        "1. Prioritize renovation budget on homes over 50 years old in lower-priced areas.\n\n"
        f"2. Increase marketing spend in **{priciest_loc}**, where demand and price are strongest.\n\n"
        "3. Highlight pool and renovation status in listings — both carry a measurable price premium.\n\n"
        f"4. Investigate why **{cheapest_loc}** underperforms (school ratings, distance, inventory age)."
    )


# ---------------------------------------------------------------------------
# PAGE 4: PREDICT A PRICE  (interactive model)
# ---------------------------------------------------------------------------
elif page == "Predict a Price":
    st.title("Predict a Price")
    st.caption(f"Model: Random Forest Regressor — R² = {r2:.2f}, avg error ≈ ${mae:,.0f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        sqft = st.number_input("Square Feet", 400, 6000, 1800)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 3.5, 2.0, step=0.5)
    with c2:
        age_years = st.slider("Age (years)", 0, 80, 15)
        garage = st.slider("Garage Spaces", 0, 3, 1)
        lot_size = st.number_input("Lot Size (sqft)", 1000, 25000, 6500)
    with c3:
        location = st.selectbox("Location", sorted(df["location"].unique()))
        school_rating = st.slider("School Rating (1-10)", 1, 10, 6)
        distance = st.slider("Distance to City (km)", 0.5, 60.0, 8.0)

    has_pool = st.checkbox("Has Pool")
    renovated = st.checkbox("Recently Renovated")

    if st.button("Predict Price", type="primary"):
        row = pd.DataFrame([{
            "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "age_years": age_years, "garage": garage, "has_pool": int(has_pool),
            "lot_size": lot_size, "renovated": int(renovated),
            "school_rating": school_rating, "distance_to_city_km": distance,
            "location": location,
        }])
        row_encoded = pd.get_dummies(row, columns=["location"])
        row_encoded = row_encoded.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(row_encoded)[0]
        st.metric("Estimated Price", f"${prediction:,.0f}")
