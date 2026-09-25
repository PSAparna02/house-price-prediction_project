import numpy as np
import pandas as pd

np.random.seed(42)
n = 1200

locations = ['North Ridge', 'West End', 'Lakeside', 'Downtown', 'Green Valley', 'Old Town', 'Hillcrest']
loc_price_factor = {'North Ridge': 1.25, 'West End': 1.05, 'Lakeside': 1.4, 'Downtown': 1.5,
                     'Green Valley': 0.95, 'Old Town': 0.85, 'Hillcrest': 1.1}

df = pd.DataFrame({
    'house_id': range(1, n + 1),
    'location': np.random.choice(locations, n),
    'sqft': np.random.normal(1800, 650, n).clip(400, 6000).round(0),
    'bedrooms': np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.2, 0.35, 0.25, 0.1, 0.05]),
    'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5], n, p=[0.15, 0.15, 0.3, 0.2, 0.15, 0.05]),
    'age_years': np.random.randint(0, 80, n),
    'garage': np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.35, 0.4, 0.1]),
    'has_pool': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    'lot_size': np.random.normal(6500, 2500, n).clip(1000, 25000).round(0),
    'renovated': np.random.choice([0, 1], n, p=[0.75, 0.25]),
    'sale_year': np.random.choice([2022, 2023, 2024, 2025, 2026], n, p=[0.1, 0.15, 0.2, 0.25, 0.3]),
    'sale_month': np.random.randint(1, 13, n),
    'school_rating': np.random.randint(1, 11, n),
    'distance_to_city_km': np.random.exponential(8, n).clip(0.5, 60).round(1),
})

# base price model with noise (this is the "ground truth" signal a regression model should find)
base = (
    df['sqft'] * 120
    + df['bedrooms'] * 8000
    + df['bathrooms'] * 6000
    - df['age_years'] * 500
    + df['garage'] * 4000
    + df['has_pool'] * 15000
    + df['lot_size'] * 3
    + df['renovated'] * 12000
    + df['school_rating'] * 3000
    - df['distance_to_city_km'] * 900
    + (df['sale_year'] - 2022) * 6000
)
loc_mult = df['location'].map(loc_price_factor)
noise = np.random.normal(0, 18000, n)
df['price'] = (base * loc_mult + noise).clip(40000, None).round(0)

# --- intentionally dirty the data, mirroring "raw, messy data" from the masterclass ---
# missing values
for col in ['bathrooms', 'garage', 'lot_size', 'school_rating']:
    idx = np.random.choice(df.index, size=int(0.04 * n), replace=False)
    df.loc[idx, col] = np.nan

# a few duplicate rows
dup_idx = np.random.choice(df.index, size=8, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# inconsistent text casing / whitespace in location
messy_idx = np.random.choice(df.index, size=30, replace=False)
df.loc[messy_idx, 'location'] = df.loc[messy_idx, 'location'].str.upper() + '  '

# a few extreme outliers
out_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[out_idx, 'price'] = df.loc[out_idx, 'price'] * 6

# negative sqft typo (data entry errors)
err_idx = np.random.choice(df.index, size=3, replace=False)
df.loc[err_idx, 'sqft'] = -df.loc[err_idx, 'sqft']

df = df.sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv('house_price_raw.csv', index=False)
print(df.shape)
print(df.isna().sum())
