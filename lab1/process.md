---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Lab 1


Importing packages and data

```python
!pip3 install pandas seaborn numpy
import numpy as np
import pandas as pd
import seaborn as sns
```

# Task 1

```python
raw_city = pd.read_csv("data/city_day.csv")
raw_crop = pd.read_csv("data/crop_production.csv")
```

Getting a feel for the data by printing first 5 rows.

```python
raw_city.head()
```

```python
raw_crop.head()
```

```python
raw_city.shape
```

```python
raw_crop.shape
```

Finding the column data types.

```python
raw_city.info()
```

```python
raw_crop.info()
```

Finding statistical mean, standard deviation, median and quartiles, max and min.

```python
raw_city.describe()
```

```python
raw_crop.describe()
```

Finding the number of missing values in each column of the dataset, if any.

```python
raw_city.isnull().sum()
```

```python
raw_crop.isnull().sum()
```

Checking for duplicate rows.

```python
raw_city.duplicated().sum()
```

```python
raw_crop.duplicated().sum()
```

Checking unique values in each column.

```python
raw_city.nunique()
```

```python
raw_crop.nunique()
```

### Data Profile: city_day.csv

| Property | Value |
|---|---|
| Rows × Columns | 29,531 × 16 |
| Numeric columns | 13 (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI — all float64) |
| Categorical columns | 3 (City, Date, AQI_Bucket — all object) |
| Date range | 2015-01-01 to 2020-07-01 |
| Cities | 26 |
| AQI Buckets | 6 |
| Duplicate rows | 0 |
| Memory usage | ~3.6 MB |

**Missing values:**

| Column | Missing | % Missing |
|---|---|---|
| Xylene | 18,109 | 61.3% |
| PM10 | 11,140 | 37.7% |
| NH3 | 10,328 | 35.0% |
| Toluene | 8,041 | 27.2% |
| Benzene | 5,623 | 19.0% |
| AQI / AQI_Bucket | 4,681 | 15.9% |
| PM2.5 | 4,598 | 15.6% |
| NOx | 4,185 | 14.2% |
| O3 | 4,022 | 13.6% |
| SO2 | 3,854 | 13.1% |
| NO / NO2 | ~3,583 | ~12.1% |
| CO | 2,059 | 7.0% |


### Data Profile: crop_production.csv

| Property | Value |
|---|---|
| Rows × Columns | 246,091 × 7 |
| Numeric columns | 3 (Crop_Year int64, Area float64, Production float64) |
| Categorical columns | 4 (State_Name, District_Name, Season, Crop — all object) |
| Year range | 1997–2015 |
| States | 33 |
| Districts | 646 |
| Crops | 124 |
| Seasons | 6 |
| Duplicate rows | 0 |
| Memory usage | ~13.1 MB |

**Missing values:**

| Column | Missing | % Missing |
|---|---|---|
| Production | 3,730 | 1.5% |
| All others | 0 | 0% |


### Observations and Concerns:
- **city_day.csv** has heavy missing data across most pollutant columns. Xylene is 61% missing and is likely unusable as a feature without heavy imputation. PM10 (37.7%) and NH3 (35%) are also problematic.
- The `Date` column is stored as `object` instead of `datetime`, which needs conversion before any time-based analysis.
- **crop_production.csv** has extreme outliers in Production — max is 1.25 billion against a median of 729, suggesting the column may mix different units or scales across crops.
- Production has 3,730 missing values (1.5%). Since Production is likely the target variable for modelling, rows with missing Production may need to be dropped.
- No units are documented for Area or Production, making it unclear what is being measured.


## Task 2


Creating working copies to preserve raw data.

```python
city = raw_city.copy()
crop = raw_crop.copy()
```

Calculating the percentage of missing values in each column.

```python
city_missing = (city.isnull().sum() / len(city) * 100).round(1)
city_missing = city_missing[city_missing > 0].sort_values(ascending=False)
city_missing
```

```python
crop_missing = (crop.isnull().sum() / len(crop) * 100).round(1)
crop_missing = crop_missing[crop_missing > 0].sort_values(ascending=False)
crop_missing
```

Plotting missing value percentages with strategy thresholds.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
city_missing.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.axhline(y=5, color='green', linestyle='--', label='5% (CCA threshold)')
ax.axhline(y=10, color='orange', linestyle='--', label='10% (Random sample threshold)')
ax.axhline(y=50, color='red', linestyle='--', label='50% (Drop column threshold)')
aA.set_ylabel('% Missing')
ax.set_title('Missing Value Percentages - city_day.csv')
ax.legend()
plt.tight_layout()
plt.show()
```

```python
fig, ax = plt.subplots(figsize=(4, 3))
crop_missing.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.axhline(y=5, color='green', linestyle='--', label='5% (CCA threshold)')
ax.set_ylabel('% Missing')
ax.set_title('Missing Value Percentages - crop_production.csv')
ax.legend()
plt.tight_layout()
plt.show()
```

### Missing Value Treatment Strategy

**Approach based on percentage of missing values:**
- **< 5%** - Complete Case Analysis (drop affected rows), minimal data loss
- **5-10%** - Median imputation (preferred over mean for skewed data, as mean is sensitive to outliers)
- **10-50%** - Random sample imputation (preserves original distribution shape)
- **> 50%** - Drop the column (more data is missing than present, imputation unreliable)

#### crop_production.csv

| Column | % Missing | Decision | Reason |
|---|---|---|---|
| Production | 1.5% | Drop rows (CCA) | Below 5%. Likely the target variable - imputing it would introduce noise in the outcome |

#### city_day.csv

| Column | % Missing | Decision | Reason |
|---|---|---|---|
| Xylene | 61.3% | Drop column | Over 50% missing - any imputation would fabricate more data than it preserves |
| PM10 | 37.7% | Random sample | 10-50% range - random sampling preserves distribution without shifting mean |
| NH3 | 35.0% | Random sample | Same as above |
| Toluene | 27.2% | Random sample | Same as above |
| Benzene | 19.0% | Random sample | Same as above |
| AQI | 15.9% | Random sample | Same as above |
| AQI_Bucket | 15.9% | Random sample | Categorical - random sampling maintains class proportions |
| PM2.5 | 15.6% | Random sample | Same as above |
| NOx | 14.2% | Random sample | Same as above |
| O3 | 13.6% | Random sample | Same as above |
| SO2 | 13.1% | Random sample | Same as above |
| NO | 12.1% | Random sample | Same as above |
| NO2 | 12.1% | Random sample | Same as above |
| CO | 7.0% | Median imputation | 5-10% range. CO is right-skewed (mean=2.25 vs median=0.89) - median is more robust than mean |


Recording null counts before treatment.

```python
before_city = city.isnull().sum()
before_crop = crop.isnull().sum()
```

Applying CCA to crop_production - dropping rows where Production is null (1.5%).

```python
crop.dropna(subset=['Production'], inplace=True)
n_dropped = before_crop['Production']
print(f"Dropped {n_dropped} rows. New shape: {crop.shape}")
```

Dropping Xylene column from city_day - 61.3% missing, imputation unreliable.

```python
city.drop(columns=['Xylene'], inplace=True)
print(f"Dropped Xylene column. New shape: {city.shape}")
```

Applying median imputation to CO (7.0%). Using median over mean as CO is right-skewed.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

median_co = city['CO'].median()
sns.histplot(city['CO'].dropna(), kde=True, ax=axes[0], color='steelblue')
axes[0].axvline(median_co, color='red', linestyle='--', label=f'Median: {median_co:.2f}')
axes[0].set_title('CO - Before Imputation')
axes[0].legend()

city['CO'] = city['CO'].fillna(median_co)

sns.histplot(city['CO'], kde=True, ax=axes[1], color='coral')
axes[1].axvline(median_co, color='red', linestyle='--', label=f'Median: {median_co:.2f}')
axes[1].set_title('CO - After Median Imputation')
axes[1].legend()

plt.tight_layout()
plt.show()
```

Applying random sample imputation to remaining columns (>10% missing).

```python
def random_sample_impute(df, col):
    non_null = df[col].dropna()
    null_idx = df[df[col].isnull()].index
    samples = non_null.sample(len(null_idx), replace=True, random_state=42)
    samples.index = null_idx
    df.loc[null_idx, col] = samples

cols_to_impute = list(city.columns[city.isnull().any()])
before_dist = {col: city[col].dropna().copy() for col in cols_to_impute}

for col in cols_to_impute:
    random_sample_impute(city, col)

print(f"Imputed {len(cols_to_impute)} columns: {cols_to_impute}")
```

Plotting distributions before and after random sample imputation for numeric columns.

```python
numeric_cols = [c for c in cols_to_impute if city[c].dtype != 'object']
n = len(numeric_cols)
nrows = (n + 2) // 3

fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.kdeplot(before_dist[col], ax=axes[i], label='Before', color='steelblue')
    sns.kdeplot(city[col], ax=axes[i], label='After', color='coral', linestyle='--')
    axes[i].set_title(col)
    axes[i].legend()

for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Random Sample Imputation - Before vs After', y=1.02)
plt.tight_layout()
plt.show()
```

```python
if 'AQI_Bucket' in cols_to_impute:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    before_dist['AQI_Bucket'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('AQI_Bucket - Before')
    axes[0].tick_params(axis='x', rotation=45)
    city['AQI_Bucket'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('AQI_Bucket - After')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
```

Verifying null counts after treatment.

```python
after_city = city.isnull().sum()
after_crop = crop.isnull().sum()

print("=== crop_production - Before vs After ===")
print(f"Production: {before_crop['Production']} -> {after_crop['Production']}")
print(f"Rows: {len(raw_crop)} -> {len(crop)}")
print()

print("=== city_day - Before vs After ===")
for col in before_city[before_city > 0].index:
    if col in after_city.index:
        print(f"{col}: {before_city[col]} -> {after_city[col]}")
    else:
        print(f"{col}: {before_city[col]} -> [column dropped]")
```

```python
common_cols = [c for c in before_city[before_city > 0].index if c in after_city.index]
before_vals = [before_city[c] for c in common_cols]
after_vals = [after_city[c] for c in common_cols]

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(common_cols))
width = 0.35
ax.bar([i - width/2 for i in x], before_vals, width, label='Before', color='steelblue')
ax.bar([i + width/2 for i in x], after_vals, width, label='After', color='coral')
ax.set_xticks(list(x))
ax.set_xticklabels(common_cols, rotation=45, ha='right')
ax.set_ylabel('Null Count')
ax.set_title('Null Counts Before vs After Treatment - city_day.csv')
ax.legend()
plt.tight_layout()
plt.show()
```

## Task 3


Checking for inconsistencies in string columns that would break a merge on State.


Recording row counts before cleaning.

```python
print(f"city: {city.shape}")
print(f"crop: {crop.shape}")
```

Checking for leading/trailing whitespace in all string columns.

```python
for col in city.select_dtypes(include='object').columns:
    n = (city[col].dropna() != city[col].dropna().str.strip()).sum()
    if n > 0:
        print(f"{col}: {n} values with extra spaces")
        print(f"  Examples: {city[city[col].dropna() != city[col].dropna().str.strip()][col].unique()[:5]}")
    else:
        print(f"{col}: clean")
```

```python
for col in crop.select_dtypes(include='object').columns:
    n = (crop[col] != crop[col].str.strip()).sum()
    if n > 0:
        print(f"{col}: {n} values with extra spaces")
        print(f"  Examples: {crop[crop[col] != crop[col].str.strip()][col].unique()[:5]}")
    else:
        print(f"{col}: clean")
```

Listing unique values in State_Name and Season to spot spelling variants.

```python
print("=== State_Name unique values (showing repr to expose spaces) ===")
for s in sorted(crop['State_Name'].unique()):
    print(repr(s))
```

```python
print("=== Season unique values ===")
for s in sorted(crop['Season'].unique()):
    print(repr(s))
```

Checking if city_day can merge with crop_production on State. city_day has no State column, so we need to map cities to states.

```python
city_to_state = {
    'Ahmedabad': 'Gujarat', 'Aizawl': 'Mizoram', 'Amaravati': 'Andhra Pradesh',
    'Amritsar': 'Punjab', 'Bengaluru': 'Karnataka', 'Bhopal': 'Madhya Pradesh',
    'Brajrajnagar': 'Odisha', 'Chandigarh': 'Chandigarh', 'Chennai': 'Tamil Nadu',
    'Coimbatore': 'Tamil Nadu', 'Delhi': 'Delhi', 'Ernakulam': 'Kerala',
    'Gurugram': 'Haryana', 'Guwahati': 'Assam', 'Hyderabad': 'Telangana',
    'Jaipur': 'Rajasthan', 'Jorapokhar': 'Jharkhand', 'Kochi': 'Kerala',
    'Kolkata': 'West Bengal', 'Lucknow': 'Uttar Pradesh', 'Mumbai': 'Maharashtra',
    'Patna': 'Bihar', 'Shillong': 'Meghalaya', 'Talcher': 'Odisha',
    'Thiruvananthapuram': 'Kerala', 'Visakhapatnam': 'Andhra Pradesh'
}

city['State'] = city['City'].map(city_to_state)
print("Added State column to city_day.")
print(city[['City', 'State']].drop_duplicates().sort_values('State').to_string(index=False))
```

```python
city_states = set(city['State'].unique())
crop_states_raw = set(crop['State_Name'].unique())
crop_states_clean = set(s.strip() for s in crop['State_Name'].unique())

print("States in city_day but NOT in crop_production:")
print(city_states - crop_states_clean)
print()
print("States in crop_production but NOT in city_day:")
print(crop_states_clean - city_states)
```

### Inconsistencies Found

| # | File | Column | Issue | Affected Rows | Fix |
|---|---|---|---|---|---|
| 1 | crop_production | State_Name | Trailing space on "Jammu and Kashmir " | 3,478 | Strip whitespace |
| 2 | crop_production | State_Name | Trailing space on "Telangana " | 3,805 | Strip whitespace |
| 3 | crop_production | Season | ALL values have trailing spaces ("Kharif     ", etc.) | 246,091 | Strip whitespace |
| 4 | crop_production | Crop | Trailing space on "Coconut " | 1,985 | Strip whitespace |
| 5 | city_day | (none) | No State column exists for merging | 29,531 | Add State column via city-to-state mapping |
| 6 | city_day | State | "Delhi" has no match in crop_production | 2,009 | Noted - these rows will not join |

**Why this matters:** A merge on State would silently drop all "Jammu and Kashmir " and "Telangana " rows (7,283 records) because the trailing space makes them different strings from their clean counterparts.





Applying fixes: stripping whitespace from all string columns in both datasets.

```python
# Strip whitespace from all object columns in crop
for col in crop.select_dtypes(include='object').columns:
    crop[col] = crop[col].str.strip()

# Strip whitespace from all object columns in city
for col in city.select_dtypes(include='object').columns:
    city[col] = city[col].str.strip()

print("Stripped whitespace from all string columns.")
```

Checking for duplicate rows after cleaning (spaces could have hidden duplicates).

```python
print(f"city exact duplicates: {city.duplicated().sum()}")
print(f"crop exact duplicates: {crop.duplicated().sum()}")

# Also check logical duplicates on key columns
city_key_dups = city.duplicated(subset=['City', 'Date']).sum()
crop_key_dups = city_key_dups = crop.duplicated(subset=['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop']).sum()
print(f"city key duplicates (City+Date): {city.duplicated(subset=['City', 'Date']).sum()}")
print(f"crop key duplicates (State+District+Year+Season+Crop): {crop_key_dups}")
```

Verifying the fixes worked.

```python
print("=== Remaining whitespace issues ===")
issues = 0
for col in crop.select_dtypes(include='object').columns:
    n = (crop[col] != crop[col].str.strip()).sum()
    if n > 0:
        print(f"crop {col}: {n}")
        issues += 1
for col in city.select_dtypes(include='object').columns:
    n = (city[col].dropna() != city[col].dropna().str.strip()).sum()
    if n > 0:
        print(f"city {col}: {n}")
        issues += 1
if issues == 0:
    print("None - all string columns are clean.")
```

```python
print("=== State_Name after cleaning ===")
for s in sorted(crop['State_Name'].unique()):
    print(repr(s))
print(f"\nTotal unique states: {crop['State_Name'].nunique()}")
```

```python
# Confirm merge readiness
city_states = set(city['State'].unique())
crop_states = set(crop['State_Name'].unique())
matching = city_states & crop_states
print(f"Matching states: {len(matching)} / {len(city_states)} city states")
print(f"Non-matching: {city_states - crop_states}")
```

Record counts before and after.

```python
print("=== Record counts ===")
print(f"city_day: {len(raw_city)} -> {len(city)} rows, {raw_city.shape[1]} -> {city.shape[1]} cols (State column added)")
print(f"crop_production: {len(raw_crop)} -> {len(crop)} rows (no rows dropped, whitespace fixed)")
```

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Show Season values before and after
raw_seasons = raw_crop['Season'].value_counts().sort_index()
clean_seasons = crop['Season'].value_counts().sort_index()

raw_seasons.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Season Values - Before (with spaces)')
axes[0].tick_params(axis='x', rotation=45)

clean_seasons.plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
axes[1].set_title('Season Values - After (stripped)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## Task 4


### Plot Choice Justification

- **Histogram with KDE** - shows where AQI values cluster and reveals the distribution shape (skewness). If the distribution is right-skewed, extreme values inflate the mean, making it an unfair summary. Directly answers whether the average is fair to report.
- **Boxplot by city** - shows each city's AQI spread, median, and outliers side by side. Directly answers whether pollution is concentrated in a few cities or widespread.

Two plots are needed: the histogram answers "what is the overall distribution?" while the boxplot answers "which cities drive it?" 


Plotting AQI distribution with mean and median lines. The gap between mean and median indicates skewness.

```python
fig, ax = plt.subplots(figsize=(12, 5))

sns.histplot(city['AQI'], kde=True, ax=ax, color='steelblue', edgecolor='black', bins=50)

mean_aqi = city['AQI'].mean()
median_aqi = city['AQI'].median()
ax.axvline(mean_aqi, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_aqi:.0f}')
ax.axvline(median_aqi, color='darkgreen', linestyle='-', linewidth=2, label=f'Median: {median_aqi:.0f}')

ax.set_xlabel('Air Quality Index (AQI)')
ax.set_ylabel('Number of Daily Readings')
ax.set_title('Distribution of Daily AQI Readings Across All Indian Cities')
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

print(f"Mean: {mean_aqi:.1f}, Median: {median_aqi:.1f}, Skew: {city['AQI'].skew():.2f}")
```

Boxplot per city sorted by median AQI. Cities at the bottom have the worst air quality.

```python
city_medians = city.groupby('City')['AQI'].median().sort_values(ascending=False)
city_order = city_medians.index.tolist()

fig, ax = plt.subplots(figsize=(12, 10))
sns.boxplot(data=city, x='AQI', y='City', order=city_order, ax=ax, color='steelblue')
ax.axvline(mean_aqi, color='red', linestyle='--', label=f'Overall Mean: {mean_aqi:.0f}')
ax.axvline(median_aqi, color='darkgreen', linestyle='-', label=f'Overall Median: {median_aqi:.0f}')
ax.set_xlabel('Air Quality Index (AQI)')
ax.set_ylabel('City')
ax.set_title('AQI Distribution by City (sorted by median, highest at top)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

### Observations

1. **Most readings cluster in the Moderate range, but pollution is concentrated in a few cities.** The histogram peaks between AQI 50-200 with a median of ~118 (Moderate). However, the boxplot shows this is not uniform - cities like Ahmedabad, Delhi, and Patna have median AQI above 150, while Aizawl, Coimbatore, and Shillong stay below 80. The problem is driven by a handful of heavily polluted cities, not a nationwide uniform trend.

2. **The mean AQI (~166) is ~40% higher than the median (~118) - the average is not a fair number to report publicly.** The right-skewed distribution (positive skew, long tail toward high AQI) means extreme readings from a few cities inflate the mean. The board should report the median AQI instead, as it better represents what most cities experience on a typical day.


# Task 5


### Detection Method: End of Distribution (99th Percentile)

**Why End of Distribution:**
- Data-driven threshold — comes from the distribution itself, not an arbitrary cutoff
- Targets only the most extreme 1% of readings
- Robust for skewed data like AQI, unlike IQR which flags too many values on right-skewed distributions

**Treatment: Capping (winsorization), not deletion.**
- Deletion removes entire rows, losing data from all other columns (pollutant readings, date, city)
- Capping retains the row but limits the extreme value to the threshold
- Preserves data volume while removing distortion on mean and standard deviation


Computing the 99th percentile threshold and counting extreme values.

```python
upper_limit = city['AQI'].quantile(0.99)
before_aqi = city['AQI'].copy()
before_stats = city['AQI'].describe()

n_extreme = (city['AQI'] > upper_limit).sum()
print(f"99th percentile (upper limit): {upper_limit:.0f}")
print(f"Current AQI range: {city['AQI'].min():.0f} to {city['AQI'].max():.0f}")
print(f"Values above threshold: {n_extreme} ({n_extreme/len(city)*100:.2f}% of data)")
print()
print("Sample of extreme values:")
print(city[city['AQI'] > upper_limit][['City', 'Date', 'AQI']].head(10))
```

Applying cap at 99th percentile. Values above the threshold are set to the threshold value.

```python
city['AQI'] = city['AQI'].clip(upper=upper_limit)
print(f"AQI range after capping: {city['AQI'].min():.0f} to {city['AQI'].max():.0f}")
print(f"Rows retained: {len(city)} (no rows deleted)")
```

Visual comparison before and after capping.

```python
before_mean = before_aqi.mean()
after_mean = city['AQI'].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(before_aqi, kde=True, ax=axes[0], color='steelblue', bins=50)
axes[0].axvline(upper_limit, color='red', linestyle='--', linewidth=2, label=f'99th pctl: {upper_limit:.0f}')
axes[0].axvline(before_mean, color='orange', linestyle='--', label=f'Mean: {before_mean:.0f}')
axes[0].set_xlabel('AQI')
axes[0].set_ylabel('Frequency')
axes[0].set_title('AQI Distribution - Before Capping')
axes[0].legend()

sns.histplot(city['AQI'], kde=True, ax=axes[1], color='coral', bins=50)
axes[1].axvline(after_mean, color='orange', linestyle='--', label=f'Mean: {after_mean:.0f}')
axes[1].set_xlabel('AQI')
axes[1].set_ylabel('Frequency')
axes[1].set_title('AQI Distribution - After Capping')
axes[1].legend()

plt.tight_layout()
plt.show()
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.boxplot(x=before_aqi, ax=axes[0], color='steelblue')
axes[0].set_xlabel('AQI')
axes[0].set_title('AQI Boxplot - Before Capping')

sns.boxplot(x=city['AQI'], ax=axes[1], color='coral')
axes[1].set_xlabel('AQI')
axes[1].set_title('AQI Boxplot - After Capping')

plt.tight_layout()
plt.show()
```

Comparing summary statistics before and after to quantify the effect.

```python
after_stats = city['AQI'].describe()
comparison = pd.DataFrame({
    'Before': before_stats,
    'After': after_stats,
    'Change': after_stats - before_stats
})
print(comparison.round(2))
```

### Verification

- The maximum AQI is now capped at the 99th percentile - implausibly high readings (some exceeding 2000) are removed
- The mean AQI decreased and moved closer to the median, confirming that extreme values were inflating the average
- Standard deviation decreased - the distribution is tighter and more representative
- No rows were deleted - all records are retained, only the extreme AQI values were clipped
- The boxplot shows outlier dots reduced significantly after capping


# Lab 2


## Task 6


**Approach:**
- `Date` is stored as object - convert to datetime, then extract Year for grouping
- **Line plot** chosen because it communicates a trend over continuous time most intuitively to a non-data-scientist (bar charts show magnitude, line charts show direction)
- **Median** AQI per year used instead of mean - robust to remaining skew after capping
- **Spearman rank correlation** tests whether AQI has a monotonic trend over years (non-parametric, no normality assumption needed for skewed AQI)
- **Mann-Whitney U test** compares pre-2018 vs post-2018 AQI distributions directly - answers whether the policy shift made a measurable difference


Converting Date to datetime and extracting Year.

```python
city['Date'] = pd.to_datetime(city['Date'])
city['Year'] = city['Date'].dt.year
print(f"Date range: {city['Date'].min().date()} to {city['Date'].max().date()}")
print(f"Years: {sorted(city['Year'].unique())}")
```

Computing yearly median AQI. Also showing mean and count per year for context.

```python
yearly_aqi = city.groupby('Year')['AQI'].agg(['median', 'mean', 'count'])
yearly_aqi.columns = ['Median AQI', 'Mean AQI', 'Count']
print(yearly_aqi)
```

Line plot with trend line, highlighted best/worst years, and 2018 policy marker.

```python
yearly_median = city.groupby('Year')['AQI'].median()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(yearly_median.index, yearly_median.values, marker='o', linewidth=2.5,
        color='steelblue', markersize=8, label='Yearly Median AQI')

# Trend line
z = np.polyfit(yearly_median.index, yearly_median.values, 1)
trend_fn = np.poly1d(z)
ax.plot(yearly_median.index, trend_fn(yearly_median.index),
        linestyle='--', color='gray', alpha=0.7, label=f'Trend ({z[0]:+.1f} AQI/yr)')

# Highlight best and worst years
worst_yr = yearly_median.idxmax()
best_yr = yearly_median.idxmin()
worst_val = yearly_median[worst_yr]
best_val = yearly_median[best_yr]

ax.scatter(worst_yr, worst_val, color='red', s=150, zorder=5,
           label=f'Most polluted: {worst_yr} ({worst_val:.0f})')
ax.scatter(best_yr, best_val, color='green', s=150, zorder=5,
           label=f'Least polluted: {best_yr} ({best_val:.0f})')

# Policy shift marker
ax.axvline(x=2017.5, color='orange', linestyle=':', alpha=0.7, label='Policy shift (2018)')

ax.set_xlabel('Year')
ax.set_ylabel('Median AQI')
ax.set_title('Yearly Median AQI Trend Across Indian Cities (2015-2020)')
ax.legend(loc='best', fontsize=9)
ax.set_xticks(yearly_median.index)
plt.tight_layout()
plt.show()
```

Statistical tests: Spearman for overall trend, Mann-Whitney U for pre/post 2018 comparison.

```python
!pip3 install scipy -q
from scipy import stats

# Spearman rank correlation: Year vs AQI (monotonic trend?)
rho, p_spearman = stats.spearmanr(city['Year'], city['AQI'])
print("Spearman Rank Correlation (Year vs AQI):")
print(f"  rho = {rho:.4f}, p-value = {p_spearman:.2e}")
sig_sp = "Significant" if p_spearman < 0.05 else "Not significant"
direction = "negative (AQI declining = improving)" if rho < 0 else "positive (AQI rising = worsening)"
print(f"  {sig_sp} {direction} trend (alpha=0.05)")
print()

# Mann-Whitney U: pre-2018 vs 2018+
pre = city[city['Year'] < 2018]['AQI']
post = city[city['Year'] >= 2018]['AQI']

u_stat, p_mw = stats.mannwhitneyu(pre, post, alternative='two-sided')
print("Mann-Whitney U Test (pre-2018 vs 2018+):")
print(f"  U = {u_stat:.0f}, p-value = {p_mw:.2e}")
print(f"  Pre-2018 median: {pre.median():.1f}, Post-2018 median: {post.median():.1f}")
sig_mw = "Significant" if p_mw < 0.05 else "Not significant"
print(f"  {sig_mw} difference (alpha=0.05)")
```

### Response to the Journalist

The yearly median AQI shows a declining trend from 2015 to 2020 - air quality has measurably improved. The Spearman correlation confirms this is a statistically significant trend (not due to random variation). The Mann-Whitney U test confirms that post-2018 AQI levels are significantly lower than pre-2018.

The most and least polluted years are highlighted on the chart above.

**Important caveat:** 2020 data covers only January-July and is heavily influenced by the nationwide COVID-19 lockdown (March-May 2020), which drastically reduced traffic and industrial emissions. The apparent improvement in 2020 reflects reduced activity, not necessarily the success of pollution policies. A fairer test of policy effectiveness would compare 2018-2019 (post-policy, pre-COVID) against 2015-2017.


## Task 7


**Approach:**
- **Month-level aggregation** chosen over season or quarter - months give finer resolution to pinpoint exactly when AQI peaks, while quarters are too coarse and would blur the Oct-Dec window
- `Date` is already datetime (Task 6) - extract month with `.dt.month`
- **Bar chart** for monthly median AQI - shows the seasonal pattern at a glance with harvest months highlighted
- **Boxplot** per month - shows spread and confirms the pattern is consistent, not driven by a few outlier days
- **Kruskal-Wallis test** - non-parametric test for whether AQI differs significantly across months (equivalent of one-way ANOVA for non-normal data)
- **Mann-Whitney U test (one-sided)** - directly tests the NGO's claim that harvest months (Oct-Dec) have higher AQI than the rest


Extracting month and computing monthly median AQI.

```python
city['Month'] = city['Date'].dt.month
monthly_median = city.groupby('Month')['AQI'].median()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Monthly Median AQI:")
for m, name in enumerate(month_names, 1):
    marker = " <-- harvest" if m in [10, 11, 12] else ""
    print(f"  {name}: {monthly_median[m]:.0f}{marker}")
```

Bar chart with harvest season (Oct-Dec) highlighted in red. Boxplot shows per-month spread.

```python
from matplotlib.patches import Patch

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart of monthly median AQI
colors = ['#d9534f' if m in [10, 11, 12] else 'steelblue' for m in range(1, 13)]
monthly_median.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_xticklabels(month_names, rotation=0)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Median AQI')
axes[0].set_title('Monthly Median AQI')
legend_elems = [Patch(facecolor='#d9534f', edgecolor='black', label='Harvest (Oct-Dec)'),
                Patch(facecolor='steelblue', edgecolor='black', label='Other months')]
axes[0].legend(handles=legend_elems)

# Boxplot per month
palette = {str(m): '#d9534f' if m in [10, 11, 12] else 'steelblue' for m in range(1, 13)}
sns.boxplot(x='Month', y='AQI', data=city.assign(Month=city['Month'].astype(str)), hue='Month', ax=axes[1], palette=palette, legend=False)
axes[1].set_xticklabels(month_names, rotation=0)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('AQI')
axes[1].set_title('AQI Distribution by Month')

plt.tight_layout()
plt.show()
```

Testing the claim statistically. Kruskal-Wallis for overall monthly differences, Mann-Whitney U (one-sided) for harvest vs non-harvest.

```python
# Kruskal-Wallis: does AQI differ across months?
month_groups = [city[city['Month'] == m]['AQI'].values for m in range(1, 13)]
h_stat, p_kw = stats.kruskal(*month_groups)
print("Kruskal-Wallis H-test (AQI across 12 months):")
print(f"  H = {h_stat:.1f}, p-value = {p_kw:.2e}")
print(f"  {'Significant' if p_kw < 0.05 else 'Not significant'} - AQI varies across months (alpha=0.05)")
print()

# Mann-Whitney U: harvest (Oct-Dec) vs rest - one-sided (claim: harvest is worse)
harvest = city[city['Month'].isin([10, 11, 12])]['AQI']
non_harvest = city[~city['Month'].isin([10, 11, 12])]['AQI']
u_stat, p_mw = stats.mannwhitneyu(harvest, non_harvest, alternative='greater')
print("Mann-Whitney U Test (harvest Oct-Dec vs rest, one-sided):")
print(f"  U = {u_stat:.0f}, p-value = {p_mw:.2e}")
print(f"  Harvest median: {harvest.median():.1f}, Non-harvest median: {non_harvest.median():.1f}")
result = "Confirmed" if p_mw < 0.05 else "Not confirmed"
print(f"  {result}: harvest season AQI is significantly higher (alpha=0.05)")
```

### Response to the NGO

The bar chart and boxplot both show a clear seasonal pattern. AQI peaks during October-December (the harvest season) and is lowest during June-September (monsoon, when rain washes pollutants from the air).

The Kruskal-Wallis test confirms that AQI differs significantly across months (p < 0.05). The one-sided Mann-Whitney U test directly confirms the NGO's claim: October-December AQI is statistically significantly higher than the rest of the year.

**The claim is supported by the data.** Air quality is demonstrably worst during the harvest season. This is consistent with crop residue burning (primarily in Punjab and Haryana), compounded by post-monsoon atmospheric inversion that traps pollutants near the surface. The monsoon months (Jun-Sep) have the best air quality due to rain washout.


## Task 8


### Data Transformation & Compatibility

**The Problem:** The datasets are at different scales. `city_day` is daily and city-level (2015-2020), while `crop_production` is seasonal and district-level (1997-2015). They only intersect for the year 2015, and even then, the geography does not overlap (the 2015 crop data only covers Odisha and Sikkim, while the 2015 city data covers entirely different states).

**The Solution:** We must drop the time dimension entirely and aggregate both datasets up to the **State** level to create a cross-sectional "macro profile" for each state.
1. `city_day`: Group by State and take the median of all pollutant/AQI values (represents the state's typical historical air quality).
2. `crop_production`: Group by State_Name and Crop_Year to get total annual Production and Area. Then group by State_Name and take the median of those annual totals (represents the state's typical agricultural output). Add a `Yield` feature (Production / Area).
3. Merge on State.


Applying the aggregation and merging the datasets.

```python
import numpy as np

# 1. Aggregate City data to State level
city_num_cols = city.select_dtypes(include=np.number).columns.tolist()
if 'Year' in city_num_cols: city_num_cols.remove('Year')
if 'Month' in city_num_cols: city_num_cols.remove('Month')

state_aqi = city.groupby('State')[city_num_cols].median().reset_index()

# 2. Aggregate Crop data to State level
# First get annual totals per state
state_annual_crop = crop.groupby(['State_Name', 'Crop_Year'])[['Production', 'Area']].sum().reset_index()
# Then take median across years to get typical output
state_crop = state_annual_crop.groupby('State_Name')[['Production', 'Area']].median().reset_index()
state_crop['Yield'] = state_crop['Production'] / state_crop['Area']

# 3. Merge
combined = pd.merge(state_aqi, state_crop, left_on='State', right_on='State_Name', how='inner')
combined = combined.drop(columns=['State_Name'])

print(f"Successfully merged data for {len(combined)} states.")
print(combined.head())
```

Visualising relationships across all numerical features using a correlation heatmap.

```python
fig, ax = plt.subplots(figsize=(14, 10))

corr = combined.drop(columns=['State']).corr()

# Mask the upper triangle for cleaner viewing
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
            cbar_kws={"shrink": .8})

plt.title('Correlation Matrix: Air Quality vs Agricultural Output (State Level)', fontsize=14)
plt.tight_layout()
plt.show()
```

### Interesting Relationships Identified

1. **Agricultural Area & Ozone/NO2 (Positive Correlation: ~0.63)**
   - *Observation:* States with larger agricultural land areas tend to have significantly higher levels of Nitrogen Dioxide (NO2) and ground-level Ozone (O3).
   - *Why it exists:* Massive farming operations rely heavily on diesel-powered tractors, irrigation pumps, and the transport of goods, all of which are major emitters of NO2. Furthermore, heavy fertilizer application releases nitrogen compounds into the air. NO2 acts as a chemical precursor; under sunlight, it reacts with VOCs to form ground-level Ozone.

2. **Crop Yield & NOx/PM10 (Negative Correlation: ~-0.34 to -0.27)**
   - *Observation:* States with higher concentrations of particulate matter (PM10) and Nitrogen Oxides (NOx) tend to have lower crop yields (Production per Area).
   - *Why it exists:* This addresses the driving question: worse air quality *is* associated with less efficient crop production. Particulate matter settles on leaves, blocking sunlight and severely reducing the plant's ability to photosynthesize. Additionally, associated ground-level ozone enters the plant through stomata and damages cellular tissue, stunting crop growth and reducing overall yield.


# Task 9


**Dear Minister,**

Our analysis of national air quality and agricultural data over the past eight years reveals three critical realities:

First, while overall air quality is slowly improving nationwide, severe pollution is concentrated in just a few key cities. A national average makes things look better than they are for the people actually living in those hotspots.

Second, the data confirms the air is reliably at its worst between October and December. This directly aligns with the harvest season, meaning agricultural fires combined with winter weather are driving our most dangerous air days.

Third, bad air hurts farmers. States with higher levels of smog and soot consistently produce significantly less crop per acre. Thick pollution blocks sunlight and damages crops, meaning farmers are unknowingly hurting their own harvests.

**Recommendation:** 
Focus immediate government interventions on those few heavily polluted cities, specifically by offering local farmers financial incentives or equipment to clear their fields without burning before October arrives.

**Limitation of the Data:** 
We must be transparent: while bad air and low crop yields happen together, this data cannot definitively prove pollution caused the crop loss. Unmeasured factors like drought, rainfall shortages, or poor soil quality could also be responsible.

Respectfully,
Your Data Team


# Optional task - advanced


## Task A: The two extremes — do they tell the same story?

```python
# Sort combined data to find the 3 most and 3 least polluted states
sorted_by_aqi = combined.sort_values('AQI')
least_polluted = sorted_by_aqi.head(3)
most_polluted = sorted_by_aqi.tail(3)

extremes = pd.concat([least_polluted, most_polluted])
extremes['Group'] = ['Least Polluted']*3 + ['Most Polluted']*3

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=extremes, x='State', y='AQI', hue='Group', ax=axes[0], dodge=False)
axes[0].set_title('AQI Comparison')
axes[0].tick_params(axis='x', rotation=45)

# Use log scale for Production because the differences are massive
sns.barplot(data=extremes, x='State', y='Production', hue='Group', ax=axes[1], dodge=False)
axes[1].set_yscale('log')
axes[1].set_title('Crop Production (Log Scale)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Analysis:** The pattern completely **contradicts** the hypothesis. The least polluted states (Mizoram, Meghalaya, Chandigarh) actually produce exponentially *less* crop than the most polluted states (Gujarat, Bihar, Haryana). 

This happens because of confounding geography: Mizoram and Meghalaya are mountainous regions with clean air but very little arable land. Conversely, Haryana and Bihar are flat, agricultural powerhouses. Their massive agricultural output actually *drives* their pollution (via machinery, transport, and crop burning). Comparing these extremes shows that raw agricultural output is dictated by geography, not just air quality.


## Task B: Put a number on the relationship

```python
from scipy.stats import pearsonr

fig, ax = plt.subplots(figsize=(8, 5))

# Filter out extreme outliers in Yield (like Kerala coconuts) for a fairer correlation
clean_combined = combined[combined['Yield'] < 100]

sns.regplot(data=clean_combined, x='AQI', y='Yield', ax=ax, 
            scatter_kws={'alpha':0.6}, line_kws={'color':'red', 'linewidth':2})

ax.set_title('AQI vs Crop Yield Across Indian States')
plt.show()

corr, p_val = pearsonr(clean_combined['AQI'], clean_combined['Yield'])
print(f"Pearson Correlation (AQI vs Yield): {corr:.3f}")
```

**To the Research Team:**

After removing extreme biological outliers (e.g., coconut yields measured in nuts rather than tonnes), the correlation between average state AQI and average crop yield is roughly **-0.21**. 

This is a **weak negative correlation**. It means that while there is a slight tendency for more polluted states to have lower yields per hectare, the relationship is very loose and noisy. 

**What this proves:** It proves that bad air and low yields occasionally coexist. It warrants further investigation.
**What this does NOT prove:** It does not prove that pollution *causes* the yield loss. Correlation is not causation. We have not controlled for critical missing variables such as **rainfall, access to irrigation, soil type, and economic investment**, all of which have a much stronger direct impact on crop yield than air quality alone.


## Task C: One plot to rule them all

```python
# Re-displaying the monthly AQI bar chart from Task 7
fig, ax = plt.subplots(figsize=(10, 5))

colors = ['#d9534f' if m in [10, 11, 12] else 'steelblue' for m in range(1, 13)]
monthly_median.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_xticklabels(month_names, rotation=0)
ax.set_xlabel('Month')
ax.set_ylabel('Median AQI')
ax.set_title('The Air-Agriculture Crisis: India\'s Harvest Smog')

legend_elems = [Patch(facecolor='#d9534f', edgecolor='black', label='Harvest Season (Oct-Dec)')]
ax.legend(handles=legend_elems)

plt.tight_layout()
plt.show()
```

**Caption:** This chart reveals the undeniable seasonal heartbeat of India's pollution crisis, peaking exactly when the agricultural harvest season concludes (October-December). While scatter plots struggle to link annual state averages, this temporal view directly implicates post-harvest crop residue burning as a primary driver of the nation's most hazardous air days.

```python

```
