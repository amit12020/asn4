# 📦 Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10,6)

# 1. Load your data
df = pd.read_csv('your_data.csv')  # replace with your file

# 2. Basic inspection
print("Shape:", df.shape)
print(df.info())
print(df.describe(include='all'))
print("Missing values:\n", df.isnull().sum().sort_values(ascending=False))

# 3. Handle missing values
# Drop columns with >30% missing
threshold = 0.3 * len(df)
drop_cols = df.columns[df.isnull().sum() > threshold]
df.drop(columns=drop_cols, inplace=True)

# Impute remaining missing values
for col in df:
    if df[col].isnull().any():
        if df[col].dtype in ['float64','int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 4. Define column types
num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

# 5. Univariate analysis: distributions & outliers
for col in num_cols:
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()

# 6. Outlier detection & capping using IQR
outlier_counts = {}
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25,0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    mask = (df[col] < lower) | (df[col] > upper)
    outlier_counts[col] = mask.sum()
    df[col] = np.clip(df[col], lower, upper)
print("Outliers per col:", outlier_counts)

# 7. Bivariate & multivariate analysis
# Correlation heatmap
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot (sampled)
sns.pairplot(df[num_cols].sample(min(len(df),500)))
plt.show()

# Scatter for first two numeric
if len(num_cols) >= 2:
    sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]], hue=df[num_cols[2]] if len(num_cols)>=3 else None)
    plt.title(f'{num_cols[0]} vs {num_cols[1]}')
    plt.show()

# 8. Categorical vs numerical & categorical counts
for col in cat_cols:
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f'Count of {col}')
    plt.show()

    for num in num_cols:
        sns.boxplot(x=df[col], y=df[num])
        plt.xticks(rotation=45)
        plt.title(f'{num} by {col}')
        plt.show()

# Optional: Feature engineering for date/time if exists
if 'date' in df.columns or 'Date' in df.columns:
    date_col = 'date' if 'date' in df.columns else 'Date'
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month

print("🧠 EDA complete. Data is ready for modeling or deeper analysis.")
