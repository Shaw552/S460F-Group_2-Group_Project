import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.interpolate import CubicSpline, interp1d
import warnings
warnings.filterwarnings('ignore')

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set chart style
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')

print("=" * 80)
print("Hong Kong Housing Price Data Mining and Analysis Project")
print("=" * 80)

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
print("\n[1] Data Loading and Preprocessing")
print("-" * 80)

# Load data
df = pd.read_csv('Datasetv2.csv')

print(f"Original data shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Clean saleable_area column (remove commas and convert to numeric)
def clean_area(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return float(value)

df['saleable_area(ft^2)'] = df['saleable_area(ft^2)'].apply(clean_area)
df.rename(columns={'saleable_area(ft^2)': 'saleable_area'}, inplace=True)

# Process date
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

# Create month and year features
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Handle categorical variable encoding
le_district = LabelEncoder()
df['district_encoded'] = le_district.fit_transform(df['district'].astype(str))

# Handle boolean variables
df['Rental'] = df['Rental'].astype(int)
df['Public_Housing'] = df['Public Housing'].astype(int)

# Select numeric features
numeric_features = ['saleable_area', 'unit_rate', 'floor', 'district_encoded', 
                    'Rental', 'Public_Housing', 'month', 'year']

# Remove missing values
df_clean = df[numeric_features + ['price']].copy()
df_clean = df_clean.dropna()

# Handle outliers (using IQR method)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_clean = remove_outliers_iqr(df_clean, 'price')
df_clean = remove_outliers_iqr(df_clean, 'saleable_area')
df_clean = remove_outliers_iqr(df_clean, 'unit_rate')

print(f"Cleaned data shape: {df_clean.shape}")
print(f"Missing value statistics:\n{df_clean.isnull().sum()}")
print(f"Price statistics:\n{df_clean['price'].describe()}")

# =============================================================================
# 2. Exploratory Data Analysis (EDA)
# =============================================================================
print("\n[2] Exploratory Data Analysis (EDA)")
print("-" * 80)

# Create output directories
import os
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# 2.1 Correlation Heatmap
correlation_matrix = df_clean[numeric_features + ['price']].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Variable Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("EDA charts have been saved to the results/figures/ directory")

# =============================================================================
# 3. Multiple Linear Regression
# =============================================================================
print("\n[3] Multiple Linear Regression Analysis")
print("-" * 80)

# Prepare features and target variable
X = df_clean[numeric_features].copy()
y = df_clean['price'].copy()

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_features)

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 3.1 Standard Multiple Linear Regression
print("\n3.1 Standard Multiple Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"R² Score: {r2_lr:.4f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"MAE: {mae_lr:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': lr.coef_,
    'abs_coefficient': np.abs(lr.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature Importance (sorted by absolute value of coefficients):")
print(feature_importance)

# 3.2 Lasso Regression (L1 Regularization)
print("\n3.2 Lasso Regression (L1 Regularization)")
# Select best alpha using cross-validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.alpha_

lasso = Lasso(alpha=best_alpha_lasso, max_iter=5000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"Best Alpha: {best_alpha_lasso:.4f}")
print(f"R² Score: {r2_lasso:.4f}")
print(f"RMSE: {rmse_lasso:.2f}")
print(f"MAE: {mae_lasso:.2f}")

# Lasso feature selection
lasso_features = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': lasso.coef_,
    'abs_coefficient': np.abs(lasso.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nLasso Selected Features (non-zero coefficients):")
print(lasso_features[lasso_features['coefficient'] != 0])

# 3.3 Ridge Regression (L2 Regularization)
print("\n3.3 Ridge Regression (L2 Regularization)")
# Select best alpha using cross-validation
ridge_cv = RidgeCV(cv=5, alphas=np.logspace(-3, 3, 100))
ridge_cv.fit(X_train, y_train)
best_alpha_ridge = ridge_cv.alpha_

ridge = Ridge(alpha=best_alpha_ridge)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print(f"Best Alpha: {best_alpha_ridge:.4f}")
print(f"R² Score: {r2_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"MAE: {mae_ridge:.2f}")

# 3.4 Stepwise Regression (using Sequential Feature Selection)
print("\n3.4 Stepwise Regression (Feature Selection)")
# Forward selection - select the top 5 most important features
n_features_to_select = min(5, len(numeric_features))
try:
    sfs_forward = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction='forward',
        scoring='r2',
        cv=3,  # Reduce CV folds for speed
        n_jobs=-1
    )
    sfs_forward.fit(X_train, y_train)
    selected_features_forward = X_train.columns[sfs_forward.get_support()].tolist()
    
    lr_forward = LinearRegression()
    lr_forward.fit(X_train[selected_features_forward], y_train)
    y_pred_forward = lr_forward.predict(X_test[selected_features_forward])
    
    r2_forward = r2_score(y_test, y_pred_forward)
    rmse_forward = np.sqrt(mean_squared_error(y_test, y_pred_forward))
    
    print(f"Selected features: {selected_features_forward}")
    print(f"R² Score: {r2_forward:.4f}")
    print(f"RMSE: {rmse_forward:.2f}")
except Exception as e:
    print(f"Stepwise regression failed: {e}")
    # Fallback to simple feature selection
    # Select features based on correlation with target
    correlations = df_clean[numeric_features + ['price']].corr()['price'].abs().sort_values(ascending=False)
    selected_features_forward = correlations.head(n_features_to_select+1).index[1:].tolist()
    
    lr_forward = LinearRegression()
    lr_forward.fit(X_train[selected_features_forward], y_train)
    y_pred_forward = lr_forward.predict(X_test[selected_features_forward])
    
    r2_forward = r2_score(y_test, y_pred_forward)
    rmse_forward = np.sqrt(mean_squared_error(y_test, y_pred_forward))
    
    print(f"Using correlation-based feature selection")
    print(f"Selected features: {selected_features_forward}")
    print(f"R² Score: {r2_forward:.4f}")
    print(f"RMSE: {rmse_forward:.2f}")

# 3.5 Regression Results Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Actual vs Predicted - Linear Regression
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price', fontsize=12)
axes[0, 0].set_ylabel('Predicted Price', fontsize=12)
axes[0, 0].set_title(f'Linear Regression: R² = {r2_lr:.4f}', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Actual vs Predicted - Lasso
axes[0, 1].scatter(y_test, y_pred_lasso, alpha=0.5, s=10, color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price', fontsize=12)
axes[0, 1].set_ylabel('Predicted Price', fontsize=12)
axes[0, 1].set_title(f'Lasso Regression: R² = {r2_lasso:.4f}', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Actual vs Predicted - Ridge
axes[1, 0].scatter(y_test, y_pred_ridge, alpha=0.5, s=10, color='green')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Price', fontsize=12)
axes[1, 0].set_ylabel('Predicted Price', fontsize=12)
axes[1, 0].set_title(f'Ridge Regression: R² = {r2_ridge:.4f}', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Residuals Plot
residuals = y_test - y_pred_lr
axes[1, 1].scatter(y_pred_lr, residuals, alpha=0.5, s=10, color='purple')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Price', fontsize=12)
axes[1, 1].set_ylabel('Residuals', fontsize=12)
axes[1, 1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/05_regression_results.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. Curve Fitting and Spline Regression
# =============================================================================
print("\n[4] Curve Fitting and Spline Regression")
print("-" * 80)

# 4.1 Spline Regression for Area vs Price
print("\n4.1 Spline Regression for Area vs Price")

# Sort data and handle duplicates
# For the same area value, calculate the average price
df_area_price = pd.DataFrame({
    'area': df_clean['saleable_area'].values,
    'price': df_clean['price'].values
})
df_area_price_agg = df_area_price.groupby('area')['price'].mean().reset_index()
df_area_price_agg = df_area_price_agg.sort_values('area')

area_sorted = df_area_price_agg['area'].values
price_sorted = df_area_price_agg['price'].values

# Spline interpolation
# For computational efficiency, use subsampling
n_samples = min(5000, len(area_sorted))
if len(area_sorted) > n_samples:
    indices = np.linspace(0, len(area_sorted)-1, n_samples, dtype=int)
    area_subset = area_sorted[indices]
    price_subset = price_sorted[indices]
else:
    area_subset = area_sorted
    price_subset = price_sorted

# Ensure x is strictly increasing (remove duplicates)
unique_mask = np.concatenate(([True], np.diff(area_subset) > 1e-10))
area_subset = area_subset[unique_mask]
price_subset = price_subset[unique_mask]

# Cubic spline interpolation
if len(area_subset) >= 4:  # Cubic spline requires at least 4 points
    spline = CubicSpline(area_subset, price_subset)
    area_plot = np.linspace(area_subset.min(), area_subset.max(), 200)
    price_spline = spline(area_plot)
else:
    # If too few points, use linear interpolation
    print("Warning: Too few data points, using linear interpolation instead of spline interpolation")
    interp_func = interp1d(area_subset, price_subset, kind='linear', fill_value='extrapolate')
    area_plot = np.linspace(area_subset.min(), area_subset.max(), 200)
    price_spline = interp_func(area_plot)

# Polynomial fitting (for comparison)
poly_coeffs = np.polyfit(area_subset, price_subset, 3)
poly_func = np.poly1d(poly_coeffs)
price_poly = poly_func(area_plot)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Spline regression
axes[0].scatter(area_subset[::10], price_subset[::10], alpha=0.3, s=5, label='Data Points')
axes[0].plot(area_plot, price_spline, 'r-', linewidth=2, label='Cubic Spline')
axes[0].set_xlabel('Area (ft²)', fontsize=12)
axes[0].set_ylabel('Price (HKD)', fontsize=12)
axes[0].set_title('Area vs Price - Spline Regression', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Polynomial fitting
axes[1].scatter(area_subset[::10], price_subset[::10], alpha=0.3, s=5, label='Data Points')
axes[1].plot(area_plot, price_poly, 'g-', linewidth=2, label='Cubic Polynomial')
axes[1].set_xlabel('Area (ft²)', fontsize=12)
axes[1].set_ylabel('Price (HKD)', fontsize=12)
axes[1].set_title('Area vs Price - Polynomial Fitting', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/07_spline_regression.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.2 Spline Regression for Floor vs Price
print("\n4.2 Spline Regression for Floor vs Price")

# Sort data and handle duplicates
# For the same floor value, calculate the average price
df_floor_price = pd.DataFrame({
    'floor': df_clean['floor'].values,
    'price': df_clean['price'].values
})
df_floor_price_agg = df_floor_price.groupby('floor')['price'].mean().reset_index()
df_floor_price_agg = df_floor_price_agg.sort_values('floor')

floor_sorted = df_floor_price_agg['floor'].values
price_floor_sorted = df_floor_price_agg['price'].values

n_samples_floor = min(5000, len(floor_sorted))
if len(floor_sorted) > n_samples_floor:
    indices_floor = np.linspace(0, len(floor_sorted)-1, n_samples_floor, dtype=int)
    floor_subset = floor_sorted[indices_floor]
    price_floor_subset = price_floor_sorted[indices_floor]
else:
    floor_subset = floor_sorted
    price_floor_subset = price_floor_sorted

# Ensure x is strictly increasing (remove duplicates)
unique_mask = np.concatenate(([True], np.diff(floor_subset) > 1e-10))
floor_subset = floor_subset[unique_mask]
price_floor_subset = price_floor_subset[unique_mask]

# Cubic spline interpolation
if len(floor_subset) >= 4:  # Cubic spline requires at least 4 points
    spline_floor = CubicSpline(floor_subset, price_floor_subset)
    floor_plot = np.linspace(floor_subset.min(), floor_subset.max(), 200)
    price_floor_spline = spline_floor(floor_plot)
else:
    # If too few points, use linear interpolation
    print("Warning: Too few data points, using linear interpolation instead of spline interpolation")
    interp_func = interp1d(floor_subset, price_floor_subset, kind='linear', fill_value='extrapolate')
    floor_plot = np.linspace(floor_subset.min(), floor_subset.max(), 200)
    price_floor_spline = interp_func(floor_plot)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(floor_subset[::10], price_floor_subset[::10], alpha=0.3, s=5, label='Data Points')
ax.plot(floor_plot, price_floor_spline, 'r-', linewidth=2, label='Cubic Spline')
ax.set_xlabel('Floor', fontsize=12)
ax.set_ylabel('Price (HKD)', fontsize=12)
ax.set_title('Floor vs Price - Spline Regression', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/08_floor_spline.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. Time Series Analysis
# =============================================================================
print("\n[5] Time Series Analysis")
print("-" * 80)

# Monthly average price statistics
monthly_price = df_clean.groupby(['year', 'month'])['price'].mean().reset_index()
monthly_price['date'] = pd.to_datetime(monthly_price[['year', 'month']].assign(day=1))

# Calculate rolling average
monthly_price = monthly_price.sort_values('date')
monthly_price['rolling_mean_3'] = monthly_price['price'].rolling(window=3, min_periods=1).mean()
monthly_price['rolling_mean_6'] = monthly_price['price'].rolling(window=6, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Monthly average price trend
axes[0].plot(monthly_price['date'], monthly_price['price'], 
             'o-', linewidth=1, markersize=4, label='Monthly Average Price', alpha=0.7)
axes[0].plot(monthly_price['date'], monthly_price['rolling_mean_3'], 
             '--', linewidth=2, label='3-Month Rolling Average', color='orange')
axes[0].plot(monthly_price['date'], monthly_price['rolling_mean_6'], 
             '--', linewidth=2, label='6-Month Rolling Average', color='green')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Average Price (HKD)', fontsize=12)
axes[0].set_title('Monthly Average Price Trend', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Monthly transaction volume statistics
monthly_volume = df_clean.groupby(['year', 'month'])['price'].count().reset_index()
monthly_volume['date'] = pd.to_datetime(monthly_volume[['year', 'month']].assign(day=1))
monthly_volume = monthly_volume.sort_values('date')

axes[1].bar(monthly_volume['date'], monthly_volume['price'], 
            alpha=0.7, color='skyblue', width=20)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Transaction Volume', fontsize=12)
axes[1].set_title('Monthly Transaction Volume', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/11_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. Model Performance Summary
# =============================================================================
print("\n[6] Model Performance Summary")
print("=" * 80)

results_summary = pd.DataFrame({
    'Model': ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'Stepwise Regression'],
    'R² Score': [r2_lr, r2_lasso, r2_ridge, r2_forward],
    'RMSE': [rmse_lr, rmse_lasso, rmse_ridge, rmse_forward],
    'MAE': [mae_lr, mae_lasso, mae_ridge, mean_absolute_error(y_test, y_pred_forward)]
})

print("\nRegression Model Performance Comparison:")
print(results_summary.to_string(index=False))

# Save results
results_summary.to_csv('results/regression_results.csv', index=False)
feature_importance.to_csv('results/feature_importance.csv', index=False)

print("\n" + "=" * 80)
print("Analysis complete! All results have been saved to the results/ directory")
print("=" * 80)
print("\nGenerated files:")
print("  - results/figures/04_correlation_heatmap.png - Correlation Heatmap")
print("  - results/figures/05_regression_results.png - Regression Results")
print("  - results/figures/07_spline_regression.png - Spline Regression")
print("  - results/figures/08_floor_spline.png - Floor Spline Regression")
print("  - results/figures/11_time_series.png - Time Series Analysis")
print("  - results/regression_results.csv - Regression Results Summary")
print("  - results/feature_importance.csv - Feature Importance")