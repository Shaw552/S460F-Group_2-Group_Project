"""
香港房价数据挖掘分析项目
包含以下数据挖掘技术：
1. 多元线性回归（逐步回归、Lasso/Ridge正则化）
2. 曲线拟合和样条回归
3. PCA降维
4. 支持向量机（SVM）分类
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.interpolate import CubicSpline, interp1d
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')

print("=" * 80)
print("香港房价数据挖掘分析项目")
print("=" * 80)

# =============================================================================
# 1. 数据加载和预处理
# =============================================================================
print("\n[1] 数据加载和预处理")
print("-" * 80)

# 加载数据
df = pd.read_csv('Datasetv2.csv')

print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 清理saleable_area列（移除逗号并转换为数值）
def clean_area(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return float(value)

df['saleable_area(ft^2)'] = df['saleable_area(ft^2)'].apply(clean_area)
df.rename(columns={'saleable_area(ft^2)': 'saleable_area'}, inplace=True)

# 处理日期
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')

# 创建月份特征
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# 处理分类变量编码
le_district = LabelEncoder()
df['district_encoded'] = le_district.fit_transform(df['district'].astype(str))

# 处理布尔变量
df['Rental'] = df['Rental'].astype(int)
df['Public_Housing'] = df['Public Housing'].astype(int)

# 选择数值特征
numeric_features = ['saleable_area', 'unit_rate', 'floor', 'district_encoded', 
                    'Rental', 'Public_Housing', 'month', 'year']

# 移除缺失值
df_clean = df[numeric_features + ['price']].copy()
df_clean = df_clean.dropna()

# 异常值处理（使用IQR方法）
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

print(f"清理后数据形状: {df_clean.shape}")
print(f"缺失值统计:\n{df_clean.isnull().sum()}")
print(f"价格统计:\n{df_clean['price'].describe()}")

# =============================================================================
# 2. 探索性数据分析（EDA）
# =============================================================================
print("\n[2] 探索性数据分析（EDA）")
print("-" * 80)

# 创建输出目录
import os
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# 2.1 房价分布直方图 + 核密度估计
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(df_clean['price'], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_title('房价分布直方图', fontsize=14, fontweight='bold')
axes[0].set_xlabel('价格 (HKD)', fontsize=12)
axes[0].set_ylabel('密度', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 核密度估计
from scipy.stats import gaussian_kde
kde = gaussian_kde(df_clean['price'])
x_range = np.linspace(df_clean['price'].min(), df_clean['price'].max(), 200)
axes[1].plot(x_range, kde(x_range), linewidth=2, color='red')
axes[1].fill_between(x_range, kde(x_range), alpha=0.3, color='red')
axes[1].set_title('房价核密度估计', fontsize=14, fontweight='bold')
axes[1].set_xlabel('价格 (HKD)', fontsize=12)
axes[1].set_ylabel('密度', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/01_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.2 散点图矩阵
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 价格 vs 面积
axes[0, 0].scatter(df_clean['saleable_area'], df_clean['price'], alpha=0.5, s=10)
axes[0, 0].set_xlabel('面积 (ft²)', fontsize=12)
axes[0, 0].set_ylabel('价格 (HKD)', fontsize=12)
axes[0, 0].set_title('价格 vs 面积', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 价格 vs 单价
axes[0, 1].scatter(df_clean['unit_rate'], df_clean['price'], alpha=0.5, s=10, color='orange')
axes[0, 1].set_xlabel('单价 (HKD/ft²)', fontsize=12)
axes[0, 1].set_ylabel('价格 (HKD)', fontsize=12)
axes[0, 1].set_title('价格 vs 单价', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 价格 vs 楼层
axes[1, 0].scatter(df_clean['floor'], df_clean['price'], alpha=0.5, s=10, color='green')
axes[1, 0].set_xlabel('楼层', fontsize=12)
axes[1, 0].set_ylabel('价格 (HKD)', fontsize=12)
axes[1, 0].set_title('价格 vs 楼层', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 面积 vs 单价
axes[1, 1].scatter(df_clean['saleable_area'], df_clean['unit_rate'], alpha=0.5, s=10, color='purple')
axes[1, 1].set_xlabel('面积 (ft²)', fontsize=12)
axes[1, 1].set_ylabel('单价 (HKD/ft²)', fontsize=12)
axes[1, 1].set_title('面积 vs 单价', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/02_scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.3 箱线图：各区域房价分布
# 选择前10个最常见的区域
top_districts = df_clean.groupby('district_encoded')['price'].count().nlargest(10).index
df_top_districts = df_clean[df_clean['district_encoded'].isin(top_districts)]

district_names = [le_district.classes_[i] for i in top_districts]

fig, ax = plt.subplots(figsize=(15, 8))
box_data = [df_top_districts[df_top_districts['district_encoded'] == d]['price'].values 
            for d in top_districts]
bp = ax.boxplot(box_data, labels=district_names, patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.set_title('各区域房价分布比较', fontsize=16, fontweight='bold')
ax.set_xlabel('区域', fontsize=12)
ax.set_ylabel('价格 (HKD)', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/03_district_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.4 相关性热力图
correlation_matrix = df_clean[numeric_features + ['price']].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('变量相关性矩阵热力图', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("EDA图表已保存到 results/figures/ 目录")

# =============================================================================
# 3. 多元线性回归
# =============================================================================
print("\n[3] 多元线性回归分析")
print("-" * 80)

# 准备特征和目标变量
X = df_clean[numeric_features].copy()
y = df_clean['price'].copy()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 3.1 标准多元线性回归
print("\n3.1 标准多元线性回归")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"R² Score: {r2_lr:.4f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"MAE: {mae_lr:.2f}")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': lr.coef_,
    'abs_coefficient': np.abs(lr.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\n特征重要性（按系数绝对值排序）:")
print(feature_importance)

# 3.2 Lasso回归（L1正则化）
print("\n3.2 Lasso回归（L1正则化）")
# 使用交叉验证选择最佳alpha
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.alpha_

lasso = Lasso(alpha=best_alpha_lasso, max_iter=5000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"最佳Alpha: {best_alpha_lasso:.4f}")
print(f"R² Score: {r2_lasso:.4f}")
print(f"RMSE: {rmse_lasso:.2f}")
print(f"MAE: {mae_lasso:.2f}")

# Lasso特征选择
lasso_features = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': lasso.coef_,
    'abs_coefficient': np.abs(lasso.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nLasso选择的特征（非零系数）:")
print(lasso_features[lasso_features['coefficient'] != 0])

# 3.3 Ridge回归（L2正则化）
print("\n3.3 Ridge回归（L2正则化）")
# 使用交叉验证选择最佳alpha
ridge_cv = RidgeCV(cv=5, alphas=np.logspace(-3, 3, 100))
ridge_cv.fit(X_train, y_train)
best_alpha_ridge = ridge_cv.alpha_

ridge = Ridge(alpha=best_alpha_ridge)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print(f"最佳Alpha: {best_alpha_ridge:.4f}")
print(f"R² Score: {r2_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"MAE: {mae_ridge:.2f}")

# 3.4 逐步回归（使用Sequential Feature Selection）
print("\n3.4 逐步回归（特征选择）")
# 前向选择 - 选择最重要的5个特征
n_features_to_select = min(5, len(numeric_features))
try:
    sfs_forward = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=n_features_to_select,
        direction='forward',
        scoring='r2',
        cv=3,  # 减少CV折数以提高速度
        n_jobs=-1
    )
    sfs_forward.fit(X_train, y_train)
    selected_features_forward = X_train.columns[sfs_forward.get_support()].tolist()
    
    lr_forward = LinearRegression()
    lr_forward.fit(X_train[selected_features_forward], y_train)
    y_pred_forward = lr_forward.predict(X_test[selected_features_forward])
    
    r2_forward = r2_score(y_test, y_pred_forward)
    rmse_forward = np.sqrt(mean_squared_error(y_test, y_pred_forward))
    
    print(f"选择的特征: {selected_features_forward}")
    print(f"R² Score: {r2_forward:.4f}")
    print(f"RMSE: {rmse_forward:.2f}")
except Exception as e:
    print(f"逐步回归执行失败: {e}")
    # 使用简单的特征选择作为备选方案
    # 基于相关系数选择特征
    correlations = df_clean[numeric_features + ['price']].corr()['price'].abs().sort_values(ascending=False)
    selected_features_forward = correlations.head(n_features_to_select+1).index[1:].tolist()
    
    lr_forward = LinearRegression()
    lr_forward.fit(X_train[selected_features_forward], y_train)
    y_pred_forward = lr_forward.predict(X_test[selected_features_forward])
    
    r2_forward = r2_score(y_test, y_pred_forward)
    rmse_forward = np.sqrt(mean_squared_error(y_test, y_pred_forward))
    
    print(f"使用基于相关性的特征选择")
    print(f"选择的特征: {selected_features_forward}")
    print(f"R² Score: {r2_forward:.4f}")
    print(f"RMSE: {rmse_forward:.2f}")

# 3.5 回归结果可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 实际值 vs 预测值 - 线性回归
axes[0, 0].scatter(y_test, y_pred_lr, alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('实际价格', fontsize=12)
axes[0, 0].set_ylabel('预测价格', fontsize=12)
axes[0, 0].set_title(f'线性回归: R² = {r2_lr:.4f}', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 实际值 vs 预测值 - Lasso
axes[0, 1].scatter(y_test, y_pred_lasso, alpha=0.5, s=10, color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('实际价格', fontsize=12)
axes[0, 1].set_ylabel('预测价格', fontsize=12)
axes[0, 1].set_title(f'Lasso回归: R² = {r2_lasso:.4f}', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 实际值 vs 预测值 - Ridge
axes[1, 0].scatter(y_test, y_pred_ridge, alpha=0.5, s=10, color='green')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('实际价格', fontsize=12)
axes[1, 0].set_ylabel('预测价格', fontsize=12)
axes[1, 0].set_title(f'Ridge回归: R² = {r2_ridge:.4f}', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 残差图
residuals = y_test - y_pred_lr
axes[1, 1].scatter(y_pred_lr, residuals, alpha=0.5, s=10, color='purple')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('预测价格', fontsize=12)
axes[1, 1].set_ylabel('残差', fontsize=12)
axes[1, 1].set_title('残差图', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/05_regression_results.png', dpi=300, bbox_inches='tight')
plt.close()

# 特征重要性条形图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 线性回归特征重要性
axes[0].barh(feature_importance['feature'], feature_importance['coefficient'])
axes[0].set_xlabel('系数值', fontsize=12)
axes[0].set_title('线性回归特征系数', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Lasso特征重要性
axes[1].barh(lasso_features['feature'], lasso_features['coefficient'])
axes[1].set_xlabel('系数值', fontsize=12)
axes[1].set_title('Lasso回归特征系数', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

# Ridge特征重要性
ridge_features = pd.DataFrame({
    'feature': numeric_features,
    'coefficient': ridge.coef_
}).sort_values('coefficient', key=abs, ascending=False)

axes[2].barh(ridge_features['feature'], ridge_features['coefficient'], color='green')
axes[2].set_xlabel('系数值', fontsize=12)
axes[2].set_title('Ridge回归特征系数', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/figures/06_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 4. 曲线拟合和样条回归
# =============================================================================
print("\n[4] 曲线拟合和样条回归")
print("-" * 80)

# 4.1 面积 vs 价格的样条回归
print("\n4.1 面积 vs 价格的样条回归")

# 对数据进行排序并处理重复值
# 对于相同的面积值，计算平均价格
df_area_price = pd.DataFrame({
    'area': df_clean['saleable_area'].values,
    'price': df_clean['price'].values
})
df_area_price_agg = df_area_price.groupby('area')['price'].mean().reset_index()
df_area_price_agg = df_area_price_agg.sort_values('area')

area_sorted = df_area_price_agg['area'].values
price_sorted = df_area_price_agg['price'].values

# 使用样条插值
# 为了计算效率，使用子采样
n_samples = min(5000, len(area_sorted))
if len(area_sorted) > n_samples:
    indices = np.linspace(0, len(area_sorted)-1, n_samples, dtype=int)
    area_subset = area_sorted[indices]
    price_subset = price_sorted[indices]
else:
    area_subset = area_sorted
    price_subset = price_sorted

# 确保 x 严格递增（去除重复值）
unique_mask = np.concatenate(([True], np.diff(area_subset) > 1e-10))
area_subset = area_subset[unique_mask]
price_subset = price_subset[unique_mask]

# 三次样条插值
if len(area_subset) >= 4:  # 三次样条至少需要4个点
    spline = CubicSpline(area_subset, price_subset)
    area_plot = np.linspace(area_subset.min(), area_subset.max(), 200)
    price_spline = spline(area_plot)
else:
    # 如果点数太少，使用线性插值
    print("警告: 数据点太少，使用线性插值代替样条插值")
    interp_func = interp1d(area_subset, price_subset, kind='linear', fill_value='extrapolate')
    area_plot = np.linspace(area_subset.min(), area_subset.max(), 200)
    price_spline = interp_func(area_plot)

# 多项式拟合（对比）
poly_coeffs = np.polyfit(area_subset, price_subset, 3)
poly_func = np.poly1d(poly_coeffs)
price_poly = poly_func(area_plot)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 样条回归
axes[0].scatter(area_subset[::10], price_subset[::10], alpha=0.3, s=5, label='数据点')
axes[0].plot(area_plot, price_spline, 'r-', linewidth=2, label='三次样条')
axes[0].set_xlabel('面积 (ft²)', fontsize=12)
axes[0].set_ylabel('价格 (HKD)', fontsize=12)
axes[0].set_title('面积 vs 价格 - 样条回归', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 多项式拟合
axes[1].scatter(area_subset[::10], price_subset[::10], alpha=0.3, s=5, label='数据点')
axes[1].plot(area_plot, price_poly, 'g-', linewidth=2, label='三次多项式')
axes[1].set_xlabel('面积 (ft²)', fontsize=12)
axes[1].set_ylabel('价格 (HKD)', fontsize=12)
axes[1].set_title('面积 vs 价格 - 多项式拟合', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/07_spline_regression.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.2 楼层 vs 价格的样条回归
print("\n4.2 楼层 vs 价格的样条回归")

# 对数据进行排序并处理重复值
# 对于相同的楼层值，计算平均价格
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

# 确保 x 严格递增（去除重复值）
unique_mask = np.concatenate(([True], np.diff(floor_subset) > 1e-10))
floor_subset = floor_subset[unique_mask]
price_floor_subset = price_floor_subset[unique_mask]

# 三次样条插值
if len(floor_subset) >= 4:  # 三次样条至少需要4个点
    spline_floor = CubicSpline(floor_subset, price_floor_subset)
    floor_plot = np.linspace(floor_subset.min(), floor_subset.max(), 200)
    price_floor_spline = spline_floor(floor_plot)
else:
    # 如果点数太少，使用线性插值
    print("警告: 数据点太少，使用线性插值代替样条插值")
    interp_func = interp1d(floor_subset, price_floor_subset, kind='linear', fill_value='extrapolate')
    floor_plot = np.linspace(floor_subset.min(), floor_subset.max(), 200)
    price_floor_spline = interp_func(floor_plot)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(floor_subset[::10], price_floor_subset[::10], alpha=0.3, s=5, label='数据点')
ax.plot(floor_plot, price_floor_spline, 'r-', linewidth=2, label='三次样条')
ax.set_xlabel('楼层', fontsize=12)
ax.set_ylabel('价格 (HKD)', fontsize=12)
ax.set_title('楼层 vs 价格 - 样条回归', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/08_floor_spline.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 5. PCA降维
# =============================================================================
print("\n[5] PCA降维分析")
print("-" * 80)

# 准备特征
X_pca = df_clean[numeric_features].copy()

# 标准化
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca)

# 执行PCA
pca = PCA()
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# 计算累积解释方差比
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

print("主成分解释方差比:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\n前2个主成分累积解释方差: {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
print(f"前3个主成分累积解释方差: {cumulative_variance[2]:.4f} ({cumulative_variance[2]*100:.2f}%)")

# 选择解释方差>=85%的主成分数量
n_components = np.where(cumulative_variance >= 0.85)[0][0] + 1
print(f"\n解释85%方差需要的主成分数量: {n_components}")

# 使用前2个主成分进行可视化
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_pca_scaled)

# 按价格区间着色
price_quantiles = df_clean['price'].quantile([0.33, 0.67])
df_clean['price_category'] = pd.cut(df_clean['price'], 
                                     bins=[0, price_quantiles[0.33], price_quantiles[0.67], float('inf')],
                                     labels=['Low', 'Medium', 'High'])

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 解释方差比
axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                'ro-', linewidth=2, markersize=6, label='累积解释方差')
axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='85%阈值')
axes[0, 0].set_xlabel('主成分', fontsize=12)
axes[0, 0].set_ylabel('解释方差比', fontsize=12)
axes[0, 0].set_title('PCA解释方差比', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PCA双标图 - 按价格类别（使用子采样以提高性能）
colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
max_points_per_category = 5000
for category in ['Low', 'Medium', 'High']:
    mask = df_clean['price_category'] == category
    category_data = X_pca_2d[mask]
    if len(category_data) > max_points_per_category:
        # 随机采样
        np.random.seed(42)
        indices = np.random.choice(len(category_data), max_points_per_category, replace=False)
        category_data = category_data[indices]
    axes[0, 1].scatter(category_data[:, 0], category_data[:, 1], 
                       alpha=0.5, s=10, label=category, c=colors[category])

axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
axes[0, 1].set_title('PCA降维可视化（按价格类别）', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 特征在主成分上的载荷
loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
for i, feature in enumerate(numeric_features):
    axes[1, 0].arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                     head_width=0.05, head_length=0.05, fc='black', ec='black')
    axes[1, 0].text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, feature, 
                    fontsize=10, ha='center', va='center')

axes[1, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
axes[1, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
axes[1, 0].set_title('PCA双标图 - 特征载荷', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(-1.2, 1.2)
axes[1, 0].set_ylim(-1.2, 1.2)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 主成分载荷热力图
loadings_df = pd.DataFrame(pca_2d.components_.T, 
                           columns=['PC1', 'PC2'], 
                           index=numeric_features)
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1, 1])
axes[1, 1].set_title('PCA主成分载荷热力图', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/09_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. 支持向量机（SVM）分类
# =============================================================================
print("\n[6] 支持向量机（SVM）分类")
print("-" * 80)

# 6.1 租赁/非租赁分类
print("\n6.1 租赁/非租赁分类")

# 准备特征和目标
X_class = df_clean[numeric_features].copy()
y_rental = df_clean['Rental'].copy()

# 标准化
scaler_svm = StandardScaler()
X_class_scaled = scaler_svm.fit_transform(X_class)

# 划分训练集和测试集
X_train_class, X_test_class, y_train_rental, y_test_rental = train_test_split(
    X_class_scaled, y_rental, test_size=0.2, random_state=42, stratify=y_rental
)

# 线性SVM
svm_linear = SVC(kernel='linear', random_state=42, probability=True)
svm_linear.fit(X_train_class, y_train_rental)
y_pred_linear = svm_linear.predict(X_test_class)

print("线性SVM结果:")
print(classification_report(y_test_rental, y_pred_linear, 
                            target_names=['非租赁', '租赁']))

# RBF核SVM
svm_rbf = SVC(kernel='rbf', random_state=42, probability=True)
svm_rbf.fit(X_train_class, y_train_rental)
y_pred_rbf = svm_rbf.predict(X_test_class)

print("\nRBF核SVM结果:")
print(classification_report(y_test_rental, y_pred_rbf, 
                            target_names=['非租赁', '租赁']))

# 6.2 价格区间分类
print("\n6.2 价格区间分类")

# 创建价格类别
y_price_category = df_clean['price_category'].copy()
le_price = LabelEncoder()
y_price_encoded = le_price.fit_transform(y_price_category)

# 划分训练集和测试集
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_class_scaled, y_price_encoded, test_size=0.2, random_state=42, stratify=y_price_encoded
)

# 线性SVM
svm_price_linear = SVC(kernel='linear', random_state=42, probability=True)
svm_price_linear.fit(X_train_price, y_train_price)
y_pred_price_linear = svm_price_linear.predict(X_test_price)

print("线性SVM价格分类结果:")
print(classification_report(y_test_price, y_pred_price_linear, 
                            target_names=le_price.classes_))

# RBF核SVM
svm_price_rbf = SVC(kernel='rbf', random_state=42, probability=True)
svm_price_rbf.fit(X_train_price, y_train_price)
y_pred_price_rbf = svm_price_rbf.predict(X_test_price)

print("\nRBF核SVM价格分类结果:")
print(classification_report(y_test_price, y_pred_price_rbf, 
                            target_names=le_price.classes_))

# 6.3 SVM分类结果可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 租赁分类混淆矩阵 - 线性
cm_rental_linear = confusion_matrix(y_test_rental, y_pred_linear)
sns.heatmap(cm_rental_linear, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['非租赁', '租赁'], yticklabels=['非租赁', '租赁'], ax=axes[0, 0])
axes[0, 0].set_title('租赁分类 - 线性SVM混淆矩阵', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('实际', fontsize=12)
axes[0, 0].set_xlabel('预测', fontsize=12)

# 租赁分类混淆矩阵 - RBF
cm_rental_rbf = confusion_matrix(y_test_rental, y_pred_rbf)
sns.heatmap(cm_rental_rbf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['非租赁', '租赁'], yticklabels=['非租赁', '租赁'], ax=axes[0, 1])
axes[0, 1].set_title('租赁分类 - RBF核SVM混淆矩阵', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('实际', fontsize=12)
axes[0, 1].set_xlabel('预测', fontsize=12)

# 价格分类混淆矩阵 - 线性
cm_price_linear = confusion_matrix(y_test_price, y_pred_price_linear)
sns.heatmap(cm_price_linear, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=le_price.classes_, yticklabels=le_price.classes_, ax=axes[1, 0])
axes[1, 0].set_title('价格分类 - 线性SVM混淆矩阵', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('实际', fontsize=12)
axes[1, 0].set_xlabel('预测', fontsize=12)

# 价格分类混淆矩阵 - RBF
cm_price_rbf = confusion_matrix(y_test_price, y_pred_price_rbf)
sns.heatmap(cm_price_rbf, annot=True, fmt='d', cmap='Purples', 
            xticklabels=le_price.classes_, yticklabels=le_price.classes_, ax=axes[1, 1])
axes[1, 1].set_title('价格分类 - RBF核SVM混淆矩阵', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('实际', fontsize=12)
axes[1, 1].set_xlabel('预测', fontsize=12)

plt.tight_layout()
plt.savefig('results/figures/10_svm_classification.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. 时间序列分析
# =============================================================================
print("\n[7] 时间序列分析")
print("-" * 80)

# 按月统计平均价格
monthly_price = df_clean.groupby(['year', 'month'])['price'].mean().reset_index()
monthly_price['date'] = pd.to_datetime(monthly_price[['year', 'month']].assign(day=1))

# 计算滚动平均
monthly_price = monthly_price.sort_values('date')
monthly_price['rolling_mean_3'] = monthly_price['price'].rolling(window=3, min_periods=1).mean()
monthly_price['rolling_mean_6'] = monthly_price['price'].rolling(window=6, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# 月度平均价格趋势
axes[0].plot(monthly_price['date'], monthly_price['price'], 
             'o-', linewidth=1, markersize=4, label='月度平均价格', alpha=0.7)
axes[0].plot(monthly_price['date'], monthly_price['rolling_mean_3'], 
             '--', linewidth=2, label='3个月滚动平均', color='orange')
axes[0].plot(monthly_price['date'], monthly_price['rolling_mean_6'], 
             '--', linewidth=2, label='6个月滚动平均', color='green')
axes[0].set_xlabel('日期', fontsize=12)
axes[0].set_ylabel('平均价格 (HKD)', fontsize=12)
axes[0].set_title('月度平均价格趋势', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# 按月统计交易量
monthly_volume = df_clean.groupby(['year', 'month'])['price'].count().reset_index()
monthly_volume['date'] = pd.to_datetime(monthly_volume[['year', 'month']].assign(day=1))
monthly_volume = monthly_volume.sort_values('date')

axes[1].bar(monthly_volume['date'], monthly_volume['price'], 
            alpha=0.7, color='skyblue', width=20)
axes[1].set_xlabel('日期', fontsize=12)
axes[1].set_ylabel('交易数量', fontsize=12)
axes[1].set_title('月度交易量', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/figures/11_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# 8. 模型性能总结
# =============================================================================
print("\n[8] 模型性能总结")
print("=" * 80)

results_summary = pd.DataFrame({
    '模型': ['线性回归', 'Lasso回归', 'Ridge回归', '逐步回归'],
    'R² Score': [r2_lr, r2_lasso, r2_ridge, r2_forward],
    'RMSE': [rmse_lr, rmse_lasso, rmse_ridge, rmse_forward],
    'MAE': [mae_lr, mae_lasso, mae_ridge, mean_absolute_error(y_test, y_pred_forward)]
})

print("\n回归模型性能对比:")
print(results_summary.to_string(index=False))

# 保存结果
results_summary.to_csv('results/regression_results.csv', index=False)
feature_importance.to_csv('results/feature_importance.csv', index=False)

print("\n" + "=" * 80)
print("分析完成！所有结果已保存到 results/ 目录")
print("=" * 80)
print("\n生成的文件:")
print("  - results/figures/01_price_distribution.png - 房价分布")
print("  - results/figures/02_scatter_matrix.png - 散点图矩阵")
print("  - results/figures/03_district_boxplot.png - 区域房价箱线图")
print("  - results/figures/04_correlation_heatmap.png - 相关性热力图")
print("  - results/figures/05_regression_results.png - 回归结果")
print("  - results/figures/06_feature_importance.png - 特征重要性")
print("  - results/figures/07_spline_regression.png - 样条回归")
print("  - results/figures/08_floor_spline.png - 楼层样条回归")
print("  - results/figures/09_pca_analysis.png - PCA分析")
print("  - results/figures/10_svm_classification.png - SVM分类")
print("  - results/figures/11_time_series.png - 时间序列分析")
print("  - results/regression_results.csv - 回归结果汇总")
print("  - results/feature_importance.csv - 特征重要性")

