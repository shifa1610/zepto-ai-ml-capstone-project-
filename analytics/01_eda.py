from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the raw dataset once and save an offline copy immediately.
df = sns.load_dataset("titanic")
df.to_csv("analytics/titanic.csv", index=False)

print("Dataset information:")
df.info()

print("\nDataset shape:", df.shape)
print("\nNumeric summary:")
print(df.describe())

# Report the missing-value percentage for every affected column.
missing_percent = df.isna().mean() * 100
missing_percent = missing_percent[missing_percent > 0]

print("\nMissing-value percentages:")
print(missing_percent.round(2).to_string())

# Apply the assignment's missing-value thresholds.
clean_df = df.copy()

for column, percent in missing_percent.items():
    if percent < 5:
        clean_df = clean_df.dropna(subset=[column])
        print(f"{column}: {percent:.2f}% missing — dropped rows with missing values.")
    elif percent <= 30:
        if pd.api.types.is_numeric_dtype(clean_df[column]):
            clean_df[column] = clean_df[column].fillna(clean_df[column].median())
            method = "median imputation"
        else:
            clean_df[column] = clean_df[column].fillna(clean_df[column].mode()[0])
            method = "mode imputation"
        print(f"{column}: {percent:.2f}% missing — used {method}.")
    else:
        clean_df = clean_df.drop(columns=[column])
        print(f"{column}: {percent:.2f}% missing — dropped the column.")

clean_df.to_csv("analytics/titanic_cleaned.csv", index=False)

print("\nCleaned dataset shape:", clean_df.shape)
print("Saved raw fallback: analytics/titanic.csv")
print("Saved cleaned data: analytics/titanic_cleaned.csv")
# Histograms, box plots, and IQR outlier counts for age and fare.
Path("analytics/plots").mkdir(parents=True, exist_ok=True)

for column in ["age", "fare"]:
    q1 = clean_df[column].quantile(0.25)
    q3 = clean_df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = (
        (clean_df[column] < lower_bound) | (clean_df[column] > upper_bound)
    ).sum()

    print(
        f"\n{column}: IQR outlier count = {outlier_count}; "
        f"bounds = [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(clean_df[column], bins=25, edgecolor="black")
    axes[0].set_title(f"{column.title()} distribution")
    axes[0].set_xlabel(column.title())
    axes[0].set_ylabel("Number of passengers")

    axes[1].boxplot(clean_df[column].dropna(), vert=True)
    axes[1].set_title(f"{column.title()} box plot")
    axes[1].set_ylabel(column.title())

    fig.tight_layout()
    fig.savefig(f"analytics/plots/{column}_histogram_boxplot.png", dpi=150)
    plt.close(fig)

fare_mean = clean_df["fare"].mean()
fare_median = clean_df["fare"].median()
fare_mode = clean_df["fare"].mode().iloc[0]

print("\nFare summary:")
print(f"Mean:   {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode:   {fare_mode:.2f}")
Path("analytics/plots").mkdir(parents=True, exist_ok=True)

for column in ["age", "fare"]:
    q1 = clean_df[column].quantile(0.25)
    q3 = clean_df[column].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ((clean_df[column] < low) | (clean_df[column] > high)).sum()

    print(f"\n{column} IQR outlier count: {outliers}")
    print(f"IQR bounds: {low:.2f} to {high:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(clean_df[column], bins=25, edgecolor="black")
    axes[0].set_title(f"{column.title()} histogram")
    axes[1].boxplot(clean_df[column].dropna())
    axes[1].set_title(f"{column.title()} box plot")
    fig.tight_layout()
    fig.savefig(f"analytics/plots/{column}_histogram_boxplot.png", dpi=150)
    plt.close(fig)

print("\nFare mean:", round(clean_df["fare"].mean(), 2))
print("Fare median:", round(clean_df["fare"].median(), 2))
print("Fare mode:", round(clean_df["fare"].mode().iloc[0], 2))
# Survival rates using boolean masks.
print("\nSurvival rate by sex:")
for sex_value in ["female", "male"]:
    mask = clean_df["sex"] == sex_value
    print(f"{sex_value}: {clean_df.loc[mask, 'survived'].mean():.3f}")

print("\nSurvival rate by passenger class:")
for class_value in [1, 2, 3]:
    mask = clean_df["pclass"] == class_value
    print(f"Class {class_value}: {clean_df.loc[mask, 'survived'].mean():.3f}")

print("\nSurvival rate by sex and passenger class:")
for sex_value in ["female", "male"]:
    for class_value in [1, 2, 3]:
        mask = (clean_df["sex"] == sex_value) & (clean_df["pclass"] == class_value)
        print(
            f"{sex_value}, class {class_value}: "
            f"{clean_df.loc[mask, 'survived'].mean():.3f}"
        )

# Correlation matrix on exactly the six required columns.
corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation = clean_df[corr_columns].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
ax.set_title("Titanic feature correlations")
fig.tight_layout()
fig.savefig("analytics/plots/correlation_heatmap.png", dpi=150)
plt.close(fig)

upper_triangle = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
)
strongest_pairs = upper_triangle.stack().abs().sort_values(ascending=False).head(2)

print("\nTwo strongest absolute off-diagonal correlations:")
for (feature_a, feature_b), _ in strongest_pairs.items():
    print(f"{feature_a} and {feature_b}: r = {correlation.loc[feature_a, feature_b]:.3f}")

# Four multivariate charts.
survival_by_sex = clean_df.groupby("sex", observed=False)["survived"].mean()
ax = survival_by_sex.plot(kind="bar", color=["steelblue", "coral"])
ax.set(title="Survival rate by sex", xlabel="Sex", ylabel="Survival rate", ylim=(0, 1))
plt.tight_layout()
plt.savefig("analytics/plots/survival_by_sex.png", dpi=150)
plt.close()

survival_by_class = clean_df.groupby("pclass")["survived"].mean()
ax = survival_by_class.plot(kind="bar", color="slateblue")
ax.set(title="Survival rate by passenger class", xlabel="Passenger class",
       ylabel="Survival rate", ylim=(0, 1))
plt.tight_layout()
plt.savefig("analytics/plots/survival_by_class.png", dpi=150)
plt.close()

sex_class_rates = (
    clean_df.groupby(["pclass", "sex"], observed=False)["survived"].mean().unstack()
)
ax = sex_class_rates.plot(kind="bar", figsize=(7, 4))
ax.set(title="Survival rate by sex and passenger class",
       xlabel="Passenger class", ylabel="Survival rate", ylim=(0, 1))
plt.tight_layout()
plt.savefig("analytics/plots/survival_by_sex_and_class.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=clean_df, x="survived", y="age", ax=axes[0])
axes[0].set(title="Age by survival outcome", xlabel="Survived (0=no, 1=yes)")
sns.boxplot(data=clean_df, x="survived", y="fare", ax=axes[1])
axes[1].set(title="Fare by survival outcome", xlabel="Survived (0=no, 1=yes)")
fig.tight_layout()
fig.savefig("analytics/plots/age_fare_by_survival.png", dpi=150)
plt.close(fig)

# EDA-only z-score check; this does not feed into model training.
standardized = (clean_df[["age", "fare"]] - clean_df[["age", "fare"]].mean()) / clean_df[
    ["age", "fare"]
].std(ddof=0)

print("\nEDA standardization check (means should be near 0; stds near 1):")
print("Means:\n", standardized.mean().round(3))
print("Standard deviations:\n", standardized.std(ddof=0).round(3))