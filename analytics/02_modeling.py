from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Continue from the cleaned data created by 01_eda.py.
df = pd.read_csv("analytics/titanic_cleaned.csv")

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
target = "survived"
X = df[features]
y = df[target]

print("Survival class balance:")
print(y.value_counts(normalize=True).round(3))

# Split before fitting any preprocessing. Stratification preserves class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=1
    ),
}

Path("analytics/plots").mkdir(parents=True, exist_ok=True)
fitted_pipelines = {}
metric_rows = []

# Train and evaluate all three models on the identical split.
plt.figure(figsize=(7, 6))

for model_name, estimator in models.items():
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    full_pipeline.fit(X_train, y_train)
    fitted_pipelines[model_name] = full_pipeline

    predictions = full_pipeline.predict(X_test)
    probabilities = full_pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    model_auc = auc(fpr, tpr)

    metric_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "auc": model_auc,
        }
    )

    print(f"\n{model_name} confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves for Titanic classifiers")
plt.legend()
plt.tight_layout()
plt.savefig("analytics/plots/classifier_roc_curves.png", dpi=150)
plt.close()

metrics_table = pd.DataFrame(metric_rows)
print("\nClassifier comparison:")
print(metrics_table.round(3).to_string(index=False))
metrics_table.to_csv("analytics/classifier_metrics.csv", index=False)

# Draw the fitted decision tree with feature and class labels.
tree_pipeline = fitted_pipelines["Decision Tree"]
tree_feature_names = tree_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

plt.figure(figsize=(18, 9))
plot_tree(
    tree_pipeline.named_steps["classifier"],
    feature_names=tree_feature_names,
    class_names=["Not survived", "Survived"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7,
)
plt.title("Decision tree (first three levels)")
plt.tight_layout()
plt.savefig("analytics/plots/decision_tree.png", dpi=150)
plt.close()

# Compare baseline, class-weighted, and SMOTE approaches.
imbalance_models = {
    "Baseline": fitted_pipelines["Logistic Regression"],
    "Class weight balanced": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
    "SMOTE on training data": ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    ),
}

imbalance_rows = []

for approach, model in imbalance_models.items():
    if approach != "Baseline":
        # SMOTE runs only while fitting on the training split.
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    imbalance_rows.append(
        {
            "approach": approach,
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
    )

imbalance_table = pd.DataFrame(imbalance_rows)
print("\nImbalance handling comparison:")
print(imbalance_table.round(3).to_string(index=False))
imbalance_table.to_csv("analytics/imbalance_comparison.csv", index=False)

# Tune Random Forest and report its out-of-bag score.
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 8],
        "classifier__max_features": ["sqrt", 0.8],
    },
    scoring="f1",
    cv=3,
    n_jobs=1,
)
grid_search.fit(X_train, y_train)

best_rf_pipeline = grid_search.best_estimator_
print("\nRandom Forest best parameters:")
print(grid_search.best_params_)
print(
    "Random Forest OOB score:",
    round(best_rf_pipeline.named_steps["classifier"].oob_score_, 3),
)

# Predict fare using a separate train/test split.
regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked",
]
X_reg = df[regression_features]
y_reg = df["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42,
)

regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            ["survived", "pclass", "age", "sibsp", "parch"],
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            ["sex", "embarked"],
        ),
    ]
)

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", regression_preprocessor),
        ("regressor", LinearRegression()),
    ]
)
regression_pipeline.fit(X_reg_train, y_reg_train)

fare_predictions = regression_pipeline.predict(X_reg_test)
residuals = y_reg_test - fare_predictions
r2 = r2_score(y_reg_test, fare_predictions)

encoded_regression_test = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_test)
feature_count = encoded_regression_test.shape[1]
sample_count = len(y_reg_test)

adjusted_r2 = 1 - (1 - r2) * (sample_count - 1) / (
    sample_count - feature_count - 1
)

regression_metrics = {
    "MAE": mean_absolute_error(y_reg_test, fare_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_reg_test, fare_predictions)),
    "R2": r2,
    "Adjusted_R2": adjusted_r2,
}

print("\nFare regression metrics:")
for metric_name, value in regression_metrics.items():
    print(f"{metric_name}: {value:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(fare_predictions, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual (actual - predicted)")
plt.title("Fare regression residual plot")
plt.tight_layout()
plt.savefig("analytics/plots/fare_regression_residuals.png", dpi=150)
plt.close()

# Keep classification and regression metrics in separate columns.
comparison = metrics_table.copy()
for column in ["MAE", "RMSE", "R2", "Adjusted_R2"]:
    comparison[column] = np.nan

regression_row = {
    "model": "Linear Regression (fare)",
    "accuracy": np.nan,
    "precision": np.nan,
    "recall": np.nan,
    "f1": np.nan,
    "auc": np.nan,
    **regression_metrics,
}

comparison = pd.concat(
    [comparison, pd.DataFrame([regression_row])],
    ignore_index=True,
)

print("\nFinal model comparison:")
print(comparison.round(3).to_string(index=False))
comparison.to_csv("analytics/model_comparison.csv", index=False)

# Save and reload the complete tuned pipeline, including preprocessing.
joblib.dump(best_rf_pipeline, "analytics/best_titanic_pipeline.joblib")
reloaded_pipeline = joblib.load("analytics/best_titanic_pipeline.joblib")

print("\nReloaded pipeline prediction from raw test features:")
print(reloaded_pipeline.predict(X_test.iloc[[0]]))
print("\nSaved analytics/best_titanic_pipeline.joblib")