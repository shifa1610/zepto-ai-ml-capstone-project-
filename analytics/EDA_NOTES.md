# Titanic EDA Notes

## Survival by sex
Women: 74% survived. Men: 18.9% survived.

My interpretation:
## Survival by passenger class
1st class: 62.6%; 2nd class: 47.3%; 3rd class: 24.2%.

My interpretation:
## Survival by sex and passenger class
Women: 1st class 96.7%, 2nd class 92.1%, 3rd class 50.0%.
Men: 1st class 36.9%, 2nd class 15.7%, 3rd class 13.5%.

My interpretation:
## Age and fare by survival outcome
Open `analytics/plots/age_fare_by_survival.png`. It has two panels: age on the left and fare on the right. In each box plot, the line inside the box is the median.

My interpretation:
## Correlations
The strongest pair is pclass and fare, with correlation -0.548.
The second strongest pair is sibsp and parch, with correlation 0.415.

My interpretation:
## Age and fare distributions
Age has 65 IQR outliers. Fare has 114 IQR outliers.
Fare mean: 32.10; median: 14.45; mode: 8.05.

My interpretation:
## Missing values and cleaning decisions
- age: 19.87% missing; filled missing values with the median.
- embarked: 0.22% missing; dropped rows with missing values.
- embark_town: 0.22% missing; dropped rows with missing values.
- deck: 77.22% missing; dropped the column because so much of it was missing.
- Cleaned dataset: 889 rows and 14 columns.

## Standardization check
After z-score standardization, age and fare both had mean 0 and standard deviation 1. This confirms the exploratory standardization worked.
## Modeling results
Class balance: 61.8% did not survive and 38.2% survived.

SMOTE had the highest F1 score (0.735). Class weighting had the highest recall (0.750). Baseline F1 was 0.734.

Fare regression: MAE 21.099, RMSE 41.702, R² 0.348, adjusted R² 0.309.

### Classifier recommendation
Write 3–5 sentences in your own words. Compare the classifier scores and say which model you recommend.

### Residual plot
Write one sentence saying whether you see heteroscedasticity. Explain whether the residual spread changes as predicted fare increases.