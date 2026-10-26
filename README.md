# Titanic Survival Classification

A complete machine learning classification project using the Titanic dataset to predict whether a passenger survived the disaster.

The project demonstrates an end-to-end machine learning workflow including exploratory data analysis, feature engineering, preprocessing, model training, evaluation, cross-validation, hyperparameter tuning, model persistence, and prediction.

---

## Project Overview

The objective is to predict the `Survived` target variable:

- `0` → Did not survive
- `1` → Survived

The dataset contains information about passengers such as:

- Passenger class
- Sex
- Age
- Number of siblings/spouses
- Number of parents/children
- Fare
- Port of embarkation
- Cabin information
- Passenger name

---

## Dataset

The dataset contains:

- 891 passengers
- 12 original features
- 1 binary target variable

### Target Distribution

| Outcome | Count | Percentage |
|---|---:|---:|
| Did not survive | 549 | 61.62% |
| Survived | 342 | 38.38% |

The target is therefore somewhat imbalanced, making metrics such as precision, recall, and F1-score useful alongside accuracy.

---

# Exploratory Data Analysis

## Missing Values

The dataset contains missing values in:

| Feature | Missing Values |
|---|---:|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

`Cabin` contains a large proportion of missing values, so instead of attempting to reconstruct the exact cabin information, a derived `CabinKnown` feature was created.

---

## Survival by Sex

Observed survival rates:

| Sex | Survival Rate |
|---|---:|
| Female | 74.20% |
| Male | 18.89% |

Sex is therefore one of the strongest predictors in the dataset.

---

## Survival by Passenger Class

| Passenger Class | Survival Rate |
|---|---:|
| 1st | 62.96% |
| 2nd | 47.28% |
| 3rd | 24.24% |

Passenger class also shows a strong relationship with survival.

---

# Feature Engineering

Three additional features were created.

### FamilySize

```python
FamilySize = SibSp + Parch + 1

