import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/titanic.csv")

print("Dataset loaded")
print("Shape:", df.shape)


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Whether cabin information exists
df["CabinKnown"] = df["Cabin"].notna().astype(int)

# Extract title from passenger name
df["Title"] = df["Name"].str.extract(
    r",\s*([^.]*)\.",
    expand=False
)

# Normalize equivalent / rare titles
df["Title"] = df["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "the Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare"
})

print("\nTitle distribution:")
print(df["Title"].value_counts())


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "CabinKnown",
    "Title"
]

X = df[features]
y = df["Survived"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


# ==========================================
# 5. DEFINE FEATURE TYPES
# ==========================================

numerical_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "CabinKnown"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Title"
]


# ==========================================
# 6. NUMERICAL PIPELINE
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 7. CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 8. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully.")


# ==========================================
# 9. LOGISTIC REGRESSION MODEL
# ==========================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ==========================================
# 10. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

print("Model training completed.")


# ==========================================
# 11. LOGISTIC REGRESSION PREDICTIONS
# ==========================================

y_pred = logistic_model.predict(X_test)

print("\nFirst 20 predictions:")
print(y_pred[:20])

print("\nFirst 20 actual values:")
print(y_test.values[:20])


# ==========================================
# 12. LOGISTIC REGRESSION EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 13. RANDOM FOREST
# ==========================================

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

print("\nTraining Random Forest...")

random_forest_model.fit(X_train, y_train)

print("Random Forest training completed.")


# ==========================================
# 14. RANDOM FOREST PREDICTIONS
# ==========================================

rf_pred = random_forest_model.predict(X_test)


# ==========================================
# 15. RANDOM FOREST EVALUATION
# ==========================================

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(f"\nAccuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==========================================
# 16. LOGISTIC REGRESSION HYPERPARAMETER TUNING
# ==========================================

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    logistic_model,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\n" + "=" * 50)
print("TUNING LOGISTIC REGRESSION")
print("=" * 50)

print("\nTuning Logistic Regression...")

grid_search.fit(X_train, y_train)

print("Tuning completed.")

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(f"{grid_search.best_score_:.4f}")


# ==========================================
# 17. EVALUATE TUNED MODEL
# ==========================================

tuned_model = grid_search.best_estimator_

tuned_pred = tuned_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_precision = precision_score(y_test, tuned_pred)
tuned_recall = recall_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred)

print("\n" + "=" * 50)
print("TUNED LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print(f"\nAccuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_pred))


# ==========================================
# 18. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

feature_names = (
    numerical_features
    + list(
        random_forest_model
        .named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )
)

importances = (
    random_forest_model
    .named_steps["classifier"]
    .feature_importances_
)

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n" + "=" * 50)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 50)

print(feature_importance_df)