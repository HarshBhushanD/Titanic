import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Normalize titles
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


# ==========================================
# 5. FEATURE TYPES
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
# 6. PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully.")


# ==========================================
# 7. LOGISTIC REGRESSION PIPELINE
# ==========================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ==========================================
# 8. HYPERPARAMETER TUNING
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

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1:")
print(f"{grid_search.best_score_:.4f}")


# ==========================================
# 9. FINAL MODEL
# ==========================================

final_model = grid_search.best_estimator_

print("\nFinal model selected.")


# ==========================================
# 10. TEST SET EVALUATION
# ==========================================

y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("FINAL MODEL RESULTS")
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
# 11. SAVE FINAL MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model_path = "models/titanic_model.joblib"

joblib.dump(final_model, model_path)

print("\n" + "=" * 50)
print("MODEL SAVED")
print("=" * 50)

print(f"\nSaved to: {model_path}")