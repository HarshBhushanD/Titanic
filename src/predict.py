import joblib
import pandas as pd


# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================

model_path = "models/titanic_model.joblib"

model = joblib.load(model_path)

print("Model loaded successfully.")


# ==========================================
# 2. CREATE PASSENGER DATA
# ==========================================

passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "FamilySize": 2,
    "CabinKnown": 0,
    "Title": "Mr"
}])


# ==========================================
# 3. MAKE PREDICTION
# ==========================================

prediction = model.predict(passenger)[0]

probability = model.predict_proba(passenger)[0]


# ==========================================
# 4. DISPLAY RESULT
# ==========================================

print("\n" + "=" * 50)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 50)

print("\nPassenger details:")
print(passenger.to_string(index=False))

print("\nPrediction:")

if prediction == 1:
    print("Survived")
else:
    print("Did not survive")

print("\nPrediction probabilities:")
print(f"Did not survive: {probability[0]:.4f}")
print(f"Survived       : {probability[1]:.4f}")