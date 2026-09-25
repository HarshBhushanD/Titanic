import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/titanic.csv")

print("Titanic Dataset")
print("=" * 50)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Survived"].value_counts())

print("\nTarget distribution (%):")
print(df["Survived"].value_counts(normalize=True) * 100)

print("\nNumerical statistics:")
print(df.describe())

print("\nUnique values:")
print(df.nunique())

print("\n" + "=" * 50)
print("SURVIVAL ANALYSIS")
print("=" * 50)

print("\nSurvival rate by Sex:")
print(df.groupby("Sex")["Survived"].mean())

print("\nSurvival rate by Passenger Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("\nSurvival count by Sex:")
print(pd.crosstab(df["Sex"], df["Survived"]))

print("\nSurvival count by Passenger Class:")
print(pd.crosstab(df["Pclass"], df["Survived"]))

print("\nCreating EDA plots...")

sns.set_theme(style="whitegrid")

# 1. Survival distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.show()

# 2. Survival by sex
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Sex", y="Survived")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

# 3. Survival by passenger class
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Pclass", y="Survived")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# 4. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# 5. Age vs Survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Survived", y="Age")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()

# 6. Family size distribution

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nFamily size distribution:")
print(df["FamilySize"].value_counts().sort_index())

print("\nSurvival rate by FamilySize:")
print(df.groupby("FamilySize")["Survived"].mean())

plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="FamilySize", y="Survived")
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.show()

# 7. Fare distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Fare", bins=40, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()

# 8. Fare vs Survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Survived", y="Fare")
plt.title("Fare Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Fare")
plt.show()

# 9. Cabin availability
df["CabinKnown"] = df["Cabin"].notna().astype(int)

print("\nCabin availability:")
print(df["CabinKnown"].value_counts())

print("\nSurvival rate by cabin availability:")
print(df.groupby("CabinKnown")["Survived"].mean())

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="CabinKnown", y="Survived")
plt.title("Survival Rate by Cabin Availability")
plt.xlabel("Cabin Information Available (0 = No, 1 = Yes)")
plt.ylabel("Survival Rate")
plt.show()

print("\nEmbarked values:")
print(df["Embarked"].value_counts())

print("\nSurvival rate by Embarked:")
print(df.groupby("Embarked")["Survived"].mean())

plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="Embarked", y="Survived")
plt.title("Survival Rate by Embarked Port")
plt.xlabel("Port")
plt.ylabel("Survival Rate")
plt.show()