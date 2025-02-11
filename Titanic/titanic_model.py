import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("tested.csv")
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

imputer = SimpleImputer(strategy="mean")
df["Age"] = imputer.fit_transform(df[["Age"]])  
df["Fare"] = imputer.fit_transform(df[["Fare"]])  
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  

label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])  
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])  

X = df.drop("Survived", axis=1)  
y = df["Survived"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

predictions = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print("\nFirst 5 Predictions:\n", predictions.head())

