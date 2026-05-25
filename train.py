import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("train.csv")

# Pré-processamento
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Fare"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "SibSp"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

acc = accuracy_score(y_test, modelo.predict(X_test))
print(f"Acurácia: {acc:.2%}")

joblib.dump(modelo, "modelo.pkl")
print("Modelo salvo em modelo.pkl")
