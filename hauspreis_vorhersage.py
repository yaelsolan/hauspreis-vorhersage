# 🏠 Hauspreis-Vorhersage mit Machine Learning (scikit-learn)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Daten laden
df = pd.read_csv("hauspreise.csv")

# 2. Features und Zielvariable definieren
X = df[["Zimmer", "Größe_m2"]]
y = df["Preis_1000€"]

# 3. Trainings- und Testdaten erzeugen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Modell erstellen und trainieren
modell = LinearRegression()
modell.fit(X_train, y_train)

# 5. Vorhersage machen
y_pred = modell.predict(X_test)

# 6. Ergebnisse ausgeben
print("🏠 Vorhersageergebnisse:")
for i in range(len(y_test)):
    print(f"Erwartet: {y_test.iloc[i]} | Vorhergesagt: {round(y_pred[i], 1)}")

# 7. Fehler berechnen
mse = mean_squared_error(y_test, y_pred)
print(f"📉 Mittlerer quadratischer Fehler (MSE): {round(mse, 2)}")

# 8. Visualisierung
plt.scatter(y_test, y_pred)
plt.xlabel("Tatsächlicher Preis (in 1000€)")
plt.ylabel("Vorhergesagter Preis (in 1000€)")
plt.title("📊 Tatsächlich vs. Vorhergesagt")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.grid(True)
plt.show()
