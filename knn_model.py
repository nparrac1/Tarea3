import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Paso 1: Cargar y preparar los datos (idéntico a los anteriores)

# Cargar datos
df = pd.read_csv("personality_synthetic_dataset.csv")

# Codificar la variable objetivo
le = LabelEncoder()
df["target"] = le.fit_transform(df["personality_type"])

# Dividir X e y
X = df.drop(columns=["personality_type", "target"])
y = df["target"]

# Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


# Paso 2: Entrenar y evaluar el modelo KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Crear y entrenar el modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predecir
y_pred = knn.predict(X_test)

# Resultados
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Paso 3:  Graficar la matriz de confusión

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - KNN")
plt.tight_layout()
plt.show()
