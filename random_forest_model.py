import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Paso 1: Cargar y preparar los datos

# Cargar el dataset
df = pd.read_csv("personality_synthetic_dataset.csv")

# Codificar etiquetas (Ambivert = 0, Extrovert = 1, Introvert = 2)
le = LabelEncoder()
df["target"] = le.fit_transform(df["personality_type"])

# Separar características (X) y etiquetas (y)
X = df.drop(columns=["personality_type", "target"])
y = df["target"]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en entrenamiento y prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


#Paso 2: Entrenar el modelo Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Crear y entrenar el clasificador
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Mostrar resultados
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


#Paso 3  Graficar la matriz de confusión



import seaborn as sns
import matplotlib.pyplot as plt

# Generar matriz
cm = confusion_matrix(y_test, y_pred)

# Visualizar con seaborn
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Random Forest")
plt.tight_layout()
plt.show()
