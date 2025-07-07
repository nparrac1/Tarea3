import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix





# Cargar el archivo CSV
df = pd.read_csv("personality_synthetic_dataset.csv")

# Mostrar las primeras filas
print(df.head())

# Información general del dataset
print(df.info())

# Conteo de clases (corregido)
print("Distribución de clases:")
print(df['personality_type'].value_counts())

# Codificar la variable objetivo
le = LabelEncoder()
df['target'] = le.fit_transform(df['personality_type'])  # Asigna: 0 = Ambivert, 1 = Extrovert, 2 = Introvert (por ejemplo)

# Separar X (entradas) e y (salida)
X = df.drop(['personality_type', 'target'], axis=1)
y = df['target']

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Verificación rápida
print("Tamaño de entrenamiento:", X_train.shape)
print("Tamaño de prueba:", X_test.shape)
print("Clases codificadas:", list(le.classes_))





# Crear el modelo de regresión logística multinomial
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))



# Crear la matriz
cm = confusion_matrix(y_test, y_pred)

# Visualizar con seaborn
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Regresión Logística")
plt.tight_layout()
plt.show()




