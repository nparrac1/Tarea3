# Clasificación de tipos de personalidad
# Dataset: personality_synthetic_dataset.csv
# Requisitos de Tarea 3: Evaluación de 3 clasificadores, visualizaciones, preprocesamiento, y reducción de dimensionalidad.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# --- CARGA Y DESCRIPCION DEL DATASET ---

# 1. Cargar el dataset
print("\n--- Carga de datos ---")
df = pd.read_csv("personality_synthetic_dataset.csv")
print(df.info())
print("\nDistribución de clases:")
print(df['personality_type'].value_counts())

# 2. Verificar valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# 3. Codificar variable objetivo
df['target'] = LabelEncoder().fit_transform(df['personality_type'])

# 4. Mostrar correlación con la clase
df_corr = df.corr(numeric_only=True)
corr_target = df_corr['target'].sort_values(ascending=False)
print("\nCorrelación de variables con la clase:")
print(corr_target)

plt.figure(figsize=(10, 12))
sns.heatmap(corr_target.to_frame(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlación con la clase (target)")
plt.tight_layout()
plt.show()

# 5. Histograma de distribución
print("\n--- Histogramas de distribución ---")
df.drop(columns=['personality_type', 'target'], errors='ignore').hist(bins=20, figsize=(15, 12))
plt.suptitle("Distribución de las variables", fontsize=16)
plt.tight_layout()
plt.show()

# 6. Outliers con IQR
df_num = df.select_dtypes(include='number')  # Solo columnas numéricas
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1

# Contar outliers por variable
outliers = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers por variable:")
print(outliers.sort_values(ascending=False))


# --- PREPROCESAMIENTO Y DIVISIÓN ---

X = df.drop(columns=['personality_type', 'target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# --- REDUCCIÓN DE DIMENSIONALIDAD CON PCA ---

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\nExplained variance ratio (PCA):")
print(pca.explained_variance_ratio_)

# --- FUNCION AUXILIAR PARA GRAFICO DE MATRIZ ---
def graficar_matriz(y_true, y_pred, title, etiquetas, cmap):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=etiquetas, yticklabels=etiquetas)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- MODELO 1: REGRESION LOGISTICA MULTICLASE ---

print("\n--- Modelo 1: Regresión Logística ---")
log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_model.fit(X_train_pca, y_train)
y_pred_log = log_model.predict(X_test_pca)

print(classification_report(y_test, y_pred_log, target_names=LabelEncoder().fit(df['personality_type']).classes_))
graficar_matriz(y_test, y_pred_log, "Matriz de Confusión - Regresión Logística", LabelEncoder().fit(df['personality_type']).classes_, "Blues")

# --- MODELO 2: RANDOM FOREST ---

print("\n--- Modelo 2: Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)
y_pred_rf = rf_model.predict(X_test_pca)

print(classification_report(y_test, y_pred_rf, target_names=LabelEncoder().fit(df['personality_type']).classes_))
graficar_matriz(y_test, y_pred_rf, "Matriz de Confusión - Random Forest", LabelEncoder().fit(df['personality_type']).classes_, "Greens")

# --- MODELO 3: KNN ---

print("\n--- Modelo 3: K-Nearest Neighbors ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_pca, y_train)
y_pred_knn = knn_model.predict(X_test_pca)

print(classification_report(y_test, y_pred_knn, target_names=LabelEncoder().fit(df['personality_type']).classes_))
graficar_matriz(y_test, y_pred_knn, "Matriz de Confusión - KNN", LabelEncoder().fit(df['personality_type']).classes_, "Oranges")
