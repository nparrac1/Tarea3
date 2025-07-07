import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("personality_synthetic_dataset.csv")

# Codificar la clase
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["target"] = le.fit_transform(df["personality_type"])  # 0 = Ambivert, etc.

# Calcular correlaciones (solo variables numéricas)
corr = df.corr(numeric_only=True)

# Ordenar por correlación con la clase
corr_target = corr["target"].sort_values(ascending=False)

# Mostrar en consola
print("\nCorrelación de cada variable con la clase (target):")
print(corr_target)

# (Opcional) Mostrar como gráfico de calor
plt.figure(figsize=(10, 12))
sns.heatmap(corr_target.to_frame(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlación de cada variable con la clase (target)")
plt.tight_layout()
plt.show()
