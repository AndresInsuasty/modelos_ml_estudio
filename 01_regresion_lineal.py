import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Leer el archivo Excel
df = pd.read_excel('data/01_datos_lineales.xlsx')

# Mostrar las primeras filas del dataframe
print("Primeras filas del dataframe:")
print(df.head())
print()

# ============================================
# CALCULAR LA REGRESIÓN LINEAL
# ============================================

# Preparar los datos: X debe ser una matriz 2D
X = df['x'].values.reshape(-1, 1)  # Convertir a matriz (n_muestras, 1)
y = df['y'].values                   # y puede ser un vector 1D

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Obtener los parámetros de la ecuación lineal: y = mx + b
pendiente = modelo.coef_[0]      # m (slope)
intercepto = modelo.intercept_   # b (intercept)

# Mostrar la ecuación lineal en consola
print("=" * 50)
print("ECUACIÓN DE LA REGRESIÓN LINEAL")
print("=" * 50)
print(f"Ecuación: y = {pendiente:.4f}x + {intercepto:.4f}")
print(f"  - Pendiente (m): {pendiente:.4f}")
print(f"  - Intercepto (b): {intercepto:.4f}")
print("=" * 50)
print()

# Calcular los valores predichos (la línea de regresión)
y_predicho = modelo.predict(X)

# Calcular el coeficiente de determinación (R²)
r2 = modelo.score(X, y)
print(f"Coeficiente de determinación (R²): {r2:.4f}")
print("(Indica qué tan bien el modelo explica los datos)")
print()

# ============================================
# GRÁFICA 1: SOLO LOS PUNTOS (SCATTER PLOT)
# ============================================

plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], color='blue', s=100, alpha=0.7, 
            edgecolors='navy', linewidth=1.5)
plt.xlabel('x', fontsize=12, fontweight='bold')
plt.ylabel('y', fontsize=12, fontweight='bold')
plt.title('Scatter Plot - Datos Originales', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# ============================================
# GRÁFICA 2: SCATTER PLOT + REGRESIÓN LINEAL
# ============================================

plt.figure(figsize=(12, 7))

# Gráfica scatter de los datos originales
plt.scatter(df['x'], df['y'], color='blue', s=100, alpha=0.6, 
            label='Datos originales', edgecolors='navy', linewidth=1.5)

# Gráfica de la línea de regresión lineal
# Ordenar por x para que la línea se vea suave
indices_ordenados = np.argsort(X.flatten())
plt.plot(X[indices_ordenados], y_predicho[indices_ordenados], 
         color='red', linewidth=3, label=f'Regresión lineal: y = {pendiente:.4f}x + {intercepto:.4f}')

plt.xlabel('x', fontsize=12, fontweight='bold')
plt.ylabel('y', fontsize=12, fontweight='bold')
plt.title('Regresión Lineal - Datos vs Modelo', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()
