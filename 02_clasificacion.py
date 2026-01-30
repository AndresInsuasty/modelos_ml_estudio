import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Configurar el estilo de las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n" + "="*70)
print("ANÁLISIS COMPLETO DE CLASIFICACIÓN CON REGRESIÓN LOGÍSTICA")
print("="*70)

# ============================================================
# 1. LECTURA Y EXPLORACIÓN DE DATOS
# ============================================================
print("\n1. LECTURA DE DATOS")
print("-" * 70)

df = pd.read_excel('data/03_clasificacion.xlsx')
print("Primeras filas del dataset:")
print(df.head(10))
print(f"\nForma del dataset: {df.shape}")
print(f"Tipos de datos:\n{df.dtypes}")

# ============================================================
# 2. ANÁLISIS EXPLORATORIO
# ============================================================
print("\n2. ANÁLISIS EXPLORATORIO")
print("-" * 70)

print("\nEstadísticas descriptivas:")
print(df.describe())

print("\nDistribución de clases (variable objetivo 'z'):")
print(df['z'].value_counts())
print("\nProporción de clases:")
print(df['z'].value_counts(normalize=True))

print("\nVerificación de valores faltantes:")
print(df.isnull().sum())

# ============================================================
# 3. VISUALIZACIÓN EXPLORATORIA
# ============================================================
print("\n3. GRÁFICAS EXPLORATORIAS")
print("-" * 70)

# Gráfica 1: Scatter plot coloreado por clase
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot original
scatter1 = axes[0].scatter(df[df['z']==0]['x'], df[df['z']==0]['y'], 
                           color='blue', alpha=0.7, s=100, label='Clase 0', edgecolors='navy')
scatter2 = axes[0].scatter(df[df['z']==1]['x'], df[df['z']==1]['y'], 
                           color='red', alpha=0.7, s=100, label='Clase 1', edgecolors='darkred')
axes[0].set_xlabel('x', fontsize=12, fontweight='bold')
axes[0].set_ylabel('y', fontsize=12, fontweight='bold')
axes[0].set_title('Distribución de Clases en el Espacio de Características', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogramas de las características por clase
axes[1].hist(df[df['z']==0]['x'], bins=20, alpha=0.6, label='Clase 0 (x)', color='blue')
axes[1].hist(df[df['z']==1]['x'], bins=20, alpha=0.6, label='Clase 1 (x)', color='red')
axes[1].set_xlabel('Valor de x', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
axes[1].set_title('Distribución de la característica X por Clase', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('imgs/02_clasificacion_1_datos_histogramas.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 4. PREPARACIÓN DE DATOS
# ============================================================
print("\n4. PREPARACIÓN DE DATOS")
print("-" * 70)

# Separar características y etiqueta
X = df[['x', 'y']]  # Características
y = df['z']         # Etiqueta (objetivo)

# Split train-test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")
print("\nDistribución en entrenamiento:")
print(y_train.value_counts())
print("\nDistribución en prueba:")
print(y_test.value_counts())

# Normalización de datos (estandarización)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDatos escalados (muestra de entrenamiento):")
print(f"Media: {X_train_scaled.mean(axis=0)}")
print(f"Desv. Est.: {X_train_scaled.std(axis=0)}")

# ============================================================
# 5. MODELO DE REGRESIÓN LOGÍSTICA
# ============================================================
print("\n5. ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA")
print("-" * 70)

# Crear y entrenar el modelo
modelo = LogisticRegression(random_state=42, max_iter=1000)
modelo.fit(X_train_scaled, y_train)

print("Coeficientes del modelo:")
print(f"  - Coeficiente para x: {modelo.coef_[0][0]:.4f}")
print(f"  - Coeficiente para y: {modelo.coef_[0][1]:.4f}")
print(f"  - Intercepto (sesgo): {modelo.intercept_[0]:.4f}")

# ============================================================
# 6. PREDICCIONES
# ============================================================
print("\n6. PREDICCIONES")
print("-" * 70)

# Predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test_scaled)
y_pred_proba = modelo.predict_proba(X_test_scaled)

print("\nPrimeras 10 predicciones:")
print(f"{'Índice':<8} {'Real':<8} {'Predicho':<12} {'Prob(0)':<12} {'Prob(1)':<12}")
print("-" * 52)
for i in range(min(10, len(y_test))):
    print(f"{i:<8} {y_test.iloc[i]:<8} {y_pred[i]:<12} {y_pred_proba[i][0]:<12.4f} {y_pred_proba[i][1]:<12.4f}")

# ============================================================
# 7. MATRIZ DE CONFUSIÓN
# ============================================================
print("\n7. MATRIZ DE CONFUSIÓN")
print("-" * 70)

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# Interpretación
tn, fp, fn, tp = cm.ravel()
print("\nInterpretación:")
print(f"  - Verdaderos Negativos (TN): {tn}")
print(f"  - Falsos Positivos (FP): {fp}")
print(f"  - Falsos Negativos (FN): {fn}")
print(f"  - Verdaderos Positivos (TP): {tp}")

# Gráfica de matriz de confusión
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Clase 0', 'Clase 1'],
            yticklabels=['Clase 0', 'Clase 1'])
ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax.set_ylabel('Real', fontsize=12, fontweight='bold')
ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('imgs/02_clasificacion_2_matriz_confusion.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 8. MÉTRICAS DE RENDIMIENTO
# ============================================================
print("\n8. MÉTRICAS DE RENDIMIENTO")
print("-" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"\nAccuracy (Exactitud): {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensibilidad): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Clase 0', 'Clase 1']))

# ============================================================
# 9. CURVA ROC Y AUC
# ============================================================
print("\n9. CURVA ROC Y AUC")
print("-" * 70)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkblue', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12, fontweight='bold')
ax.set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('imgs/02_clasificacion_3_curva_roc.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 10. FRONTERA DE DECISIÓN DEL MODELO
# ============================================================
print("\n10. VISUALIZACIÓN DE LA FRONTERA DE DECISIÓN")
print("-" * 70)

# Crear una malla de puntos
h = 0.02  # Tamaño del paso en la malla
x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
y_min, y_max = X['y'].min() - 1, X['y'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predicciones en la malla
Z = modelo.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Gráfica
fig, ax = plt.subplots(figsize=(12, 8))

# Contorno de la frontera de decisión
contour = ax.contourf(xx, yy, Z, levels=1, colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

# Puntos de datos de prueba
scatter1 = ax.scatter(X_test[y_test==0]['x'], X_test[y_test==0]['y'], 
                      color='blue', s=100, alpha=0.7, label='Clase 0 (Real)', edgecolors='navy', marker='o')
scatter2 = ax.scatter(X_test[y_test==1]['x'], X_test[y_test==1]['y'], 
                      color='red', s=100, alpha=0.7, label='Clase 1 (Real)', edgecolors='darkred', marker='o')

# Predicciones incorrectas
incorrect = (y_test.values != y_pred)
if incorrect.sum() > 0:
    ax.scatter(X_test[incorrect]['x'], X_test[incorrect]['y'], 
              color='yellow', s=200, alpha=0.9, label='Predicción Incorrecta', 
              edgecolors='black', marker='X', linewidths=2)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('y', fontsize=12, fontweight='bold')
ax.set_title('Frontera de Decisión del Modelo de Regresión Logística', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('imgs/02_clasificacion_4_frontera_decision.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 11. ANÁLISIS DE PROBABILIDADES PREDICHAS
# ============================================================
print("\n11. ANÁLISIS DE PROBABILIDADES")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_pred_proba[y_test==0, 1], bins=20, alpha=0.6, label='Clase 0', color='blue')
ax.hist(y_pred_proba[y_test==1, 1], bins=20, alpha=0.6, label='Clase 1', color='red')
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Umbral de decisión (0.5)')
ax.set_xlabel('Probabilidad predicha (Clase 1)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('imgs/02_clasificacion_5_distribucion_probabilidades.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)
print("\nEl modelo de Regresión Logística alcanzó:")
print(f"  ✓ Accuracy: {accuracy:.2%}")
print(f"  ✓ Precision: {precision:.2%}")
print(f"  ✓ Recall: {recall:.2%}")
print(f"  ✓ F1-Score: {f1:.2%}")
print(f"  ✓ ROC-AUC: {roc_auc:.2%}")
print(f"\nEl modelo clasificó correctamente {(y_test == y_pred).sum()} de {len(y_test)} instancias de prueba.")
print("="*70 + "\n")
