import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score)
import warnings
warnings.filterwarnings('ignore')

# Configurar el estilo de las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n" + "="*80)
print("CLASIFICACIÓN CON DATASET REAL: IRIS")
print("="*80)

# ============================================================
# 1. CARGA DEL DATASET IRIS
# ============================================================
print("\n1. CARGA DEL DATASET IRIS")
print("-" * 80)

# Cargar el dataset iris
iris = load_iris()
X = iris.data  # Características (features)
y = iris.target  # Etiqueta (target)

# Crear un DataFrame para mejor manejo
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['target_name'] = df['target'].map({0: iris.target_names[0], 
                                       1: iris.target_names[1], 
                                       2: iris.target_names[2]})

print("\nDataset Iris cargado exitosamente")
print(f"Número de muestras: {X.shape[0]}")
print(f"Número de características: {X.shape[1]}")
print(f"Número de clases: {len(iris.target_names)}")
print(f"Clases: {iris.target_names}")
print(f"Características: {iris.feature_names}")

# ============================================================
# 2. EXPLORACIÓN INICIAL DE DATOS
# ============================================================
print("\n2. EXPLORACIÓN INICIAL")
print("-" * 80)

print("\nPrimeras 10 filas del dataset:")
print(df.head(10))

print("\nÚltimas 5 filas del dataset:")
print(df.tail(5))

print(f"\nForma del dataset: {df.shape}")
print(f"Tipos de datos:\n{df.dtypes}")

# ============================================================
# 3. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("\n3. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

print("\nEstadísticas generales:")
print(df.describe())

print("\nDistribución de clases:")
print(df['target_name'].value_counts())
print("\nProporción de clases:")
print(df['target_name'].value_counts(normalize=True))

print("\nVerificación de valores faltantes:")
print(df.isnull().sum())

print("\nEstadísticas por clase:")
for clase in iris.target_names:
    print(f"\n{clase}:")
    print(df[df['target_name'] == clase].describe().round(3))

# ============================================================
# 4. VISUALIZACIÓN EXPLORATORIA - PARTE 1
# ============================================================
print("\n4. VISUALIZACIONES EXPLORATORIAS - PARTE 1")
print("-" * 80)

# Histogramas de cada característica por clase
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(iris.feature_names):
    for clase_idx, clase in enumerate(iris.target_names):
        datos = df[df['target'] == clase_idx][feature]
        axes[idx].hist(datos, alpha=0.6, label=clase, bins=15)
    
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Distribución de {feature} por Clase', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('imgs/03_clasificacion_iris_1_histogramas_features.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 5. VISUALIZACIÓN EXPLORATORIA - PARTE 2
# ============================================================
print("\n5. VISUALIZACIONES EXPLORATORIAS - PARTE 2")
print("-" * 80)

# Scatter plot pairwise de características principales
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Largo vs Ancho del Sépalo
for clase_idx, clase in enumerate(iris.target_names):
    datos = df[df['target'] == clase_idx]
    axes[0].scatter(datos['sepal length (cm)'], datos['sepal width (cm)'], 
                   label=clase, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

axes[0].set_xlabel('Largo del Sépalo (cm)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Ancho del Sépalo (cm)', fontsize=12, fontweight='bold')
axes[0].set_title('Características del Sépalo', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Largo vs Ancho del Pétalo
for clase_idx, clase in enumerate(iris.target_names):
    datos = df[df['target'] == clase_idx]
    axes[1].scatter(datos['petal length (cm)'], datos['petal width (cm)'], 
                   label=clase, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

axes[1].set_xlabel('Largo del Pétalo (cm)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Ancho del Pétalo (cm)', fontsize=12, fontweight='bold')
axes[1].set_title('Características del Pétalo', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/03_clasificacion_iris_2_scatter_sepalos_petalos.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 6. MATRIZ DE CORRELACIÓN
# ============================================================
print("\n6. MATRIZ DE CORRELACIÓN")
print("-" * 80)

# Calcular correlación
correlation_matrix = df[iris.feature_names].corr()
print("\nMatriz de correlación:")
print(correlation_matrix.round(3))

# Visualizar matriz de correlación
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlación - Dataset Iris', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('imgs/03_clasificacion_iris_3_matriz_correlacion.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 7. PREPARACIÓN DE DATOS PARA CLASIFICACIÓN
# ============================================================
print("\n7. PREPARACIÓN DE DATOS")
print("-" * 80)

# Dividir en entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")

print("\nDistribución de clases en entrenamiento:")
for clase_idx, clase in enumerate(iris.target_names):
    count = np.sum(y_train == clase_idx)
    print(f"  {clase}: {count}")

print("\nDistribución de clases en prueba:")
for clase_idx, clase in enumerate(iris.target_names):
    count = np.sum(y_test == clase_idx)
    print(f"  {clase}: {count}")

# Normalización de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nDatos escalados - Media (debería ser ~0):")
print(X_train_scaled.mean(axis=0).round(4))
print("\nDatos escalados - Desv. Est. (debería ser ~1):")
print(X_train_scaled.std(axis=0).round(4))

# ============================================================
# 8. MODELO 1: REGRESIÓN LOGÍSTICA
# ============================================================
print("\n8. MODELO 1: REGRESIÓN LOGÍSTICA")
print("-" * 80)

modelo_lr = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
modelo_lr.fit(X_train_scaled, y_train)

# Predicciones
y_pred_lr = modelo_lr.predict(X_test_scaled)
y_pred_proba_lr = modelo_lr.predict_proba(X_test_scaled)

print("\nModelo de Regresión Logística entrenado")
print("Coeficientes del modelo:")
for clase_idx, clase in enumerate(iris.target_names):
    print(f"\n  Clase '{clase}':")
    for feat_idx, feature in enumerate(iris.feature_names):
        print(f"    {feature}: {modelo_lr.coef_[clase_idx][feat_idx]:.4f}")

# ============================================================
# 9. MODELO 2: RANDOM FOREST
# ============================================================
print("\n9. MODELO 2: RANDOM FOREST")
print("-" * 80)

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train_scaled, y_train)

# Predicciones
y_pred_rf = modelo_rf.predict(X_test_scaled)
y_pred_proba_rf = modelo_rf.predict_proba(X_test_scaled)

print("\nModelo Random Forest entrenado")
print("Importancia de características:")
for feat_idx, feature in enumerate(iris.feature_names):
    print(f"  {feature}: {modelo_rf.feature_importances_[feat_idx]:.4f}")

# ============================================================
# 10. EVALUACIÓN - REGRESIÓN LOGÍSTICA
# ============================================================
print("\n10. EVALUACIÓN - REGRESIÓN LOGÍSTICA")
print("-" * 80)

acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nAccuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nMatriz de Confusión:")
print(cm_lr)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))

# ============================================================
# 11. EVALUACIÓN - RANDOM FOREST
# ============================================================
print("\n11. EVALUACIÓN - RANDOM FOREST")
print("-" * 80)

acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nAccuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nMatriz de Confusión:")
print(cm_rf)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# ============================================================
# 12. VISUALIZACIÓN DE MATRICES DE CONFUSIÓN
# ============================================================
print("\n12. VISUALIZACIÓN DE MATRICES DE CONFUSIÓN")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regresión Logística
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[0].set_xlabel('Predicción', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Real', fontsize=12, fontweight='bold')
axes[0].set_title(f'Matriz de Confusión - Regresión Logística\n(Accuracy: {acc_lr:.2%})', 
                  fontsize=12, fontweight='bold')

# Random Forest
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[1].set_xlabel('Predicción', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Real', fontsize=12, fontweight='bold')
axes[1].set_title(f'Matriz de Confusión - Random Forest\n(Accuracy: {acc_rf:.2%})', 
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# 13. COMPARACIÓN DE MODELOS
# ============================================================
print("\n13. COMPARACIÓN DE MODELOS")
print("-" * 80)

# Calcular métricas
modelos = ['Regresión Logística', 'Random Forest']
accuracy_scores = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_rf)
]

# Precision por clase
precision_lr = [precision_score(y_test, y_pred_lr, labels=[0], average=None)[0],
                 precision_score(y_test, y_pred_lr, labels=[1], average=None)[0],
                 precision_score(y_test, y_pred_lr, labels=[2], average=None)[0]]

precision_rf = [precision_score(y_test, y_pred_rf, labels=[0], average=None)[0],
                precision_score(y_test, y_pred_rf, labels=[1], average=None)[0],
                precision_score(y_test, y_pred_rf, labels=[2], average=None)[0]]

print("\nComparación de Accuracy:")
print(f"  Regresión Logística: {accuracy_scores[0]:.4f}")
print(f"  Random Forest: {accuracy_scores[1]:.4f}")

# Visualizar comparación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(modelos, accuracy_scores, color=['steelblue', 'forestgreen'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Comparación de Accuracy', fontsize=13, fontweight='bold')
axes[0].set_ylim([0.9, 1.0])
for i, v in enumerate(accuracy_scores):
    axes[0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Precision by class
x = np.arange(len(iris.target_names))
width = 0.35
axes[1].bar(x - width/2, precision_lr, width, label='Regresión Logística', 
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].bar(x + width/2, precision_rf, width, label='Random Forest', 
            color='forestgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
axes[1].set_title('Precision por Clase', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(iris.target_names)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('imgs/03_clasificacion_iris_4_comparacion_matrices_metricas.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 14. PREDICCIONES EN EJEMPLOS NUEVOS
# ============================================================
print("\n14. PREDICCIONES EN EJEMPLOS")
print("-" * 80)

# Crear ejemplos ficticios
ejemplos = np.array([
    [5.0, 3.5, 1.3, 0.3],  # Probable Setosa
    [6.0, 2.7, 5.1, 1.6],  # Probable Versicolor
    [7.5, 3.0, 6.5, 2.0]   # Probable Virginica
])

# Normalizar ejemplos
ejemplos_scaled = scaler.transform(ejemplos)

print("\nPredicciones para nuevos ejemplos:")
print(f"{'Largo Sépalo':<15} {'Ancho Sépalo':<15} {'Largo Pétalo':<15} {'Ancho Pétalo':<15}")
print(f"{'-'*60}")
for i, ejemplo in enumerate(ejemplos):
    print(f"{ejemplo[0]:<15.1f} {ejemplo[1]:<15.1f} {ejemplo[2]:<15.1f} {ejemplo[3]:<15.1f}")

print(f"\n{'Modelo':<25} {'Predicción':<20} {'Confianza':<15}")
print(f"{'-'*60}")

for modelo_name, modelo, X_scaled in [('Regresión Logística', modelo_lr, ejemplos_scaled),
                                       ('Random Forest', modelo_rf, ejemplos_scaled)]:
    predicciones = modelo.predict(X_scaled)
    probabilidades = modelo.predict_proba(X_scaled)
    
    for i, (pred, probs) in enumerate(zip(predicciones, probabilidades)):
        clase_nombre = iris.target_names[pred]
        confianza = np.max(probs)
        print(f"{modelo_name:<25} {clase_nombre:<20} {confianza:<15.4f}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"""
Dataset: IRIS (150 muestras, 4 características, 3 clases)
Características: {', '.join(iris.feature_names)}
Clases: {', '.join(iris.target_names)}

Split: 80% entrenamiento (120 muestras), 20% prueba (30 muestras)

RESULTADOS:
  Regresión Logística: {accuracy_scores[0]:.2%} de accuracy
  Random Forest:       {accuracy_scores[1]:.2%} de accuracy

El mejor modelo es: {'Random Forest' if accuracy_scores[1] > accuracy_scores[0] else 'Regresión Logística'}
""")
print("="*80 + "\n")
