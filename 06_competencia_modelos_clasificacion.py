"""
COMPETENCIA DE MODELOS DE CLASIFICACIÓN
========================================

Este script demuestra cómo comparar múltiples modelos de clasificación
para identificar cuál se ajusta mejor a un problema específico.

Dataset: Breast Cancer Wisconsin (Diagnóstico)
Objetivo: Predecir si un tumor es maligno (M) o benigno (B)

MODELOS A COMPARAR:
1. Regresión Logística
2. K-Nearest Neighbors (KNN)
3. Árbol de Decisión
4. Random Forest
5. Support Vector Machine (SVM)
6. Gaussian Naive Bayes
7. Gradient Boosting
8. AdaBoost

MÉTRICAS DE EVALUACIÓN:
- Accuracy: Proporción de predicciones correctas
- Precision: De los predichos como positivos, cuántos lo son realmente
- Recall: De los positivos reales, cuántos fueron detectados
- F1-Score: Media armónica de precision y recall
- ROC-AUC: Área bajo la curva ROC (capacidad discriminativa)

Autor: Script educativo para comparación de modelos
Fecha: 2026
"""

# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# Modelos de Clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Configuración
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("="*80)
print("COMPETENCIA DE MODELOS DE CLASIFICACIÓN")
print("Comparación de 8 Algoritmos en Dataset de Cáncer de Mama")
print("="*80 + "\n")

# ============================================================
# 1. CARGA Y EXPLORACIÓN DE DATOS
# ============================================================
print("\n1. CARGA Y EXPLORACIÓN DE DATOS")
print("-" * 80)

# Cargar dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Información del dataset
print(f"\n📊 INFORMACIÓN DEL DATASET:")
print(f"  - Total de muestras: {X.shape[0]}")
print(f"  - Número de características: {X.shape[1]}")
print(f"  - Clases: {data.target_names}")
print(f"  - Distribución de clases:")
print(f"    • Maligno (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"    • Benigno (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

# Crear DataFrame completo
df = pd.DataFrame(X, columns=data.feature_names)
df['diagnosis'] = y
df['diagnosis_name'] = df['diagnosis'].map({0: 'Maligno', 1: 'Benigno'})

print("\n📋 Primeras características del dataset:")
print(df[['mean radius', 'mean texture', 'mean area', 'diagnosis_name']].head())

print("\n📈 Estadísticas básicas de las primeras 5 características:")
print(df.iloc[:, :5].describe().round(2))

# ============================================================
# 2. VISUALIZACIÓN DE DATOS
# ============================================================
print("\n2. VISUALIZACIÓN DE DATOS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribución de clases
class_counts = df['diagnosis_name'].value_counts()
colors_pie = ['#ff6b6b', '#51cf66']
axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, explode=(0.05, 0))
axes[0, 0].set_title('Distribución de Diagnósticos', fontsize=12, fontweight='bold')

# 2. Comparación de características principales
features_to_plot = ['mean radius', 'mean texture', 'mean area', 'mean smoothness']
df_melted = df[features_to_plot + ['diagnosis_name']].melt(
    id_vars='diagnosis_name', var_name='Característica', value_name='Valor'
)
sns.violinplot(data=df_melted, x='Característica', y='Valor', hue='diagnosis_name',
               split=True, ax=axes[0, 1], palette=['#ff6b6b', '#51cf66'])
axes[0, 1].set_title('Distribución de Características por Diagnóstico', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=15)
axes[0, 1].legend(title='Diagnóstico')

# 3. Correlación entre características principales
correlation_features = ['mean radius', 'mean texture', 'mean perimeter', 
                       'mean area', 'mean smoothness', 'diagnosis']
corr_matrix = df[correlation_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[1, 0], cbar_kws={'label': 'Correlación'})
axes[1, 0].set_title('Matriz de Correlación', fontsize=12, fontweight='bold')

# 4. Scatter plot de dos características principales
axes[1, 1].scatter(df[df['diagnosis']==1]['mean radius'], 
                   df[df['diagnosis']==1]['mean area'],
                   alpha=0.6, s=50, c='#51cf66', edgecolors='black', 
                   linewidth=0.5, label='Benigno')
axes[1, 1].scatter(df[df['diagnosis']==0]['mean radius'], 
                   df[df['diagnosis']==0]['mean area'],
                   alpha=0.6, s=50, c='#ff6b6b', edgecolors='black', 
                   linewidth=0.5, label='Maligno')
axes[1, 1].set_xlabel('Radio Medio', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Área Media', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Separación de Clases: Radio vs Área', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/06_clasificacion_comp_1_exploracion_datos.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 3. PREPARACIÓN DE DATOS
# ============================================================
print("\n3. PREPARACIÓN DE DATOS")
print("-" * 80)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️ División de datos:")
print(f"  - Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"  - Prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Normalización de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n🔄 Normalización completada con StandardScaler")
print("   (Media = 0, Desviación Estándar = 1)")

# ============================================================
# 4. DEFINICIÓN DE MODELOS
# ============================================================
print("\n4. DEFINICIÓN DE MODELOS")
print("-" * 80)

# Diccionario de modelos a comparar
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Gaussian Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

print(f"\n🤖 Se compararán {len(models)} modelos de clasificación:")
for i, name in enumerate(models.keys(), 1):
    print(f"  {i}. {name}")

# ============================================================
# 5. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ============================================================
print("\n5. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("-" * 80)

results = []

for name, model in models.items():
    print(f"\n⚙️ Entrenando {name}...", end=" ")
    
    # Entrenar modelo
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Guardar resultados
    results.append({
        'Modelo': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'CV Accuracy': cv_mean,
        'CV Std': cv_std
    })
    
    print(f"✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

# Crear DataFrame con resultados
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('F1-Score', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("RESULTADOS DE LA COMPETENCIA")
print("="*80)
print(df_results.to_string(index=False))
print("="*80 + "\n")

# ============================================================
# 6. VISUALIZACIÓN DE COMPARACIÓN DE MODELOS
# ============================================================
print("\n6. VISUALIZACIÓN DE COMPARACIÓN DE MODELOS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Comparación de Accuracy
colors = sns.color_palette("viridis", len(df_results))
axes[0, 0].barh(df_results['Modelo'], df_results['Accuracy'], color=colors, edgecolor='black')
axes[0, 0].set_xlabel('Accuracy', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Comparación de Accuracy por Modelo', fontsize=12, fontweight='bold')
axes[0, 0].set_xlim(0.85, 1.0)
axes[0, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(df_results['Accuracy']):
    axes[0, 0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')

# 2. Comparación de F1-Score
axes[0, 1].barh(df_results['Modelo'], df_results['F1-Score'], color=colors, edgecolor='black')
axes[0, 1].set_xlabel('F1-Score', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Comparación de F1-Score por Modelo', fontsize=12, fontweight='bold')
axes[0, 1].set_xlim(0.85, 1.0)
axes[0, 1].grid(axis='x', alpha=0.3)
for i, v in enumerate(df_results['F1-Score']):
    axes[0, 1].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')

# 3. Precision vs Recall
axes[1, 0].scatter(df_results['Precision'], df_results['Recall'], 
                  s=200, alpha=0.6, c=range(len(df_results)), 
                  cmap='viridis', edgecolors='black', linewidth=2)
for i, model in enumerate(df_results['Modelo']):
    axes[1, 0].annotate(model, 
                       (df_results.iloc[i]['Precision'], df_results.iloc[i]['Recall']),
                       fontsize=8, ha='right', va='bottom')
axes[1, 0].set_xlabel('Precision', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Recall', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0.85, 1.0)
axes[1, 0].set_ylim(0.85, 1.0)

# 4. Comparación de múltiples métricas (Radar)
# Seleccionar top 5 modelos
top_5 = df_results.head(5)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

axes[1, 1] = plt.subplot(2, 2, 4, projection='polar')
for i, row in top_5.iterrows():
    values = [row[m] for m in metrics]
    values += values[:1]
    axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=row['Modelo'])
    axes[1, 1].fill(angles, values, alpha=0.15)

axes[1, 1].set_xticks(angles[:-1])
axes[1, 1].set_xticklabels(metrics, fontsize=9)
axes[1, 1].set_ylim(0.85, 1.0)
axes[1, 1].set_title('Comparación Multimétrica (Top 5)', fontsize=12, fontweight='bold', pad=20)
axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('imgs/06_clasificacion_comp_2_comparacion_metricas.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 7. ANÁLISIS DEL MEJOR MODELO
# ============================================================
print("\n7. ANÁLISIS DEL MEJOR MODELO")
print("-" * 80)

# Seleccionar mejor modelo
best_model_name = df_results.iloc[0]['Modelo']
best_model = models[best_model_name]

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"\nMétricas en conjunto de prueba:")
print(f"  - Accuracy:  {df_results.iloc[0]['Accuracy']:.4f}")
print(f"  - Precision: {df_results.iloc[0]['Precision']:.4f}")
print(f"  - Recall:    {df_results.iloc[0]['Recall']:.4f}")
print(f"  - F1-Score:  {df_results.iloc[0]['F1-Score']:.4f}")
print(f"  - ROC-AUC:   {df_results.iloc[0]['ROC-AUC']:.4f}")
print(f"\nValidación Cruzada (5-fold):")
print(f"  - CV Accuracy: {df_results.iloc[0]['CV Accuracy']:.4f} ± {df_results.iloc[0]['CV Std']:.4f}")

# Predicciones del mejor modelo
y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Reporte de clasificación
print("\n📊 REPORTE DE CLASIFICACIÓN DETALLADO:")
print(classification_report(y_test, y_pred_best, target_names=data.target_names))

# ============================================================
# 8. MATRIZ DE CONFUSIÓN Y CURVA ROC
# ============================================================
print("\n8. MATRIZ DE CONFUSIÓN Y CURVA ROC")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=data.target_names, yticklabels=data.target_names,
            cbar_kws={'label': 'Número de casos'})
axes[0].set_xlabel('Predicción', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Valor Real', fontsize=11, fontweight='bold')
axes[0].set_title(f'Matriz de Confusión - {best_model_name}', fontsize=12, fontweight='bold')

# Añadir porcentajes
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm.sum() * 100
        axes[0].text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=9, color='red')

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best)
roc_auc = df_results.iloc[0]['ROC-AUC']

axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'{best_model_name} (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11, fontweight='bold')
axes[1].set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/06_clasificacion_comp_3_mejor_modelo_confusion_roc.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

print(f"\n✓ Matriz de confusión generada")
print(f"  - Verdaderos Negativos: {cm[0,0]}")
print(f"  - Falsos Positivos: {cm[0,1]}")
print(f"  - Falsos Negativos: {cm[1,0]}")
print(f"  - Verdaderos Positivos: {cm[1,1]}")

# ============================================================
# 9. CURVAS ROC DE TODOS LOS MODELOS
# ============================================================
print("\n9. CURVAS ROC DE TODOS LOS MODELOS")
print("-" * 80)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador Aleatorio (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12, fontweight='bold')
plt.title('Curvas ROC - Comparación de Todos los Modelos', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('imgs/06_clasificacion_comp_4_curvas_roc_todos.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 10. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================
print("\n10. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
print("-" * 80)

# Verificar si el mejor modelo tiene feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n🔍 Top 10 características más importantes para {best_model_name}:")
    for i in range(min(10, len(indices))):
        print(f"  {i+1}. {data.feature_names[indices[i]]:<30} - {importances[indices[i]]:.4f}")
    
    # Visualización
    plt.figure(figsize=(10, 6))
    top_n = 15
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], 
            color='steelblue', edgecolor='black')
    plt.yticks(range(top_n), [data.feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Importancia', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Características más Importantes - {best_model_name}', 
             fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('imgs/06_clasificacion_comp_5_feature_importance.jpg', format='jpg', bbox_inches='tight', dpi=100)
    plt.show()
    
elif hasattr(best_model, 'coef_'):
    coefficients = np.abs(best_model.coef_[0])
    indices = np.argsort(coefficients)[::-1]
    
    print(f"\n🔍 Top 10 características con mayor peso para {best_model_name}:")
    for i in range(min(10, len(indices))):
        print(f"  {i+1}. {data.feature_names[indices[i]]:<30} - {coefficients[indices[i]]:.4f}")
    
    # Visualización
    plt.figure(figsize=(10, 6))
    top_n = 15
    plt.barh(range(top_n), coefficients[indices[:top_n]][::-1], 
            color='coral', edgecolor='black')
    plt.yticks(range(top_n), [data.feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Peso Absoluto del Coeficiente', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Características con Mayor Peso - {best_model_name}', 
             fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('imgs/06_clasificacion_comp_5_feature_importance.jpg', format='jpg', bbox_inches='tight', dpi=100)
    plt.show()
else:
    print(f"\n⚠️ {best_model_name} no proporciona importancia de características directamente")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"""
Dataset: Breast Cancer Wisconsin
Total de muestras: {X.shape[0]}
Características: {X.shape[1]}
Clases: Maligno (0) / Benigno (1)

PROCESO REALIZADO:
1. ✓ Exploración de datos con estadísticas y visualizaciones
2. ✓ Análisis de distribución de clases
3. ✓ Normalización de datos con StandardScaler
4. ✓ Comparación de {len(models)} modelos de clasificación
5. ✓ Evaluación con validación cruzada (5 folds)
6. ✓ Selección del mejor modelo: {best_model_name}
7. ✓ Análisis de matriz de confusión y curva ROC
8. ✓ Análisis de importancia de características

COMPARACIÓN DE MODELOS:
Los {len(models)} modelos fueron evaluados usando múltiples métricas.
Top 3 modelos por F1-Score:
""")

for i in range(min(3, len(df_results))):
    modelo = df_results.iloc[i]
    print(f"  {i+1}. {modelo['Modelo']:<25} - F1: {modelo['F1-Score']:.4f}, Accuracy: {modelo['Accuracy']:.4f}")

print(f"""
MÉTRICAS DEL MEJOR MODELO ({best_model_name}):
  - Accuracy:  {df_results.iloc[0]['Accuracy']:.4f} - Proporción de predicciones correctas
  - Precision: {df_results.iloc[0]['Precision']:.4f} - De los predichos positivos, cuántos son correctos
  - Recall:    {df_results.iloc[0]['Recall']:.4f} - De los positivos reales, cuántos fueron detectados
  - F1-Score:  {df_results.iloc[0]['F1-Score']:.4f} - Media armónica de precision y recall
  - ROC-AUC:   {df_results.iloc[0]['ROC-AUC']:.4f} - Capacidad discriminativa del modelo

INTERPRETACIÓN:
✓ El modelo {best_model_name} demostró el mejor rendimiento general
✓ Accuracy de {df_results.iloc[0]['Accuracy']:.4f} indica {df_results.iloc[0]['Accuracy']*100:.1f}% de predicciones correctas
✓ F1-Score de {df_results.iloc[0]['F1-Score']:.4f} muestra excelente balance precision-recall
✓ ROC-AUC de {df_results.iloc[0]['ROC-AUC']:.4f} indica excelente capacidad discriminativa

VALIDACIÓN CRUZADA:
El modelo fue validado con 5-fold cross-validation para asegurar
que no hay sobreajuste y que generaliza bien a datos nuevos.
CV Accuracy: {df_results.iloc[0]['CV Accuracy']:.4f} ± {df_results.iloc[0]['CV Std']:.4f}
""")
print("="*80 + "\n")

print("💡 RECOMENDACIONES PARA MEJORAR:")
print("  1. Feature Engineering: Crear interacciones entre características")
print("  2. Tuning de Hiperparámetros: Usar GridSearchCV o RandomizedSearchCV")
print("  3. Ensemble Methods: Combinar modelos con Voting o Stacking")
print("  4. Análisis de Errores: Estudiar casos mal clasificados")
print("  5. Balanceo de Clases: Si hay desbalance, usar SMOTE o class_weight")
print("\n" + "="*80 + "\n")

print("📊 MÉTRICAS DE CLASIFICACIÓN EXPLICADAS:")
print("  • Accuracy:  % total de aciertos (puede ser engañosa con clases desbalanceadas)")
print("  • Precision: De los que predije como positivos, cuántos realmente lo son")
print("  • Recall:    De todos los positivos reales, cuántos logré detectar")
print("  • F1-Score:  Balance entre precision y recall (ideal cuando importan ambos)")
print("  • ROC-AUC:   Qué tan bien separa el modelo las clases (1.0 = perfecto)")
print("\n" + "="*80 + "\n")

print("🎯 CUÁNDO USAR CADA MODELO:")
print("  • Logistic Regression: Rápido, interpretable, bueno para baseline")
print("  • Random Forest: Robusto, maneja no-linealidad, menos overfitting")
print("  • SVM: Excelente para datasets pequeños/medianos con buena separación")
print("  • Gradient Boosting: Alto rendimiento, pero más lento de entrenar")
print("  • KNN: Simple, no requiere entrenamiento, pero lento en predicción")
print("  • Naive Bayes: Muy rápido, asume independencia entre características")
print("\n" + "="*80 + "\n")
