import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Importar múltiples modelos de regresión
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# Configurar el estilo de las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n" + "="*80)
print("COMPETENCIA DE MODELOS DE REGRESIÓN")
print("="*80)
print("\nEste script muestra cómo comparar múltiples modelos de Machine Learning")
print("para resolver un problema de regresión y seleccionar el mejor.")
print("\nModelos a comparar:")
print("  1. Regresión Lineal")
print("  2. Ridge Regression")
print("  3. Lasso Regression")
print("  4. ElasticNet")
print("  5. Decision Tree")
print("  6. Random Forest")
print("  7. Gradient Boosting")
print("  8. AdaBoost")
print("  9. K-Nearest Neighbors")
print("  10. Support Vector Machine")
print("  11. XGBoost")

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================
print("\n1. CARGA DEL DATASET: CALIFORNIA HOUSING")
print("-" * 80)

# Cargar el dataset de precios de viviendas en California
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

print("\nDataset: California Housing Prices")
print("Este dataset contiene información sobre viviendas en California (1990):")
print("  - MedInc: Ingreso medio del hogar")
print("  - HouseAge: Edad media de la vivienda")
print("  - AveRooms: Promedio de habitaciones por hogar")
print("  - AveBedrms: Promedio de dormitorios por hogar")
print("  - Population: Población del bloque")
print("  - AveOccup: Promedio de ocupantes por hogar")
print("  - Latitude: Latitud del bloque")
print("  - Longitude: Longitud del bloque")
print("  - MedHouseVal: Valor medio de la vivienda (en $100,000) - OBJETIVO")

print(f"\nForma del dataset: {df.shape}")
print(f"Número de muestras: {df.shape[0]}")
print(f"Número de características: {df.shape[1] - 1}")

# ============================================================
# 2. EXPLORACIÓN INICIAL DE DATOS
# ============================================================
print("\n2. EXPLORACIÓN INICIAL")
print("-" * 80)

print("\nPrimeras 10 filas del dataset:")
print(df.head(10))

print("\nÚltimas 5 filas del dataset:")
print(df.tail(5))

print("\nInformación del dataset:")
print(df.info())

# ============================================================
# 3. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("\n3. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

print("\nEstadísticas generales:")
print(df.describe())

print("\nVerificación de valores faltantes:")
print(df.isnull().sum())

print("\nDistribución de la variable objetivo (MedHouseVal):")
print(df['MedHouseVal'].describe())

# ============================================================
# 4. VISUALIZACIÓN EXPLORATORIA - PARTE 1
# ============================================================
print("\n4. VISUALIZACIONES EXPLORATORIAS - PARTE 1")
print("-" * 80)

# Distribución de la variable objetivo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['MedHouseVal'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Valor Medio de Vivienda ($100k)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
axes[0].set_title('Distribución del Precio de Viviendas', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].boxplot(df['MedHouseVal'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Valor Medio de Vivienda ($100k)', fontsize=12, fontweight='bold')
axes[1].set_title('Box Plot del Precio de Viviendas', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================
# 5. VISUALIZACIÓN EXPLORATORIA - PARTE 2
# ============================================================
print("\n5. VISUALIZACIONES EXPLORATORIAS - PARTE 2")
print("-" * 80)

# Distribuciones de las características principales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

caracteristicas_principales = ['MedInc', 'HouseAge', 'AveRooms', 'Population']
colores = ['steelblue', 'forestgreen', 'coral', 'purple']

for idx, (feature, color) in enumerate(zip(caracteristicas_principales, colores)):
    axes[idx].hist(df[feature], bins=40, alpha=0.7, color=color, edgecolor='black')
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Distribución de {feature}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================
# 6. RELACIÓN ENTRE VARIABLES Y OBJETIVO
# ============================================================
print("\n6. RELACIÓN ENTRE VARIABLES Y OBJETIVO")
print("-" * 80)

# Scatter plots de características vs objetivo
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (feature, color) in enumerate(zip(caracteristicas_principales, colores)):
    # Usar sample para mejor visualización (dataset muy grande)
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)
    axes[idx].scatter(sample_df[feature], sample_df['MedHouseVal'], 
                     alpha=0.5, s=20, color=color, edgecolors='black', linewidth=0.3)
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('MedHouseVal', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{feature} vs Precio', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 7. MATRIZ DE CORRELACIÓN
# ============================================================
print("\n7. MATRIZ DE CORRELACIÓN")
print("-" * 80)

# Calcular correlación
correlation_matrix = df.corr()
print("\nMatriz de correlación:")
print(correlation_matrix.round(3))

# Correlación con la variable objetivo
print("\nCorrelación con la variable objetivo (MedHouseVal):")
correlaciones = correlation_matrix['MedHouseVal'].sort_values(ascending=False)
print(correlaciones)

# Visualizar matriz de correlación
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlación - California Housing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# 8. MAPA GEOGRÁFICO DE PRECIOS
# ============================================================
print("\n8. VISUALIZACIÓN GEOGRÁFICA")
print("-" * 80)

# Visualizar precios en el mapa de California
fig, ax = plt.subplots(figsize=(12, 8))

# Usar sample para mejor rendimiento
sample_df = df.sample(n=min(5000, len(df)), random_state=42)

scatter = ax.scatter(sample_df['Longitude'], sample_df['Latitude'], 
                    c=sample_df['MedHouseVal'], s=30, alpha=0.6, 
                    cmap='YlOrRd', edgecolors='black', linewidth=0.3)

ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')
ax.set_title('Distribución Geográfica de Precios de Viviendas en California', 
             fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Valor Medio de Vivienda ($100k)', ax=ax)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 9. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================
print("\n9. PREPARACIÓN DE DATOS")
print("-" * 80)

# Tomar una muestra del dataset para agilizar el entrenamiento
sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

print(f"\nUsando una muestra de {sample_size} registros para el análisis")
print(f"Esto permite un entrenamiento más rápido sin perder generalidad")

# Separar características (X) y variable objetivo (y)
X = df_sample.drop('MedHouseVal', axis=1)
y = df_sample['MedHouseVal']

# División train/test (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Normalización de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Datos normalizados con StandardScaler")

# ============================================================
# 10. DEFINICIÓN DE MODELOS A COMPARAR
# ============================================================
print("\n10. DEFINICIÓN DE MODELOS")
print("-" * 80)

# Crear diccionario con todos los modelos a comparar
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

print(f"\nTotal de modelos a comparar: {len(models)}")
for i, name in enumerate(models.keys(), 1):
    print(f"  {i}. {name}")

# ============================================================
# 11. COMPARACIÓN DE MODELOS CON VALIDACIÓN CRUZADA
# ============================================================
print("\n11. COMPARACIÓN DE MODELOS CON VALIDACIÓN CRUZADA")
print("-" * 80)
print("\nEntrenando y evaluando modelos...")
print("Esto puede tomar unos minutos...\n")

# Almacenar resultados
results = []

for name, model in models.items():
    print(f"Evaluando {name}...", end=' ')
    
    # Validación cruzada con 5 folds
    # Usamos R² como métrica principal
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                 cv=5, scoring='r2', n_jobs=-1)
    
    # Entrenar el modelo con todos los datos de entrenamiento
    model.fit(X_train_scaled, y_train)
    
    # Predicciones en conjunto de prueba
    y_pred = model.predict(X_test_scaled)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    # Guardar resultados
    results.append({
        'Modelo': name,
        'R² (CV Mean)': cv_scores.mean(),
        'R² (CV Std)': cv_scores.std(),
        'R² (Test)': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    })
    
    print(f"✓ (R² = {r2:.4f}, RMSE = {rmse:.4f})")

print("\n✓ Evaluación completada")

# ============================================================
# 12. TABLA DE RESULTADOS
# ============================================================
print("\n12. TABLA DE RESULTADOS COMPARATIVOS")
print("-" * 80)

# Crear DataFrame con resultados
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('R² (Test)', ascending=False).reset_index(drop=True)

print("\nResultados ordenados por R² en conjunto de prueba:")
print(df_results.to_string(index=False))

# Identificar el mejor modelo
best_model_name = df_results.iloc[0]['Modelo']
best_r2 = df_results.iloc[0]['R² (Test)']
best_rmse = df_results.iloc[0]['RMSE']

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"   R² = {best_r2:.4f}")
print(f"   RMSE = {best_rmse:.4f}")

# ============================================================
# 13. VISUALIZACIÓN DE COMPARACIÓN - PARTE 1
# ============================================================
print("\n13. VISUALIZACIÓN DE COMPARACIÓN - PARTE 1")
print("-" * 80)

# Gráfica de barras con R² y RMSE
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# R² Score
colors_r2 = ['gold' if x == best_model_name else 'steelblue' for x in df_results['Modelo']]
axes[0].barh(df_results['Modelo'], df_results['R² (Test)'], 
            color=colors_r2, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('Comparación de R² Score (mayor es mejor)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Umbral excelente (0.8)')
axes[0].legend()
for i, v in enumerate(df_results['R² (Test)']):
    axes[0].text(v - 0.02, i, f'{v:.4f}', ha='right', va='center', 
                fontweight='bold', color='white' if v > 0.1 else 'black')

# RMSE
colors_rmse = ['gold' if x == best_model_name else 'coral' for x in df_results['Modelo']]
axes[1].barh(df_results['Modelo'], df_results['RMSE'], 
            color=colors_rmse, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('RMSE', fontsize=12, fontweight='bold')
axes[1].set_title('Comparación de RMSE (menor es mejor)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df_results['RMSE']):
    axes[1].text(v + 0.01, i, f'{v:.3f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# 14. VISUALIZACIÓN DE COMPARACIÓN - PARTE 2
# ============================================================
print("\n14. VISUALIZACIÓN DE COMPARACIÓN - PARTE 2")
print("-" * 80)

# Gráfica de dispersión: R² vs RMSE
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(df_results['R² (Test)'], df_results['RMSE'], 
                    s=200, alpha=0.6, c=range(len(df_results)), 
                    cmap='viridis', edgecolors='black', linewidth=2)

# Etiquetar cada punto con el nombre del modelo
for idx, row in df_results.iterrows():
    ax.annotate(row['Modelo'], 
               (row['R² (Test)'], row['RMSE']),
               xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold' if row['Modelo'] == best_model_name else 'normal')

ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('R² vs RMSE: Ubicación ideal es arriba-derecha/abajo-derecha', 
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Marcar región ideal
ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='R² > 0.8 (excelente)')
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 15. ANÁLISIS DEL MEJOR MODELO
# ============================================================
print("\n15. ANÁLISIS DETALLADO DEL MEJOR MODELO")
print("-" * 80)

# Entrenar nuevamente el mejor modelo
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

print(f"\nModelo: {best_model_name}")
print(f"Parámetros: {best_model.get_params()}")

# Métricas detalladas
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)
mape_best = mean_absolute_percentage_error(y_test, y_pred_best) * 100

print("\nMétricas en conjunto de prueba:")
print(f"  MAE (Error Absoluto Medio):           {mae_best:.4f} ($100k)")
print(f"  MSE (Error Cuadrático Medio):         {mse_best:.4f}")
print(f"  RMSE (Raíz Error Cuadrático Medio):   {rmse_best:.4f} ($100k)")
print(f"  R² (Coeficiente de Determinación):    {r2_best:.4f}")
print(f"  MAPE (Error Porcentual Abs. Medio):   {mape_best:.2f}%")

# Análisis de residuos
residuos = y_test - y_pred_best

print(f"\nAnálisis de residuos:")
print(f"  Media de residuos:        {residuos.mean():.4f}")
print(f"  Std de residuos:          {residuos.std():.4f}")
print(f"  Min residuo:              {residuos.min():.4f}")
print(f"  Max residuo:              {residuos.max():.4f}")

# ============================================================
# 16. GRÁFICAS DE EVALUACIÓN DEL MEJOR MODELO
# ============================================================
print("\n16. GRÁFICAS DE EVALUACIÓN DEL MEJOR MODELO")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Predicciones vs Valores Reales
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.5, s=30, 
                  color='steelblue', edgecolors='black', linewidth=0.3)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
axes[0, 0].set_xlabel('Valor Real', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Valor Predicho', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribución de Residuos
axes[0, 1].hist(residuos, bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
axes[0, 1].set_xlabel('Residuo', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Residuos vs Predicciones
axes[1, 0].scatter(y_pred_best, residuos, alpha=0.5, s=30, 
                  color='purple', edgecolors='black', linewidth=0.3)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Valor Predicho', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Residuo', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Residuos vs Predicciones', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Q-Q Plot (aproximado)
from scipy import stats
stats.probplot(residuos, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normalidad de Residuos)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 17. IMPORTANCIA DE CARACTERÍSTICAS (si aplica)
# ============================================================
print("\n17. IMPORTANCIA DE CARACTERÍSTICAS")
print("-" * 80)

# Obtener importancia de características si el modelo lo soporta
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns
    
    # Crear DataFrame de importancias
    df_importance = pd.DataFrame({
        'Característica': feature_names,
        'Importancia': importances
    }).sort_values('Importancia', ascending=False)
    
    print("\nImportancia de características:")
    print(df_importance.to_string(index=False))
    
    # Gráfica de importancia
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_importance['Característica'], df_importance['Importancia'], 
           color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_title(f'Importancia de Características - {best_model_name}', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
elif hasattr(best_model, 'coef_'):
    coefficients = best_model.coef_
    feature_names = X.columns
    
    # Crear DataFrame de coeficientes
    df_coef = pd.DataFrame({
        'Característica': feature_names,
        'Coeficiente': coefficients,
        'Abs_Coeficiente': np.abs(coefficients)
    }).sort_values('Abs_Coeficiente', ascending=False)
    
    print("\nCoeficientes del modelo:")
    print(df_coef[['Característica', 'Coeficiente']].to_string(index=False))
    
    # Gráfica de coeficientes
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_coef = ['red' if c < 0 else 'green' for c in df_coef['Coeficiente']]
    ax.barh(df_coef['Característica'], df_coef['Coeficiente'], 
           color=colors_coef, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Coeficiente', fontsize=12, fontweight='bold')
    ax.set_title(f'Coeficientes del Modelo - {best_model_name}', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
else:
    print(f"\nEl modelo {best_model_name} no proporciona importancia de características directamente")

# ============================================================
# 18. GUARDAR EL MEJOR MODELO
# ============================================================
print("\n18. GUARDAR EL MEJOR MODELO")
print("-" * 80)

import joblib

# Guardar el modelo y el scaler
joblib.dump(best_model, 'best_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✓ Modelo guardado como 'best_regression_model.pkl'")
print("✓ Scaler guardado como 'scaler.pkl'")
print("\nPara cargar el modelo:")
print("  modelo = joblib.load('best_regression_model.pkl')")
print("  scaler = joblib.load('scaler.pkl')")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"""
Dataset: California Housing ({sample_size} muestras de {df.shape[0]} totales)
Características: {df.shape[1] - 1}
Variable Objetivo: Valor Medio de Vivienda (MedHouseVal)

PROCESO REALIZADO:
1. ✓ Exploración de datos con estadísticas y visualizaciones
2. ✓ Análisis de correlaciones entre variables
3. ✓ Visualización geográfica de precios
4. ✓ Normalización de datos con StandardScaler
5. ✓ Comparación de {len(models)} modelos de regresión
6. ✓ Evaluación con validación cruzada (5 folds)
7. ✓ Selección del mejor modelo: {best_model_name}
8. ✓ Análisis exhaustivo de residuos y predicciones

COMPARACIÓN DE MODELOS:
Los {len(models)} modelos fueron evaluados usando múltiples métricas.
Top 3 modelos por R²:
""")

for i in range(min(3, len(df_results))):
    modelo = df_results.iloc[i]
    print(f"  {i+1}. {modelo['Modelo']:<20} - R²: {modelo['R² (Test)']:.4f}, RMSE: {modelo['RMSE']:.4f}")

print(f"""
MÉTRICAS DEL MEJOR MODELO ({best_model_name}):
  - MAE:  {mae_best:.4f} ($100k) - Error promedio absoluto
  - RMSE: {rmse_best:.4f} ($100k) - Raíz del error cuadrático medio
  - R²:   {r2_best:.4f} - Varianza explicada
  - MAPE: {mape_best:.2f}% - Error porcentual medio

INTERPRETACIÓN:
✓ El modelo {best_model_name} demostró el mejor rendimiento general
✓ R² de {r2_best:.4f} indica que explica el {r2_best*100:.1f}% de la variabilidad
✓ Error promedio (RMSE) de ${rmse_best*100:.2f}k en predicciones
✓ Precio medio real: ${df['MedHouseVal'].mean()*100:.2f}k

VALIDACIÓN CRUZADA:
El modelo fue validado con 5-fold cross-validation para asegurar
que no hay sobreajuste y que generaliza bien a datos nuevos.
""")
print("="*80 + "\n")

print("💡 RECOMENDACIONES PARA MEJORAR:")
print("  1. Feature Engineering: Crear nuevas características (ej: ratios, interacciones)")
print("  2. Tuning de Hiperparámetros: Usar GridSearchCV o RandomizedSearchCV")
print("  3. Ensemble Methods: Combinar múltiples modelos (Voting, Stacking)")
print("  4. Análisis de Outliers: Investigar y tratar valores atípicos")
print("  5. Más Datos: Usar el dataset completo para mejor generalización")
print("\n" + "="*80 + "\n")

print("📊 CÓMO SE MIDIÓ LA COMPETENCIA:")
print("  ✓ Validación Cruzada: Cada modelo fue evaluado en 5 particiones diferentes")
print("  ✓ Múltiples Métricas: R², RMSE, MAE, MAPE para visión completa")
print("  ✓ Conjunto de Prueba: Datos nunca vistos durante el entrenamiento")
print("  ✓ Comparación Visual: Gráficas para identificar fortalezas/debilidades")
print("\n" + "="*80 + "\n")
