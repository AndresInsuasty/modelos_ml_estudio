import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configurar el estilo de las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n" + "="*80)
print("ANÁLISIS DE CLUSTERIZACIÓN CON DATASET REAL: PINGÜINOS DE PALMER")
print("="*80)

# ============================================================
# 1. CARGA DEL DATASET DE PINGÜINOS
# ============================================================
print("\n1. CARGA DEL DATASET DE PINGÜINOS")
print("-" * 80)

# Cargar el dataset de pingüinos de Palmer
df_original = sns.load_dataset('penguins')

print("\nDataset de Pingüinos de Palmer cargado exitosamente")
print("Este dataset contiene mediciones de 3 especies de pingüinos:")
print("  - Adelie")
print("  - Chinstrap")
print("  - Gentoo")
print("\nCaracterísticas medidas:")
print("  - Largo del pico (bill_length_mm)")
print("  - Profundidad del pico (bill_depth_mm)")
print("  - Largo de la aleta (flipper_length_mm)")
print("  - Masa corporal (body_mass_g)")
print("  - Isla donde habitan")
print("  - Sexo")

# ============================================================
# 2. EXPLORACIÓN INICIAL DE DATOS
# ============================================================
print("\n2. EXPLORACIÓN INICIAL")
print("-" * 80)

print("\nPrimeras 10 filas del dataset:")
print(df_original.head(10))

print("\nInformación del dataset:")
print(df_original.info())

print(f"\nForma del dataset: {df_original.shape}")
print(f"Número de filas: {df_original.shape[0]}")
print(f"Número de columnas: {df_original.shape[1]}")

# ============================================================
# 3. LIMPIEZA DE DATOS
# ============================================================
print("\n3. LIMPIEZA DE DATOS")
print("-" * 80)

print("\nValores faltantes por columna:")
print(df_original.isnull().sum())

# Eliminar filas con valores faltantes
df = df_original.dropna()

print("\nDataset después de eliminar valores faltantes:")
print(f"Forma: {df.shape}")
print(f"Se eliminaron {df_original.shape[0] - df.shape[0]} filas")

# ============================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("\n4. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

print("\nEstadísticas generales:")
print(df.describe())

print("\nDistribución de especies:")
print(df['species'].value_counts())

print("\nDistribución por isla:")
print(df['island'].value_counts())

print("\nDistribución por sexo:")
print(df['sex'].value_counts())

# ============================================================
# 5. VISUALIZACIÓN EXPLORATORIA - PARTE 1
# ============================================================
print("\n5. VISUALIZACIONES EXPLORATORIAS - PARTE 1")
print("-" * 80)

# Distribuciones de características numéricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

caracteristicas = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
colores = {'Adelie': 'orange', 'Chinstrap': 'purple', 'Gentoo': 'teal'}

for idx, feature in enumerate(caracteristicas):
    for especie in df['species'].unique():
        datos = df[df['species'] == especie][feature]
        axes[idx].hist(datos, alpha=0.6, label=especie, bins=15, color=colores[especie])
    
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Distribución de {feature}', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('imgs/04_clusterizacion_1_histogramas_features.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 6. VISUALIZACIÓN EXPLORATORIA - PARTE 2
# ============================================================
print("\n6. VISUALIZACIONES EXPLORATORIAS - PARTE 2")
print("-" * 80)

# Scatter plots de pares de características
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Largo del pico vs Profundidad del pico
for especie in df['species'].unique():
    datos = df[df['species'] == especie]
    axes[0].scatter(datos['bill_length_mm'], datos['bill_depth_mm'], 
                   label=especie, s=100, alpha=0.7, color=colores[especie],
                   edgecolors='black', linewidth=0.5)

axes[0].set_xlabel('Largo del Pico (mm)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Profundidad del Pico (mm)', fontsize=12, fontweight='bold')
axes[0].set_title('Características del Pico', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Largo de aleta vs Masa corporal
for especie in df['species'].unique():
    datos = df[df['species'] == especie]
    axes[1].scatter(datos['flipper_length_mm'], datos['body_mass_g'], 
                   label=especie, s=100, alpha=0.7, color=colores[especie],
                   edgecolors='black', linewidth=0.5)

axes[1].set_xlabel('Largo de Aleta (mm)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Masa Corporal (g)', fontsize=12, fontweight='bold')
axes[1].set_title('Tamaño Físico', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/04_clusterizacion_2_scatter_caracteristicas.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 7. MATRIZ DE CORRELACIÓN
# ============================================================
print("\n7. MATRIZ DE CORRELACIÓN")
print("-" * 80)

# Calcular correlación
correlation_matrix = df[caracteristicas].corr()
print("\nMatriz de correlación:")
print(correlation_matrix.round(3))

# Visualizar matriz de correlación
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlación - Pingüinos de Palmer', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('imgs/04_clusterizacion_3_matriz_correlacion.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 8. PREPARACIÓN DE DATOS PARA CLUSTERING
# ============================================================
print("\n8. PREPARACIÓN DE DATOS")
print("-" * 80)

# Seleccionar solo las características numéricas
X = df[caracteristicas].values

print("\nCaracterísticas seleccionadas para clustering:")
print(f"  {caracteristicas}")
print(f"\nForma de los datos: {X.shape}")

# Normalización de datos (muy importante para clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDatos escalados - Media (debería ser ~0):")
print(X_scaled.mean(axis=0).round(4))
print("\nDatos escalados - Desv. Est. (debería ser ~1):")
print(X_scaled.std(axis=0).round(4))

# Guardar las etiquetas reales para comparación
etiquetas_reales = df['species'].values

# ============================================================
# 9. MÉTODO DEL CODO (ELBOW METHOD)
# ============================================================
print("\n9. MÉTODO DEL CODO - ENCONTRAR K ÓPTIMO")
print("-" * 80)

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"K={k}: Inercia={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")

# Visualizar método del codo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=10)
axes[0].set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Inercia (WCSS)', fontsize=12, fontweight='bold')
axes[0].set_title('Método del Codo', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=10)
axes[1].set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].set_title('Silhouette Score por K', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/04_clusterizacion_4_metodo_codo.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 10. K-MEANS CLUSTERING (K=3)
# ============================================================
print("\n10. K-MEANS CLUSTERING (K=3)")
print("-" * 80)

# Aplicar K-Means con k=3 (sabemos que hay 3 especies)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_scaled)

print("\nModelo K-Means entrenado con k=3")
print(f"Número de iteraciones: {kmeans.n_iter_}")
print(f"Inercia final: {kmeans.inertia_:.2f}")

print("\nDistribución de muestras por cluster:")
for i in range(3):
    count = np.sum(clusters_kmeans == i)
    print(f"  Cluster {i}: {count} pingüinos")

# Métricas de evaluación
silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans)
calinski_harabasz_kmeans = calinski_harabasz_score(X_scaled, clusters_kmeans)

print("\nMétricas de evaluación:")
print(f"  Silhouette Score: {silhouette_kmeans:.4f} (cercano a 1 es mejor)")
print(f"  Davies-Bouldin Index: {davies_bouldin_kmeans:.4f} (cercano a 0 es mejor)")
print(f"  Calinski-Harabasz Index: {calinski_harabasz_kmeans:.2f} (más alto es mejor)")

# ============================================================
# 11. CLUSTERING JERÁRQUICO
# ============================================================
print("\n11. CLUSTERING JERÁRQUICO (AGGLOMERATIVE)")
print("-" * 80)

# Aplicar clustering jerárquico
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters_hierarchical = hierarchical.fit_predict(X_scaled)

print("\nModelo de Clustering Jerárquico entrenado")
print("\nDistribución de muestras por cluster:")
for i in range(3):
    count = np.sum(clusters_hierarchical == i)
    print(f"  Cluster {i}: {count} pingüinos")

# Métricas de evaluación
silhouette_hier = silhouette_score(X_scaled, clusters_hierarchical)
davies_bouldin_hier = davies_bouldin_score(X_scaled, clusters_hierarchical)
calinski_harabasz_hier = calinski_harabasz_score(X_scaled, clusters_hierarchical)

print("\nMétricas de evaluación:")
print(f"  Silhouette Score: {silhouette_hier:.4f}")
print(f"  Davies-Bouldin Index: {davies_bouldin_hier:.4f}")
print(f"  Calinski-Harabasz Index: {calinski_harabasz_hier:.2f}")

# ============================================================
# 12. DBSCAN CLUSTERING
# ============================================================
print("\n12. DBSCAN CLUSTERING")
print("-" * 80)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
n_noise = list(clusters_dbscan).count(-1)

print("\nModelo DBSCAN entrenado")
print(f"Número de clusters encontrados: {n_clusters_dbscan}")
print(f"Número de puntos de ruido: {n_noise}")

print("\nDistribución de muestras por cluster:")
for i in set(clusters_dbscan):
    count = np.sum(clusters_dbscan == i)
    if i == -1:
        print(f"  Ruido: {count} pingüinos")
    else:
        print(f"  Cluster {i}: {count} pingüinos")

if n_clusters_dbscan > 1:
    # Solo calcular métricas si hay más de un cluster
    clusters_dbscan_filtered = clusters_dbscan[clusters_dbscan != -1]
    X_scaled_filtered = X_scaled[clusters_dbscan != -1]
    
    if len(set(clusters_dbscan_filtered)) > 1:
        silhouette_dbscan = silhouette_score(X_scaled_filtered, clusters_dbscan_filtered)
        print("\nMétricas de evaluación (sin ruido):")
        print(f"  Silhouette Score: {silhouette_dbscan:.4f}")

# ============================================================
# 13. VISUALIZACIÓN CON PCA (2D)
# ============================================================
print("\n13. VISUALIZACIÓN CON PCA (REDUCCIÓN A 2D)")
print("-" * 80)

# Reducir a 2 dimensiones para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nVarianza explicada por componentes principales:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# Visualizar clusters
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Etiquetas reales (especies)
for especie in df['species'].unique():
    mask = (etiquetas_reales == especie)
    axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=especie, s=100, alpha=0.7, color=colores[especie],
                      edgecolors='black', linewidth=0.5)
axes[0, 0].set_xlabel('PC1', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('PC2', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Etiquetas Reales (Especies)', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# K-Means
colores_clusters = ['red', 'blue', 'green']
for i in range(3):
    mask = (clusters_kmeans == i)
    axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Cluster {i}', s=100, alpha=0.7, color=colores_clusters[i],
                      edgecolors='black', linewidth=0.5)
# Centroides
centroides_pca = pca.transform(kmeans.cluster_centers_)
axes[0, 1].scatter(centroides_pca[:, 0], centroides_pca[:, 1], 
                  marker='X', s=300, c='yellow', edgecolors='black', linewidth=2, label='Centroides')
axes[0, 1].set_xlabel('PC1', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('PC2', fontsize=12, fontweight='bold')
axes[0, 1].set_title(f'K-Means (Silhouette: {silhouette_kmeans:.3f})', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Clustering Jerárquico
for i in range(3):
    mask = (clusters_hierarchical == i)
    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f'Cluster {i}', s=100, alpha=0.7, color=colores_clusters[i],
                      edgecolors='black', linewidth=0.5)
axes[1, 0].set_xlabel('PC1', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('PC2', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'Jerárquico (Silhouette: {silhouette_hier:.3f})', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# DBSCAN
unique_labels = set(clusters_dbscan)
colors_dbscan = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors_dbscan):
    if k == -1:
        col = 'gray'
        label = 'Ruido'
    else:
        label = f'Cluster {k}'
    
    mask = (clusters_dbscan == k)
    axes[1, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=label, s=100, alpha=0.7, color=col,
                      edgecolors='black', linewidth=0.5)
axes[1, 1].set_xlabel('PC1', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('PC2', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'DBSCAN ({n_clusters_dbscan} clusters, {n_noise} ruido)', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imgs/04_clusterizacion_5_clusters_pca.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 14. COMPARACIÓN DE ALGORITMOS
# ============================================================
print("\n14. COMPARACIÓN DE ALGORITMOS")
print("-" * 80)

comparacion = pd.DataFrame({
    'Algoritmo': ['K-Means', 'Jerárquico', 'DBSCAN'],
    'Silhouette Score': [silhouette_kmeans, silhouette_hier, 
                         silhouette_dbscan if n_clusters_dbscan > 1 else 0],
    'Davies-Bouldin': [davies_bouldin_kmeans, davies_bouldin_hier, '-'],
    'Calinski-Harabasz': [calinski_harabasz_kmeans, calinski_harabasz_hier, '-']
})

print(f"\n{comparacion.to_string(index=False)}")

# Visualizar comparación
fig, ax = plt.subplots(figsize=(10, 6))
algoritmos = ['K-Means', 'Jerárquico']
scores = [silhouette_kmeans, silhouette_hier]
colors_bars = ['steelblue', 'forestgreen']

bars = ax.bar(algoritmos, scores, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Algoritmos de Clustering', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, 
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('imgs/04_clusterizacion_6_comparacion_algoritmos.jpg', format='jpg', bbox_inches='tight', dpi=100)
plt.show()

# ============================================================
# 15. ANÁLISIS DE CORRESPONDENCIA CON ESPECIES REALES
# ============================================================
print("\n15. CORRESPONDENCIA CON ESPECIES REALES")
print("-" * 80)

# Crear tabla de contingencia
df_resultado = df.copy()
df_resultado['cluster_kmeans'] = clusters_kmeans

tabla_contingencia = pd.crosstab(df_resultado['species'], 
                                  df_resultado['cluster_kmeans'],
                                  margins=True)

print("\nTabla de Contingencia (Especies vs Clusters K-Means):")
print(tabla_contingencia)

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"""
Dataset: Pingüinos de Palmer ({df.shape[0]} muestras, {len(caracteristicas)} características)
Características: {', '.join(caracteristicas)}
Especies reales: {', '.join(df['species'].unique())}

Normalización: StandardScaler aplicado
Reducción dimensional: PCA para visualización (varianza explicada: {pca.explained_variance_ratio_.sum()*100:.2f}%)

ALGORITMOS APLICADOS:
1. K-Means (k=3):
   - Silhouette Score: {silhouette_kmeans:.4f}
   - Davies-Bouldin: {davies_bouldin_kmeans:.4f}
   - Calinski-Harabasz: {calinski_harabasz_kmeans:.2f}

2. Clustering Jerárquico (k=3):
   - Silhouette Score: {silhouette_hier:.4f}
   - Davies-Bouldin: {davies_bouldin_hier:.4f}
   - Calinski-Harabasz: {calinski_harabasz_hier:.2f}

3. DBSCAN:
   - Clusters encontrados: {n_clusters_dbscan}
   - Puntos de ruido: {n_noise}

MEJOR ALGORITMO: {'K-Means' if silhouette_kmeans > silhouette_hier else 'Jerárquico'}

Los clusters encontrados corresponden muy bien con las 3 especies de pingüinos,
demostrando que las características físicas son distintivas para cada especie.
""")
print("="*80 + "\n")
