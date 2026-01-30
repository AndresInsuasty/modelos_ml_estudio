import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def analizar_datos(archivo, titulo, sufijo_archivo):
    """
    Lee un archivo Excel, muestra los puntos en scatter plot
    y ajusta una regresión lineal.
    
    Parameters:
    archivo (str): Ruta del archivo Excel
    titulo (str): Título para las gráficas
    sufijo_archivo (str): Sufijo para los nombres de archivo (ej: 'datos_lineales_sin_ruido')
    """
    # Leer el archivo Excel
    df = pd.read_excel(archivo)
    
    # Mostrar las primeras filas del dataframe
    print(f"\n{'='*60}")
    print(f"Análisis: {titulo}")
    print(f"{'='*60}")
    print("Primeras filas del dataframe:")
    print(df.head())
    print(f"\nDimensiones: {df.shape}")
    
    # Preparar los datos para la regresión
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    
    # Entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Obtener parámetros
    pendiente = modelo.coef_[0]
    intercepto = modelo.intercept_
    r2 = modelo.score(X, y)
    
    # Mostrar información de la regresión
    print(f"\nEcuación: y = {pendiente:.4f}x + {intercepto:.4f}")
    print(f"R² score: {r2:.4f}")
    
    # Crear la gráfica scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['x'], df['y'], color='blue', s=100, alpha=0.7, 
                edgecolors='navy', linewidth=1.5, label='Datos')
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.title(f'{titulo} - Scatter Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'imgs/01_regresion_{sufijo_archivo}_scatter.jpg', format='jpg', bbox_inches='tight', dpi=100)
    plt.show()
    
    # Crear la gráfica con regresión lineal
    plt.figure(figsize=(12, 7))
    
    # Scatter plot
    plt.scatter(df['x'], df['y'], color='blue', s=100, alpha=0.6, 
                label='Datos originales', edgecolors='navy', linewidth=1.5)
    
    # Línea de regresión
    y_predicho = modelo.predict(X)
    indices_ordenados = np.argsort(X.flatten())
    plt.plot(X[indices_ordenados], y_predicho[indices_ordenados], 
             color='red', linewidth=3, 
             label=f'Regresión: y = {pendiente:.4f}x + {intercepto:.4f}')
    
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('y', fontsize=12, fontweight='bold')
    plt.title(f'{titulo} - Con Regresión Lineal', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'imgs/01_regresion_{sufijo_archivo}_modelo_ajustado.jpg', format='jpg', bbox_inches='tight', dpi=100)
    plt.show()


# ============================================================
# ANÁLISIS DEL PRIMER CONJUNTO DE DATOS (SIN RUIDO)
# ============================================================
analizar_datos('data/01_datos_lineales.xlsx', 'Datos Lineales - Sin Ruido', 'datos_lineales')

# ============================================================
# ANÁLISIS DEL SEGUNDO CONJUNTO DE DATOS (CON RUIDO)
# ============================================================
analizar_datos('data/02_datos_lineales_ruido.xlsx', 'Datos Lineales - Con Ruido', 'datos_lineales_con_ruido')
