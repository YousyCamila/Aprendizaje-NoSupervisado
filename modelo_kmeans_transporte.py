"""
Modelo de Aprendizaje No Supervisado - K-Means Clustering
Sistema de Transporte Masivo Urbano
Materia: Inteligencia Artificial - Metodos No Supervisados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. CARGA Y EXPLORACION DE DATOS
# ─────────────────────────────────────────────
print("=" * 60)
print("SISTEMA DE CLUSTERING - TRANSPORTE MASIVO URBANO")
print("=" * 60)

df = pd.read_csv('dataset_transporte.csv')

print(f"\n[1] Dataset cargado: {df.shape[0]} registros, {df.shape[1]} variables")
print("\nPrimeras filas:")
print(df.head())
print("\nEstadisticas descriptivas:")
print(df[['hora_salida','pasajeros','distancia_km','tiempo_viaje_min','temperatura_c','lluvia_mm']].describe().round(2))
print("\nDistribucion por ruta:")
print(df['ruta'].value_counts())

# ─────────────────────────────────────────────
# 2. SELECCION Y PREPROCESAMIENTO DE VARIABLES
# ─────────────────────────────────────────────
print("\n[2] Preprocesamiento de variables...")


features = ['hora_salida', 'pasajeros', 'distancia_km', 'tiempo_viaje_min', 'temperatura_c', 'lluvia_mm']
X = df[features].copy()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   Variables seleccionadas: {features}")
print("   Escalado con StandardScaler: OK")

# ─────────────────────────────────────────────
# 3. METODO DEL CODO (Elbow Method)
# ─────────────────────────────────────────────
print("\n[3] Ejecutando metodo del codo para encontrar K optimo...")

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil)
    print(f"   K={k}: Inercia={km.inertia_:.2f} | Silhouette={sil:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Seleccion del numero optimo de clusters (K)', fontsize=14, fontweight='bold')

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='K optimo = 4')
axes[0].set_xlabel('Numero de clusters (K)')
axes[0].set_ylabel('Inercia (suma de distancias al centroide)')
axes[0].set_title('Metodo del Codo')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 'gs-', linewidth=2, markersize=8)
axes[1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='K optimo = 4')
axes[1].set_xlabel('Numero de clusters (K)')
axes[1].set_ylabel('Coeficiente de Silhouette')
axes[1].set_title('Analisis de Silhouette')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafico_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Grafico guardado: grafico_elbow.png")

# ─────────────────────────────────────────────
# 4. MODELO FINAL K-MEANS (K=4)
# ─────────────────────────────────────────────
K_OPTIMO = 4
print(f"\n[4] Entrenando modelo K-Means con K={K_OPTIMO}...")

kmeans = KMeans(n_clusters=K_OPTIMO, init='k-means++', n_init=20, random_state=42, max_iter=300)
kmeans.fit(X_scaled)

df['cluster'] = kmeans.labels_
df['cluster_label'] = df['cluster'].astype(str)

print(f"   Inercia final: {kmeans.inertia_:.4f}")
print(f"   Iteraciones necesarias: {kmeans.n_iter_}")
print(f"   Distribucion de clusters:")
print(df['cluster'].value_counts().sort_index())

# ─────────────────────────────────────────────
# 5. METRICAS DE EVALUACION
# ─────────────────────────────────────────────
print("\n[5] Calculando metricas de evaluacion...")

sil_score = silhouette_score(X_scaled, kmeans.labels_)
db_score = davies_bouldin_score(X_scaled, kmeans.labels_)

print(f"   Silhouette Score:        {sil_score:.4f}  (rango [-1,1], mayor es mejor)")
print(f"   Davies-Bouldin Index:    {db_score:.4f}  (menor es mejor)")
print(f"   Inercia:                 {kmeans.inertia_:.4f}")

# ─────────────────────────────────────────────
# 6. ANALISIS DE CENTROIDES
# ─────────────────────────────────────────────
print("\n[6] Analisis de centroides (valores originales)...")

centroides_orig = scaler.inverse_transform(kmeans.cluster_centers_)
df_centroides = pd.DataFrame(centroides_orig, columns=features)
df_centroides.index.name = 'cluster'
print(df_centroides.round(2))


cluster_stats = df.groupby('cluster')[features].mean()
print("\nCaracterizacion de clusters:")
for c in range(K_OPTIMO):
    stats = cluster_stats.loc[c]
    pasajeros_nivel = "ALTO" if stats['pasajeros'] > 550 else ("MEDIO" if stats['pasajeros'] > 300 else "BAJO")
    hora_tipo = "PICO" if stats['hora_salida'] in [6,7,8,9,16,17,18,19] else "VALLE"
    n_viajes = len(df[df['cluster'] == c])
    print(f"   Cluster {c}: Demanda {pasajeros_nivel} | Hora {hora_tipo} | {n_viajes} viajes | "
          f"Pasajeros prom: {stats['pasajeros']:.0f} | Hora prom: {stats['hora_salida']:.1f}h")

# ─────────────────────────────────────────────
# 7. VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n[7] Generando visualizaciones...")

colores = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
nombres_clusters = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3'}


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
varianza_explicada = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(range(K_OPTIMO)):
    mask = df['cluster'] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colores[i], label=f'Cluster {c} (n={mask.sum()})',
               alpha=0.7, s=60, edgecolors='white', linewidths=0.5)


centroides_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
           c='black', marker='X', s=200, zorder=5, label='Centroides')

ax.set_xlabel(f'Componente Principal 1 ({varianza_explicada[0]*100:.1f}% varianza)')
ax.set_ylabel(f'Componente Principal 2 ({varianza_explicada[1]*100:.1f}% varianza)')
ax.set_title('Visualizacion de Clusters - PCA 2D\nSistema de Transporte Masivo', fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(11, 6))
for i, c in enumerate(range(K_OPTIMO)):
    mask = df['cluster'] == c
    ax.scatter(df.loc[mask, 'hora_salida'], df.loc[mask, 'pasajeros'],
               c=colores[i], label=f'Cluster {c}', alpha=0.6, s=55,
               edgecolors='white', linewidths=0.5)

ax.set_xlabel('Hora de Salida')
ax.set_ylabel('Numero de Pasajeros')
ax.set_title('Clusters: Demanda de Pasajeros por Hora del Dia', fontsize=13)
ax.set_xticks(range(5, 23))
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvspan(6.5, 9.5, alpha=0.08, color='orange', label='Pico AM')
ax.axvspan(16.5, 19.5, alpha=0.08, color='orange')
plt.tight_layout()
plt.savefig('grafico_pasajeros_hora.png', dpi=150, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 5))
df_cent_norm = pd.DataFrame(kmeans.cluster_centers_, columns=features)
sns.heatmap(df_cent_norm.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
            xticklabels=[f'Cluster {i}' for i in range(K_OPTIMO)],
            yticklabels=features, ax=ax, linewidths=0.5)
ax.set_title('Centroides Normalizados por Cluster\n(valores en desviaciones estandar)', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_heatmap_centroides.png', dpi=150, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(11, 6))
pivot = pd.crosstab(df['ruta'], df['cluster'])
pivot.plot(kind='bar', ax=ax, color=colores[:K_OPTIMO], edgecolor='white', width=0.75)
ax.set_xlabel('Ruta')
ax.set_ylabel('Numero de Viajes')
ax.set_title('Distribucion de Clusters por Ruta', fontsize=13)
ax.legend([f'Cluster {i}' for i in range(K_OPTIMO)], title='Cluster')
ax.tick_params(axis='x', rotation=20)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('grafico_clusters_por_ruta.png', dpi=150, bbox_inches='tight')
plt.close()

print("   Graficos guardados:")
print("   - grafico_elbow.png")
print("   - grafico_pca_clusters.png")
print("   - grafico_pasajeros_hora.png")
print("   - grafico_heatmap_centroides.png")
print("   - grafico_clusters_por_ruta.png")

# ─────────────────────────────────────────────
# 8. EXPORTAR RESULTADOS
# ─────────────────────────────────────────────
print("\n[8] Exportando resultados...")
df.to_csv('dataset_transporte_clusterizado.csv', index=False)
print("   Archivo guardado: dataset_transporte_clusterizado.csv")

# ─────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN DEL MODELO")
print("=" * 60)
print(f"  Algoritmo:            K-Means (k-means++)")
print(f"  Numero de clusters:   {K_OPTIMO}")
print(f"  Registros procesados: {len(df)}")
print(f"  Variables usadas:     {len(features)}")
print(f"  Silhouette Score:     {sil_score:.4f}")
print(f"  Davies-Bouldin:       {db_score:.4f}")
print(f"  Inercia:              {kmeans.inertia_:.4f}")
print("=" * 60)
print("\nEjecucion completada exitosamente.")
