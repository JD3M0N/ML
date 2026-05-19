# Task 2 - Aprendizaje No Supervisado

## Objetivo

El objetivo fue explorar estructuras ocultas en los pacientes sin usar `condition` para entrenar los modelos no supervisados. La variable objetivo se uso solo despues, para interpretar si los grupos descubiertos se relacionan con la condicion medica.

## Preparacion

Fuente de datos usada: `data/processed/foot_clean.csv`.
Las variables continuas se estandarizaron, las discretas se codificaron con one-hot encoding y las binarias se conservaron sin cambios.

## PCA

Las dos primeras componentes explicaron 39.93% de la varianza acumulada.
Se necesitaron 8 componentes para alcanzar al menos 80% de varianza y 12 componentes para alcanzar al menos 90%.

## K-Means

El mejor valor evaluado por silhouette fue k=2, con silhouette=0.1682, Davies-Bouldin=2.0747 y Calinski-Harabasz=65.2677.
Los perfiles de clusters y el cruce posterior con `condition` se guardaron para interpretar que variables caracterizan cada grupo.
Las variables con mayor diferencia media entre clusters fueron: feature_10, feature_6, feature_5, feature_11, feature_13.
En el cruce posterior, el cluster 0 contiene 77.6% de pacientes con `condition=0`, mientras que el cluster 1 contiene 72.3% de pacientes con `condition=1`.

## Clustering jerarquico

El mejor valor evaluado fue k=2, con silhouette=0.1562. Esta tecnica sirve como contraste frente a K-Means.

## DBSCAN

La mejor configuracion evaluable fue eps=2.0 y min_samples=5, con 2 clusters, 185 puntos de ruido y silhouette=0.0166.

## Conclusion

Los experimentos sugieren una estructura latente parcialmente relacionada con `condition`, especialmente al comparar los clusters de K-Means con la variable objetivo despues del agrupamiento. Aun asi, las metricas internas deben interpretarse con cautela: los clusters son perfiles exploratorios de pacientes y no diagnosticos.