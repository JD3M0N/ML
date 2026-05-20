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

El metodo del codo extendido se evaluo para k=1..15. La inercia baja al aumentar k, como es esperable, pero no aparece una rodilla posterior que justifique una segmentacion mucho mas fina.
Se conserva k=2 como particion principal porque separa dos perfiles generales, mantiene una interpretacion clara y se relaciona de forma marcada con `condition` en el cruce posterior.
Tambien se compararon k=3 y k=5 como lecturas mas granulares; generan subgrupos mas especificos, pero fragmentan la muestra y se dejan como exploracion secundaria.
Los perfiles de clusters y el cruce posterior con `condition` se guardaron para interpretar que variables caracterizan cada grupo.
Las variables con mayor diferencia media entre clusters fueron: feature_10, feature_6, feature_5, feature_11, feature_13.
En el cruce posterior, el cluster 0 contiene 77.6% de pacientes con `condition=0`, mientras que el cluster 1 contiene 72.3% de pacientes con `condition=1`.

## Clustering jerarquico

El analisis de disparidad del dendrograma muestra su mayor salto de altura en la fusion 296, con salto=7.8584.
Ese salto ocurre al final del proceso, antes de fusionar los dos macrogrupos restantes, lo que apoya una lectura de dos grupos principales.
Esta tecnica sirve como contraste frente a K-Means porque llega a una lectura global compatible desde otra familia de clustering.

## DBSCAN

La configuracion con dos clusters y menos ruido fue eps=2.0 y min_samples=5, pero aun conserva 185 puntos de ruido.

## Trabajo extra: silhouette y t-SNE

Silhouette se conserva como metrica complementaria: en la evaluacion base el mayor valor fue k=2, con silhouette=0.1682.
En clustering jerarquico, la mejor lectura complementaria por silhouette fue k=2, con silhouette=0.1562.
t-SNE se conserva como visualizacion adicional para mirar vecindarios locales, pero no se usa como criterio principal de seleccion.

## Conclusion

Los experimentos sugieren una estructura latente parcialmente relacionada con `condition`, especialmente al combinar K-Means con k=2, el metodo del codo, la disparidad jerarquica y el cruce posterior con la variable objetivo. Los clusters son perfiles exploratorios de pacientes y no diagnosticos.