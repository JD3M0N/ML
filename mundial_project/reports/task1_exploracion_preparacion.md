# Task 1 - Exploracion y Preparacion de Datos

## Objetivo

Esta entrega cubre la primera parte del proyecto: explorar a fondo el dataset medico anonimizado, revisar su calidad, manejar valores faltantes y anomalias, transformar las variables para modelos matematicos y generar visualizaciones relacionadas con la variable objetivo `condition`.

## Hallazgos principales

El dataset original `data/foot.csv` contiene 297 registros, 13 features y la variable objetivo binaria `condition`. La clase 0 tiene 160 pacientes y la clase 1 tiene 137 pacientes, por lo que no existe un desbalance extremo en la variable objetivo.

La auditoria reproducible se guarda en `outputs/task1/calidad_datos.csv`. En la version actual del dataset no se encontraron valores faltantes ni filas duplicadas. Todas las columnas son numericas, lo que permite aplicar transformaciones matematicas directas despues de separar `condition`.

El resumen por feature se guarda en `outputs/task1/resumen_features.csv`. Ahi se documentan tipo de dato, cardinalidad, rango, media, desviacion estandar, clasificacion de la feature y diferencias de media entre `condition=0` y `condition=1`.

## Limpieza y manejo de anomalias

El script `src/task1_preparacion_datos.py` elimina duplicados si aparecen en futuras ejecuciones. Tambien incluye imputacion reproducible: mediana para columnas numericas y moda para columnas no numericas. En el CSV actual no fue necesario imputar porque no hay nulos.

Las anomalias se detectan con el metodo IQR y se reportan, pero no se eliminan automaticamente. Esta decision es conservadora porque, en un contexto medico, los valores extremos pueden representar casos clinicamente importantes y no simples errores de captura. Por eso el dataset limpio conserva los registros originales despues de quitar duplicados y resolver nulos si existieran.

## Transformaciones preparadas

Se generan dos datasets preparados:

- `data/processed/foot_clean.csv`: datos limpios con las features originales y `condition`.
- `data/processed/foot_model_ready.csv`: features numericas estandarizadas con media 0 y desviacion estandar 1, manteniendo `condition` sin transformar.

La estandarizacion deja las variables en una escala comparable para modelos matematicos sensibles a magnitudes, como metodos basados en distancia, PCA, regresiones regularizadas o modelos lineales.

## Visualizaciones de Task 1

Las visualizaciones especificas de Task 1 se guardan en `outputs/task1/`:

- `balance_condition.png`: distribucion de la variable objetivo.
- `features_continuas_por_condition.png`: boxplots de features continuas separadas por clase.
- `features_discretas_por_condition.png`: barras normalizadas de features binarias o discretas por clase.
- `resumen_features.png`: tabla visual con rangos, nulos, outliers reportados y diferencias entre clases.

Estas graficas complementan las salidas ya existentes del proyecto: rangos, estadisticas resumen, frecuencias, correlaciones, estandarizacion, matriz de dispersion y analisis de outliers.

## Estado para las siguientes tareas

Con esta preparacion, Task 1 queda reproducible y trazable. Task 2 puede usar `foot_model_ready.csv` para PCA o clustering. Task 3 puede usar `foot_clean.csv` o `foot_model_ready.csv` segun el modelo supervisado elegido, siempre conservando `condition` como variable objetivo.
