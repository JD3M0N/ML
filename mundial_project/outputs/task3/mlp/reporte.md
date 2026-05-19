# Task 3 - Red Neuronal MLP

## Objetivo

Este experimento entrena un modelo supervisado para predecir `condition` y evalua su utilidad en un contexto medico, priorizando la reduccion de falsos negativos mediante `recall_condition_1`.

## Datos y validacion

- CSV usado: `data\processed\foot_clean.csv`.
- Test final: particion estratificada del 20%, reservada hasta el cierre del experimento.
- Validacion interna: `RepeatedStratifiedKFold` con 5 folds y 30 repeticiones.
- Preprocesamiento: estandarizacion de variables continuas, one-hot encoding de discretas y paso directo de binarias dentro del `Pipeline`.
- Coste medico: FN=5, FP=1, normalizado por numero de pacientes.

## Resultados de validacion cruzada

| Metrica | Media | Desviacion |
|---|---:|---:|
| Accuracy | 0.7320 | 0.0755 |
| Precision condition=1 | 0.7861 | 0.0946 |
| Recall condition=1 | 0.5720 | 0.1549 |
| F1 condition=1 | 0.6524 | 0.1247 |
| ROC-AUC | 0.8051 | 0.0821 |
| Coste medico | 1.0551 | 0.3544 |

## Resultados en test final

- Accuracy: 0.5833
- Precision condition=1: 0.6364
- Recall condition=1: 0.2500
- F1 condition=1: 0.3590
- ROC-AUC: 0.7143
- Falsos negativos: 21
- Falsos positivos: 4
- Coste medico normalizado: 1.8167

## Interpretacion medica

La metrica principal es el recall de `condition=1` porque un falso negativo implica clasificar como sano a un paciente con la condicion. La matriz de confusion debe revisarse junto al coste medico: si el recall es bajo o aparecen muchos falsos negativos, el modelo no deberia considerarse adecuado aunque tenga buena accuracy.

## Artefactos

- `cv_metricas_por_fold.csv`
- `cv_metricas_resumen.csv`
- `test_metricas.csv`
- `test_matriz_confusion.csv`
- `test_matriz_confusion.png`
- `test_roc_curve.png`
- `learning_curve_recall.png`
