# Task 3 - K-Nearest Neighbors

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
| Accuracy | 0.8225 | 0.0495 |
| Precision condition=1 | 0.8260 | 0.0715 |
| Recall condition=1 | 0.7853 | 0.0762 |
| F1 condition=1 | 0.8022 | 0.0561 |
| ROC-AUC | 0.8806 | 0.0477 |
| Coste medico | 0.5725 | 0.1774 |

## Resultados en test final

- Accuracy: 0.8167
- Precision condition=1: 0.7931
- Recall condition=1: 0.8214
- F1 condition=1: 0.8070
- ROC-AUC: 0.8488
- Falsos negativos: 5
- Falsos positivos: 6
- Coste medico normalizado: 0.5167

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
