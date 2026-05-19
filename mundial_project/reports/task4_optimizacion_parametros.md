# Task 4 - Optimizacion de Parametros

## Estrategia

Se optimizaron los tres modelos mas prometedores de Task 3: Regresion Logistica, SVM RBF y KNN. La busqueda se guio por coste medico normalizado con FN=5 y FP=1. Esta metrica sigue penalizando mas los falsos negativos, pero evita seleccionar configuraciones degeneradas que consiguen recall alto prediciendo demasiados positivos. Como desempate se uso mayor `recall_condition_1` y luego mayor ROC-AUC.

## Comparacion de modelos optimizados

| modelo | cv_recall_condition_1_mean | cv_medical_cost_mean | cv_roc_auc_mean | test_recall_condition_1 | test_false_negatives | test_false_positives | test_medical_cost | delta_cv_recall | delta_cv_medical_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regresion Logistica | 0.8480 | 0.4217 | 0.9224 | 0.8214 | 5.0000 | 8.0000 | 0.5500 | 0.0344 | -0.0718 |
| SVM RBF | 0.8337 | 0.4601 | 0.9124 | 0.8571 | 4.0000 | 8.0000 | 0.4667 | 0.0270 | -0.0569 |
| KNN | 0.7955 | 0.5372 | 0.9097 | 0.8214 | 5.0000 | 5.0000 | 0.5000 | 0.0102 | -0.0353 |

## Mejores configuraciones

- **Regresion Logistica**: `model__C`=1.0, `model__class_weight`=balanced, `model__penalty`=l1, `model__solver`=liblinear
- **SVM RBF**: `model__C`=1.0, `model__class_weight`=balanced, `model__gamma`=scale
- **KNN**: `model__n_neighbors`=7, `model__p`=1, `model__weights`=distance

## Lectura medica

El modelo recomendado por validacion cruzada es **Regresion Logistica**, con recall medio 0.8480 y coste medico medio 0.4217. En test final, el menor coste medico lo obtiene **SVM RBF**, con 4 falsos negativos y 8 falsos positivos.

La seleccion principal se mantiene en validacion cruzada, no en accuracy ni en una sola particion de test. Un aumento de falsos positivos puede aceptarse si reduce falsos negativos, pero debe revisarse junto al coste medico.

## Variables mas influyentes

La influencia se calculo con `permutation_importance` usando recall como scoring sobre el pipeline completo. Las variables con mayor influencia global fueron:

- `feature_8`: importancia normalizada media 0.9420
- `feature_7`: importancia normalizada media 0.3786
- `feature_12`: importancia normalizada media 0.3560
- `feature_5`: importancia normalizada media 0.3306
- `feature_4`: importancia normalizada media 0.2159
