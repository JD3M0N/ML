# Task 4 - Optimizacion de Parametros

## Estrategia

Se optimizaron los tres modelos mas prometedores de Task 3: Regresion Logistica, SVM RBF y KNN. La busqueda se guio por `recall_condition_1`, porque el falso negativo es el error mas costoso en este contexto medico. Como desempate se uso menor coste medico normalizado y luego mayor ROC-AUC.

## Comparacion de modelos optimizados

| modelo | cv_recall_condition_1_mean | cv_medical_cost_mean | cv_roc_auc_mean | test_recall_condition_1 | test_false_negatives | test_false_positives | test_medical_cost | delta_cv_recall | delta_cv_medical_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM RBF | 1.0000 | 0.5401 | 0.8463 | 0.0000 | 28.0000 | 0.0000 | 2.3333 | 0.1934 | 0.0230 |
| Regresion Logistica | 0.8480 | 0.4217 | 0.9224 | 0.8214 | 5.0000 | 8.0000 | 0.5500 | 0.0344 | -0.0718 |
| KNN | 0.7955 | 0.5372 | 0.9097 | 0.8214 | 5.0000 | 5.0000 | 0.5000 | 0.0102 | -0.0353 |

## Mejores configuraciones

- **SVM RBF**: `model__C`=0.1, `model__class_weight`=balanced, `model__gamma`=1.0
- **Regresion Logistica**: `model__C`=1.0, `model__class_weight`=balanced, `model__penalty`=l1, `model__solver`=liblinear
- **KNN**: `model__n_neighbors`=7, `model__p`=1, `model__weights`=distance

## Lectura medica

El modelo recomendado por validacion cruzada es **SVM RBF**, con recall medio 1.0000 y coste medico medio 0.5401. En test final, el menor coste medico lo obtiene **KNN**, con 5 falsos negativos y 5 falsos positivos.

La seleccion principal se mantiene en validacion cruzada, no en accuracy ni en una sola particion de test. Un aumento de falsos positivos puede aceptarse si reduce falsos negativos, pero debe revisarse junto al coste medico.

## Variables mas influyentes

La influencia se calculo con `permutation_importance` usando recall como scoring sobre el pipeline completo. Las variables con mayor influencia global fueron:

- `feature_8`: importancia normalizada media 0.6667
- `feature_12`: importancia normalizada media 0.1893
- `feature_5`: importancia normalizada media 0.0988
- `feature_4`: importancia normalizada media 0.0782
- `feature_2`: importancia normalizada media 0.0658
