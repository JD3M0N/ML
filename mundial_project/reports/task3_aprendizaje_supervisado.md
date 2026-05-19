# Task 3 - Comparacion de Aprendizaje Supervisado

## Estrategia de evaluacion

Todos los modelos se evaluaron con la misma particion final estratificada y la misma validacion interna: `RepeatedStratifiedKFold` con 5 folds y 30 repeticiones. El preprocesamiento se mantuvo dentro de un `Pipeline`, por lo que cada fold ajusta escalado y codificacion solo con sus datos de entrenamiento.

La metrica principal es `recall_condition_1`, porque en un escenario de salud el error mas grave es el falso negativo: clasificar como sano a un paciente que realmente presenta la condicion. Tambien se reportan F1, ROC-AUC, matriz de confusion y un coste medico normalizado con FN=5 y FP=1. La accuracy se conserva como referencia secundaria, pero no debe decidir el mejor modelo.

## Comparacion consolidada

| modelo | cv_recall_condition_1_mean | cv_recall_condition_1_std | cv_roc_auc_mean | cv_medical_cost_mean | cv_false_negatives_mean | test_recall_condition_1 | test_roc_auc | test_false_negatives | test_false_positives | test_medical_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regresion Logistica | 0.8135 | 0.0853 | 0.9160 | 0.4935 | 4.0667 | 0.8214 | 0.8962 | 5.0000 | 6.0000 | 0.5167 |
| SVM RBF | 0.8066 | 0.0797 | 0.9112 | 0.5170 | 4.2133 | 0.8571 | 0.8817 | 4.0000 | 8.0000 | 0.4667 |
| KNN | 0.7853 | 0.0762 | 0.8806 | 0.5725 | 4.6800 | 0.8214 | 0.8488 | 5.0000 | 6.0000 | 0.5167 |
| Naive Bayes | 0.7282 | 0.1496 | 0.8831 | 0.6984 | 5.9267 | 0.8214 | 0.8382 | 5.0000 | 5.0000 | 0.5000 |
| Arbol de Decision | 0.7193 | 0.0954 | 0.8253 | 0.7399 | 6.1133 | 0.7857 | 0.8248 | 6.0000 | 10.0000 | 0.6667 |
| Red Neuronal MLP | 0.5720 | 0.1549 | 0.8051 | 1.0551 | 9.3267 | 0.2500 | 0.7143 | 21.0000 | 4.0000 | 1.8167 |

## Lectura de resultados

El mejor rendimiento medio en validacion cruzada segun la metrica principal lo obtiene **Regresion Logistica**, con recall medio de 0.8135 y coste medico medio de 0.4935. Esta es la comparacion mas robusta porque promedia 150 validaciones estratificadas.

En el test final, el menor coste medico lo obtiene **SVM RBF**, con 4 falsos negativos, 8 falsos positivos y coste 0.4667. Este resultado es importante, pero debe interpretarse con cautela porque el test contiene una sola particion.

El peor comportamiento medico lo muestra **Red Neuronal MLP**, con coste medio de validacion 1.0551. Este modelo genera demasiados falsos negativos para un escenario clinico y queda documentado como experimento no recomendable en esta configuracion.

## Conclusion

Para seleccion inicial de Task 3, el modelo mas solido es **Regresion Logistica**: logra el mayor recall medio en validacion cruzada, el mejor ROC-AUC medio y el menor coste medico medio. **SVM RBF** queda como alternativa prometedora porque en test reduce los falsos negativos a 4, aunque su validacion media queda levemente por debajo. **MLP** no debe priorizarse en esta configuracion porque falla precisamente en el error mas critico: deja muchos pacientes positivos sin detectar.
