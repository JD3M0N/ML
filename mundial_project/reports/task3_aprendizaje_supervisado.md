# Task 3 - Aprendizaje Supervisado

## Estrategia de evaluación

Todos los modelos se evaluaron con la misma partición final estratificada y la misma validación interna: `RepeatedStratifiedKFold` con 5 folds y 30 repeticiones. El preprocesamiento se mantuvo dentro de un `Pipeline`, por lo que cada fold ajusta escalado y codificación solo con sus datos de entrenamiento. Por esta razón se usa `foot_clean.csv` y no un dataset estandarizado previamente con todo el conjunto.

La métrica principal es `recall_condition_1`, porque en un escenario de salud el error más grave es el falso negativo: clasificar como sano a un paciente que realmente presenta la condición. También se reportan F1, ROC-AUC, matriz de confusión y un coste médico normalizado con FN=5 y FP=1. El falso negativo se pondera cinco veces más porque puede retrasar diagnóstico o seguimiento; el falso positivo también tiene coste, pero no deja sin detectar un caso real.

Task 3 mantiene el umbral de decisión por defecto para comparar modelos base de forma homogénea. En un escenario clínico real se debería optimizar ese umbral para reducir falsos negativos, incluso si aumentan los falsos positivos.

| decisión | valor | justificación |
| --- | --- | --- |
| Test final | 20% estratificado | Reserva una evaluación final no usada durante la comparación interna. |
| Validación interna | RepeatedStratifiedKFold, 5 folds x 30 repeticiones | Reduce la dependencia de una única partición en un dataset moderado. |
| Dataset de entrada | foot_clean.csv + Pipeline | Evita fuga de información al ajustar escalado y one-hot solo con entrenamiento. |
| Métrica principal | recall_condition_1 | Mide cuántos pacientes con condition=1 son detectados. |
| Coste médico | FN=5, FP=1 | Penaliza más dejar sin detectar un caso real que activar una alarma falsa. |
| Umbral | Umbral por defecto | Permite comparar modelos base de forma homogénea antes de optimizar umbrales. |

## Experimentos individuales

| modelo | familia | por_que | fortaleza | limitacion | lectura_resultado | conclusion_experimento |
| --- | --- | --- | --- | --- | --- | --- |
| Regresión Logística | Modelo lineal probabilístico | Baseline interpretable para estimar una frontera lineal y probabilidades. | Alta interpretabilidad y buen equilibrio entre recall, ROC-AUC y coste. | Puede quedarse corta si la relación entre variables y condition no es lineal. | Recall CV 0.8135, coste médico CV 0.4935 y 5 falsos negativos en test. | Es el baseline más defendible: detecta cerca del 81% de positivos en CV, mantiene el menor coste médico medio y además permite explicar coeficientes. Su límite es que todavía deja 5 falsos negativos en test. |
| Naive Bayes | Modelo probabilístico simple | Contrasta un enfoque rápido y robusto con supuestos fuertes de independencia. | Sirve como baseline sencillo y suele funcionar con pocos datos. | El supuesto de independencia puede reducir recall si hay relaciones entre variables. | Recall CV 0.7282, coste médico CV 0.6984 y 5 falsos negativos en test. | Es útil como baseline probabilístico rápido, pero el supuesto de independencia parece limitarlo: presenta menor recall CV y mayor variabilidad que los modelos más robustos. Su buen test debe leerse con cautela por ser una sola partición. |
| KNN | Modelo basado en vecinos | Evalúa si pacientes cercanos en el espacio de variables comparten condition. | Captura fronteras no lineales sin imponer una forma paramétrica. | Es sensible al escalado, a ruido local y al tamaño efectivo del dataset. | Recall CV 0.7853, coste médico CV 0.5725 y 5 falsos negativos en test. | Funciona como contraste local: el rendimiento es razonable, pero queda por debajo de los mejores en recall CV y depende mucho de distancias y escalado. No ofrece una ventaja clínica clara frente a modelos más estables. |
| Árbol de Decisión CART | Árbol interpretable | Baseline de reglas explícitas para comparar contra modelos más flexibles. | Sus reglas son fáciles de explicar. | Puede ser inestable y perder capacidad predictiva frente a ensamblados. | Recall CV 0.7193, coste médico CV 0.7399 y 6 falsos negativos en test. | Aporta interpretabilidad mediante reglas, pero pierde rendimiento clínico: su coste CV sube a 0.7399 y en test deja 6 falsos negativos con 10 falsos positivos. Sirve como baseline, no como candidato final. |
| XGBoost | Ensamblado de árboles potenciados | Prueba un modelo de árboles más potente que CART sin sustituir el baseline simple. | Captura interacciones no lineales con regularización y boosting. | Menos interpretable que CART; debe justificar mejora real en métricas médicas. | Modelo potente de contraste: recall CV 0.8064, coste 0.5147 y 5 falsos negativos en test; solo reemplazaría a CART si mejora las métricas médicas. | Mejora claramente al árbol CART y captura interacciones no lineales, pero su recall CV (0.8064) no supera a Regresión Logística. Aporta evidencia de que boosting es útil, no de que deba reemplazar al modelo base. |
| SVM RBF | Margen máximo no lineal | Evalúa una frontera no lineal suave mediante kernel RBF. | Buen rendimiento en espacios no lineales de tamaño moderado. | Menos interpretable y dependiente de C/gamma. | Recall CV 0.8066, coste médico CV 0.5170 y 4 falsos negativos en test. | Confirma que una frontera no lineal ayuda: queda casi empatado con Regresión Logística en CV y en test reduce los falsos negativos a 4. Se considera alternativa prometedora, aunque menos interpretable. |
| Red Neuronal MLP | Red neuronal feed-forward | Documenta si un modelo más flexible mejora la detección de condition=1. | Puede modelar relaciones no lineales complejas. | Puede ser inestable en datasets pequeños y perder recall clínico. | Experimento débil: recall CV 0.5720, coste 1.0551 y 21 falsos negativos en test; se conserva como intento no recomendable. | El experimento muestra que más complejidad no garantiza mejor detección: recall test 0.2500 y 21 falsos negativos. En esta configuración no es recomendable para priorizar pacientes con condition=1. |

## Comparación consolidada

| rank | modelo | ranking_reason | cv_recall_condition_1_mean | cv_recall_condition_1_std | cv_roc_auc_mean | cv_medical_cost_mean | cv_false_negatives_mean | test_recall_condition_1 | test_roc_auc | test_false_negatives | test_false_positives | test_medical_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | Regresión Logística | Recall CV=0.8135, coste CV=0.4935, FN test=5. | 0.8135 | 0.0853 | 0.9160 | 0.4935 | 4.0667 | 0.8214 | 0.8962 | 5.0000 | 6.0000 | 0.5167 |
| 2.0000 | SVM RBF | Recall CV=0.8066, coste CV=0.5170, FN test=4. | 0.8066 | 0.0797 | 0.9112 | 0.5170 | 4.2133 | 0.8571 | 0.8817 | 4.0000 | 8.0000 | 0.4667 |
| 3.0000 | XGBoost | Recall CV=0.8064, coste CV=0.5147, FN test=5. | 0.8064 | 0.0725 | 0.9128 | 0.5147 | 4.2200 | 0.8214 | 0.8895 | 5.0000 | 7.0000 | 0.5333 |
| 4.0000 | KNN | Recall CV=0.7853, coste CV=0.5725, FN test=5. | 0.7853 | 0.0762 | 0.8806 | 0.5725 | 4.6800 | 0.8214 | 0.8488 | 5.0000 | 6.0000 | 0.5167 |
| 5.0000 | Naive Bayes | Recall CV=0.7282, coste CV=0.6984, FN test=5. | 0.7282 | 0.1496 | 0.8831 | 0.6984 | 5.9267 | 0.8214 | 0.8382 | 5.0000 | 5.0000 | 0.5000 |
| 6.0000 | Árbol de Decisión CART | Recall CV=0.7193, coste CV=0.7399, FN test=6. | 0.7193 | 0.0954 | 0.8253 | 0.7399 | 6.1133 | 0.7857 | 0.8248 | 6.0000 | 10.0000 | 0.6667 |
| 7.0000 | Red Neuronal MLP | Recall CV=0.5720, coste CV=1.0551, FN test=21. | 0.5720 | 0.1549 | 0.8051 | 1.0551 | 9.3267 | 0.2500 | 0.7143 | 21.0000 | 4.0000 | 1.8167 |

## Lectura de resultados

El mejor rendimiento medio en validación cruzada según la métrica principal lo obtiene **Regresión Logística**, con recall medio de 0.8135 y coste médico medio de 0.4935. Esta es la comparación más robusta porque promedia 150 validaciones estratificadas.

En el test final, el menor coste médico lo obtiene **SVM RBF**, con 4 falsos negativos, 8 falsos positivos y coste 0.4667. Este resultado es útil, pero debe interpretarse con cautela porque el test contiene una sola partición.

El peor comportamiento médico lo muestra **Red Neuronal MLP**, con coste medio de validación 1.0551. Este modelo genera demasiados falsos negativos para un escenario clínico y queda documentado como experimento no recomendable en esta configuración.

## Variables y lectura clínica

Las matrices de confusión se revisan junto al coste médico porque muestran directamente cuántos pacientes positivos quedan sin detectar. La curva ROC resume separación probabilística, pero no reemplaza el análisis de falsos negativos.

Para el MLP se añade una lectura de importancia por gradientes de entrada. Esta técnica mide sensibilidad del output ante cambios en las variables preprocesadas; sirve para interpretar el comportamiento del modelo, pero no implica causalidad clínica.

## Conclusión

Para selección inicial de Task 3, el modelo recomendado debe salir de la validación cruzada y no de una única partición de test. **Regresión Logística** queda como candidato base más sólido por recall medio, coste médico y estabilidad relativa. **SVM RBF** queda como alternativa a revisar por su comportamiento en test. XGBoost se incluye como contraste potente de árboles, mientras que CART se conserva como baseline interpretable. El MLP queda documentado incluso si falla, porque el enunciado exige registrar experimentos débiles o no recomendables.
