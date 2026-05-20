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

## Diagnostico de la curva de aprendizaje

Diagnostico automatizado:
Train recall (ult.): 0.674  |  Val recall (ult.): 0.625  |  Gap: 0.050  |  Tendencia: estable
Interpretacion: Buen ajuste: poca diferencia entre entrenamiento y validacion y recall razonable en validacion.

## Posibles causas de bajo accuracy en redes neuronales y recomendaciones

- Desequilibrio de clases: si la clase positiva es rara, la accuracy global puede ser alta aunque el modelo falle en detectar la condicion.
- Infraajuste: arquitectura o capacidad insuficiente, learning rate inapropiado, o pocas iteraciones.
- Sobreajuste: modelo demasiado complejo sin regularizacion adecuada; gap grande entre entrenamiento y validacion.
- Preprocesado: features irrelevantes o mal escaladas afectan la convergencia; revisar estandarizacion y encoding.
- Hiperparametros: `alpha` (regularizacion), `learning_rate_init`, `hidden_layer_sizes` y `max_iter` influyen fuertemente.
- Early stopping: si esta activado puede detener antes de convergencia si la validacion es ruidosa.

Recomendaciones:
- Revisar balance de clases y usar `class_weight='balanced'` o re-muestreo si procede.
- Probar aumentar `max_iter`, ajustar `learning_rate_init` y `alpha`, y explorar diferentes `hidden_layer_sizes`.
- Usar validacion cruzada estable y observar curvas de aprendizaje (ya generadas) para decidir si mas datos ayudarian.
- Priorizar metrics de interes (recall para la clase positiva) en lugar de accuracy.
