# Task 4 - SVM RBF Optimizado

## Objetivo

Este experimento optimiza hiperparametros para predecir `condition`, guiado por el coste medico normalizado con FN=5 y FP=1. En el contexto medico del proyecto se prioriza detectar pacientes con `condition=1`, pero se evita seleccionar configuraciones degeneradas que maximizan recall a costa de muchos falsos positivos o mala generalizacion.

## Mejor configuracion

```json
{
  "model__C": 1.0,
  "model__class_weight": "balanced",
  "model__gamma": "scale"
}
```

## Validacion cruzada

- Recall condition=1: 0.8337
- Coste medico normalizado: 0.4601
- ROC-AUC: 0.9124

## Test final reservado

- Recall condition=1: 0.8571
- Falsos negativos: 4
- Falsos positivos: 8
- Coste medico normalizado: 0.4667
- ROC-AUC: 0.8795

## Variables influyentes

La importancia se calculo con `permutation_importance` sobre el pipeline completo y usando recall como scoring. Las variables con mayor caida de recall al permutarse son:

- `feature_7`: importancia media 0.0548
- `feature_8`: importancia media 0.0452
- `feature_5`: importancia media 0.0381
- `feature_12`: importancia media 0.0274
- `feature_3`: importancia media 0.0250
