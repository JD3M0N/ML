# Task 4 - KNN Optimizado

## Objetivo

Este experimento optimiza hiperparametros para predecir `condition`, guiado por el coste medico normalizado con FN=5 y FP=1. En el contexto medico del proyecto se prioriza detectar pacientes con `condition=1`, pero se evita seleccionar configuraciones degeneradas que maximizan recall a costa de muchos falsos positivos o mala generalizacion.

## Mejor configuracion

```json
{
  "model__n_neighbors": 7,
  "model__p": 1,
  "model__weights": "distance"
}
```

## Validacion cruzada

- Recall condition=1: 0.7955
- Coste medico normalizado: 0.5372
- ROC-AUC: 0.9097

## Test final reservado

- Recall condition=1: 0.8214
- Falsos negativos: 5
- Falsos positivos: 5
- Coste medico normalizado: 0.5000
- ROC-AUC: 0.8555

## Variables influyentes

La importancia se calculo con `permutation_importance` sobre el pipeline completo y usando recall como scoring. Las variables con mayor caida de recall al permutarse son:

- `feature_8`: importancia media 0.0964
- `feature_12`: importancia media 0.0548
- `feature_5`: importancia media 0.0286
- `feature_4`: importancia media 0.0226
- `feature_2`: importancia media 0.0190
