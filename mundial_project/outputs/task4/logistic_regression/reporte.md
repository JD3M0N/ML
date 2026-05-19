# Task 4 - Regresion Logistica Optimizada

## Objetivo

Este experimento optimiza hiperparametros para predecir `condition`, guiado por `recall_condition_1`. En el contexto medico del proyecto se prioriza detectar pacientes con `condition=1`, porque el falso negativo es el error mas costoso. El coste auxiliar mantiene FN=5 y FP=1.

## Mejor configuracion

```json
{
  "model__C": 1.0,
  "model__class_weight": "balanced",
  "model__penalty": "l1",
  "model__solver": "liblinear"
}
```

## Validacion cruzada

- Recall condition=1: 0.8480
- Coste medico normalizado: 0.4217
- ROC-AUC: 0.9224

## Test final reservado

- Recall condition=1: 0.8214
- Falsos negativos: 5
- Falsos positivos: 8
- Coste medico normalizado: 0.5500
- ROC-AUC: 0.8873

## Variables influyentes

La importancia se calculo con `permutation_importance` sobre el pipeline completo y usando recall como scoring. Las variables con mayor caida de recall al permutarse son:

- `feature_8`: importancia media 0.0321
- `feature_5`: importancia media 0.0000
- `feature_4`: importancia media 0.0000
- `feature_6`: importancia media 0.0000
- `feature_1`: importancia media 0.0000
