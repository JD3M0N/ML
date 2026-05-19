# Task 4 - SVM RBF Optimizado

## Objetivo

Este experimento optimiza hiperparametros para predecir `condition`, guiado por `recall_condition_1`. En el contexto medico del proyecto se prioriza detectar pacientes con `condition=1`, porque el falso negativo es el error mas costoso. El coste auxiliar mantiene FN=5 y FP=1.

## Mejor configuracion

```json
{
  "model__C": 0.1,
  "model__class_weight": "balanced",
  "model__gamma": 1.0
}
```

## Validacion cruzada

- Recall condition=1: 1.0000
- Coste medico normalizado: 0.5401
- ROC-AUC: 0.8463

## Test final reservado

- Recall condition=1: 0.0000
- Falsos negativos: 28
- Falsos positivos: 0
- Coste medico normalizado: 2.3333
- ROC-AUC: 0.1842

## Variables influyentes

La importancia se calculo con `permutation_importance` sobre el pipeline completo y usando recall como scoring. Las variables con mayor caida de recall al permutarse son:

- `feature_1`: importancia media 0.0000
- `feature_2`: importancia media 0.0000
- `feature_3`: importancia media 0.0000
- `feature_4`: importancia media 0.0000
- `feature_5`: importancia media 0.0000
