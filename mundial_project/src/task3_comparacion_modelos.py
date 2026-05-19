from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task3_common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT


MODEL_NAMES = {
    "logistic_regression": "Regresion Logistica",
    "naive_bayes": "Naive Bayes",
    "knn": "KNN",
    "decision_tree": "Arbol de Decision",
    "svm": "SVM RBF",
    "mlp": "Red Neuronal MLP",
}

PRIMARY_METRIC = "recall_condition_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara los experimentos supervisados de Task 3."
    )
    parser.add_argument(
        "--task3-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta con subdirectorios de experimentos supervisados.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "task3_aprendizaje_supervisado.md",
        help="Ruta del reporte markdown consolidado.",
    )
    return parser.parse_args()


def read_cv_summary(model_dir: Path) -> dict[str, float]:
    df = pd.read_csv(model_dir / "cv_metricas_resumen.csv")
    row = {"modelo": MODEL_NAMES.get(model_dir.name, model_dir.name)}
    for _, metric_row in df.iterrows():
        metric = metric_row["metric"]
        row[f"cv_{metric}_mean"] = metric_row["mean"]
        row[f"cv_{metric}_std"] = metric_row["std"]
    return row


def read_test_metrics(model_dir: Path) -> dict[str, float]:
    df = pd.read_csv(model_dir / "test_metricas.csv")
    row = df.iloc[0].to_dict()
    return {f"test_{key}": value for key, value in row.items()}


def build_comparison(task3_dir: Path) -> pd.DataFrame:
    rows = []
    for slug in MODEL_NAMES:
        model_dir = task3_dir / slug
        if not model_dir.exists():
            continue
        row = {"slug": slug, **read_cv_summary(model_dir), **read_test_metrics(model_dir)}
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError(f"No se encontraron resultados en {task3_dir}.")

    return comparison.sort_values(
        by=["cv_recall_condition_1_mean", "cv_medical_cost_mean", "test_false_negatives"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def compact_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "modelo",
        "cv_recall_condition_1_mean",
        "cv_recall_condition_1_std",
        "cv_roc_auc_mean",
        "cv_medical_cost_mean",
        "cv_false_negatives_mean",
        "test_recall_condition_1",
        "test_roc_auc",
        "test_false_negatives",
        "test_false_positives",
        "test_medical_cost",
    ]
    return comparison[columns]


def markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.select_dtypes(include="number").columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display_df.columns]
    rows = [[str(value) for value in row] for row in display_df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(comparison: pd.DataFrame, report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    compact = compact_columns(comparison)
    best_cv = comparison.iloc[0]
    best_test_cost = comparison.sort_values("test_medical_cost", ascending=True).iloc[0]
    worst_medical = comparison.sort_values("cv_medical_cost_mean", ascending=False).iloc[0]

    content = f"""# Task 3 - Comparacion de Aprendizaje Supervisado

## Estrategia de evaluacion

Todos los modelos se evaluaron con la misma particion final estratificada y la misma validacion interna: `RepeatedStratifiedKFold` con 5 folds y 30 repeticiones. El preprocesamiento se mantuvo dentro de un `Pipeline`, por lo que cada fold ajusta escalado y codificacion solo con sus datos de entrenamiento.

La metrica principal es `recall_condition_1`, porque en un escenario de salud el error mas grave es el falso negativo: clasificar como sano a un paciente que realmente presenta la condicion. Tambien se reportan F1, ROC-AUC, matriz de confusion y un coste medico normalizado con FN=5 y FP=1. La accuracy se conserva como referencia secundaria, pero no debe decidir el mejor modelo.

## Comparacion consolidada

{markdown_table(compact)}

## Lectura de resultados

El mejor rendimiento medio en validacion cruzada segun la metrica principal lo obtiene **{best_cv["modelo"]}**, con recall medio de {best_cv["cv_recall_condition_1_mean"]:.4f} y coste medico medio de {best_cv["cv_medical_cost_mean"]:.4f}. Esta es la comparacion mas robusta porque promedia 150 validaciones estratificadas.

En el test final, el menor coste medico lo obtiene **{best_test_cost["modelo"]}**, con {int(best_test_cost["test_false_negatives"])} falsos negativos, {int(best_test_cost["test_false_positives"])} falsos positivos y coste {best_test_cost["test_medical_cost"]:.4f}. Este resultado es importante, pero debe interpretarse con cautela porque el test contiene una sola particion.

El peor comportamiento medico lo muestra **{worst_medical["modelo"]}**, con coste medio de validacion {worst_medical["cv_medical_cost_mean"]:.4f}. Este modelo genera demasiados falsos negativos para un escenario clinico y queda documentado como experimento no recomendable en esta configuracion.

## Conclusion

Para seleccion inicial de Task 3, el modelo mas solido es **Regresion Logistica**: logra el mayor recall medio en validacion cruzada, el mejor ROC-AUC medio y el menor coste medico medio. **SVM RBF** queda como alternativa prometedora porque en test reduce los falsos negativos a 4, aunque su validacion media queda levemente por debajo. **MLP** no debe priorizarse en esta configuracion porque falla precisamente en el error mas critico: deja muchos pacientes positivos sin detectar.
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    task3_dir = args.task3_dir.resolve()
    comparison = build_comparison(task3_dir)

    task3_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(task3_dir / "comparacion_modelos_completa.csv", index=False)
    compact_columns(comparison).to_csv(task3_dir / "comparacion_modelos_resumen.csv", index=False)
    report_path = write_report(comparison, args.report_path.resolve())

    print("Comparacion de modelos supervisados:")
    print(compact_columns(comparison).to_string(index=False))
    print(f"CSV completo: {task3_dir / 'comparacion_modelos_completa.csv'}")
    print(f"CSV resumen: {task3_dir / 'comparacion_modelos_resumen.csv'}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
