from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

import task4_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolida resultados de optimizacion Task 4.")
    parser.add_argument(
        "--task4-dir",
        type=Path,
        default=task4_common.DEFAULT_OUTPUT_DIR,
        help="Carpeta con subdirectorios de modelos optimizados.",
    )
    parser.add_argument(
        "--task3-dir",
        type=Path,
        default=task4_common.task3_common.DEFAULT_OUTPUT_DIR,
        help="Carpeta con resultados base de Task 3.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=task4_common.DEFAULT_REPORT_PATH,
        help="Ruta del reporte markdown consolidado.",
    )
    return parser.parse_args()


def read_best_params(model_dir: Path) -> dict[str, object]:
    params_path = model_dir / "mejores_parametros.csv"
    if not params_path.exists():
        return {}
    row = pd.read_csv(params_path).iloc[0].to_dict()
    return {key: value for key, value in row.items() if pd.notna(value)}


def read_cv_summary(model_dir: Path) -> dict[str, float]:
    df = pd.read_csv(model_dir / "cv_metricas_resumen.csv")
    row = {}
    for _, metric_row in df.iterrows():
        metric = metric_row["metric"]
        row[f"cv_{metric}_mean"] = metric_row["mean"]
        row[f"cv_{metric}_std"] = metric_row["std"]
    return row


def read_test_metrics(model_dir: Path) -> dict[str, float]:
    df = pd.read_csv(model_dir / "test_metricas.csv")
    return {f"test_{key}": value for key, value in df.iloc[0].to_dict().items()}


def build_comparison(task4_dir: Path, task3_dir: Path) -> pd.DataFrame:
    rows = []
    for slug in task4_common.MODEL_SLUGS:
        model_dir = task4_dir / slug
        if not model_dir.exists():
            continue
        row = {
            "slug": slug,
            "modelo": task4_common.MODEL_NAMES[slug],
            "best_params": read_best_params(model_dir),
            **read_cv_summary(model_dir),
            **read_test_metrics(model_dir),
        }
        baseline_path = task3_dir / slug / "cv_metricas_resumen.csv"
        baseline_test_path = task3_dir / slug / "test_metricas.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            recall = baseline.loc[baseline["metric"] == "recall_condition_1", "mean"].iloc[0]
            cost = baseline.loc[baseline["metric"] == "medical_cost", "mean"].iloc[0]
            row["task3_cv_recall_condition_1_mean"] = recall
            row["task3_cv_medical_cost_mean"] = cost
            row["delta_cv_recall"] = row["cv_recall_condition_1_mean"] - recall
            row["delta_cv_medical_cost"] = row["cv_medical_cost_mean"] - cost
        if baseline_test_path.exists():
            baseline_test = pd.read_csv(baseline_test_path).iloc[0]
            row["task3_test_recall_condition_1"] = baseline_test["recall_condition_1"]
            row["task3_test_medical_cost"] = baseline_test["medical_cost"]
            row["delta_test_recall"] = row["test_recall_condition_1"] - baseline_test["recall_condition_1"]
            row["delta_test_medical_cost"] = row["test_medical_cost"] - baseline_test["medical_cost"]
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError(f"No se encontraron resultados de Task 4 en {task4_dir}.")
    return comparison.sort_values(
        by=["cv_recall_condition_1_mean", "cv_medical_cost_mean", "test_false_negatives"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_global_feature_ranking(task4_dir: Path) -> pd.DataFrame:
    frames = []
    for slug in task4_common.MODEL_SLUGS:
        path = task4_dir / slug / "feature_importance.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        max_importance = df["importance_mean"].clip(lower=0).max()
        normalizer = max_importance if max_importance > 0 else 1.0
        df["importance_normalized"] = df["importance_mean"].clip(lower=0) / normalizer
        df["modelo"] = task4_common.MODEL_NAMES[slug]
        frames.append(df)
    if not frames:
        raise ValueError("No se encontraron archivos feature_importance.csv.")
    features = pd.concat(frames, ignore_index=True)
    return (
        features.groupby("feature", as_index=False)
        .agg(
            importance_normalized_mean=("importance_normalized", "mean"),
            importance_mean_avg=("importance_mean", "mean"),
            models_with_positive_importance=("importance_mean", lambda values: int((values > 0).sum())),
        )
        .sort_values(
            by=["importance_normalized_mean", "models_with_positive_importance"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    display_df = df[columns].copy()
    for column in display_df.columns:
        if display_df[column].dtype == "object":
            display_df[column] = display_df[column].map(lambda value: str(value))
        else:
            display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display_df.columns]
    rows = [[str(value) for value in row] for row in display_df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compact_ranking(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "modelo",
        "cv_recall_condition_1_mean",
        "cv_medical_cost_mean",
        "cv_roc_auc_mean",
        "test_recall_condition_1",
        "test_false_negatives",
        "test_false_positives",
        "test_medical_cost",
        "delta_cv_recall",
        "delta_cv_medical_cost",
    ]
    return comparison[[column for column in columns if column in comparison.columns]]


def params_to_text(params: object) -> str:
    if isinstance(params, str):
        try:
            params = ast.literal_eval(params)
        except (SyntaxError, ValueError):
            return params
    if not isinstance(params, dict):
        return str(params)
    return ", ".join(f"`{key}`={value}" for key, value in params.items())


def write_report(comparison: pd.DataFrame, feature_ranking: pd.DataFrame, report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = comparison.iloc[0]
    best_test = comparison.sort_values("test_medical_cost", ascending=True).iloc[0]
    top_features = feature_ranking.head(5)
    top_feature_lines = "\n".join(
        f"- `{row.feature}`: importancia normalizada media {row.importance_normalized_mean:.4f}"
        for row in top_features.itertuples()
    )
    params_lines = "\n".join(
        f"- **{row.modelo}**: {params_to_text(row.best_params)}"
        for row in comparison.itertuples()
    )
    content = f"""# Task 4 - Optimizacion de Parametros

## Estrategia

Se optimizaron los tres modelos mas prometedores de Task 3: Regresion Logistica, SVM RBF y KNN. La busqueda se guio por `recall_condition_1`, porque el falso negativo es el error mas costoso en este contexto medico. Como desempate se uso menor coste medico normalizado y luego mayor ROC-AUC.

## Comparacion de modelos optimizados

{markdown_table(compact_ranking(comparison), compact_ranking(comparison).columns.tolist())}

## Mejores configuraciones

{params_lines}

## Lectura medica

El modelo recomendado por validacion cruzada es **{best["modelo"]}**, con recall medio {best["cv_recall_condition_1_mean"]:.4f} y coste medico medio {best["cv_medical_cost_mean"]:.4f}. En test final, el menor coste medico lo obtiene **{best_test["modelo"]}**, con {int(best_test["test_false_negatives"])} falsos negativos y {int(best_test["test_false_positives"])} falsos positivos.

La seleccion principal se mantiene en validacion cruzada, no en accuracy ni en una sola particion de test. Un aumento de falsos positivos puede aceptarse si reduce falsos negativos, pero debe revisarse junto al coste medico.

## Variables mas influyentes

La influencia se calculo con `permutation_importance` usando recall como scoring sobre el pipeline completo. Las variables con mayor influencia global fueron:

{top_feature_lines}
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    task4_dir = args.task4_dir.resolve()
    task3_dir = args.task3_dir.resolve()
    task4_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(task4_dir, task3_dir)
    feature_ranking = build_global_feature_ranking(task4_dir)
    ranking = compact_ranking(comparison)

    comparison.to_csv(task4_dir / "comparacion_optimizacion.csv", index=False)
    ranking.to_csv(task4_dir / "ranking_modelos_optimizados.csv", index=False)
    feature_ranking.to_csv(task4_dir / "variables_influyentes_global.csv", index=False)
    report_path = write_report(comparison, feature_ranking, args.report_path.resolve())

    print("Comparacion de modelos optimizados:")
    print(ranking.to_string(index=False))
    print("Variables mas influyentes:")
    print(feature_ranking.head(10).to_string(index=False))
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
