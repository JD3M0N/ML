from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

import task3_common


DEFAULT_OUTPUT_DIR = task3_common.PROJECT_ROOT / "outputs" / "task4"
DEFAULT_REPORT_PATH = task3_common.PROJECT_ROOT / "reports" / "task4_optimizacion_parametros.md"
PRIMARY_METRIC = "recall_condition_1"
SELECTION_METRIC = "medical_cost"
MODEL_SLUGS = ["logistic_regression", "svm", "knn"]
MODEL_NAMES = {
    "logistic_regression": "Regresion Logistica",
    "svm": "SVM RBF",
    "knn": "KNN",
}
ORIGINAL_FEATURES = (
    task3_common.CONTINUOUS_FEATURES
    + task3_common.CATEGORICAL_FEATURES
    + task3_common.BINARY_FEATURES
)


def add_task4_args(parser: argparse.ArgumentParser) -> None:
    task3_common.add_common_args(parser)
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR, n_repeats=10)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Numero de trabajos paralelos para GridSearchCV.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=30,
        help="Repeticiones para permutation_importance.",
    )


def medical_cost_scorer(fn_cost: float, fp_cost: float) -> Any:
    return make_scorer(
        task3_common.medical_cost,
        greater_is_better=False,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )


def make_scoring(fn_cost: float, fp_cost: float) -> dict[str, Any]:
    return {
        "recall_condition_1": "recall",
        "roc_auc": "roc_auc",
        "neg_medical_cost": medical_cost_scorer(fn_cost, fp_cost),
    }


def choose_best_index(cv_results: dict[str, Any]) -> int:
    results = pd.DataFrame(cv_results)
    ranked = results.sort_values(
        by=[
            "mean_test_neg_medical_cost",
            "mean_test_recall_condition_1",
            "mean_test_roc_auc",
        ],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return int(ranked.index[0])


def grid_search(
    estimator: Any,
    param_grid: list[dict[str, Any]] | dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    fn_cost: float,
    fp_cost: float,
    n_jobs: int,
) -> GridSearchCV:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_scoring(fn_cost, fp_cost),
        refit=choose_best_index,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search


def tidy_grid_results(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    keep_columns = [
        "rank_test_recall_condition_1",
        "mean_test_recall_condition_1",
        "std_test_recall_condition_1",
        "mean_test_neg_medical_cost",
        "std_test_neg_medical_cost",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_train_recall_condition_1",
        "mean_train_neg_medical_cost",
        "mean_train_roc_auc",
        "params",
    ]
    param_columns = [column for column in results.columns if column.startswith("param_")]
    tidy = results[param_columns + keep_columns].copy()
    tidy["mean_test_medical_cost"] = -tidy["mean_test_neg_medical_cost"]
    tidy["std_test_medical_cost"] = tidy["std_test_neg_medical_cost"]
    tidy["mean_train_medical_cost"] = -tidy["mean_train_neg_medical_cost"]
    return tidy.sort_values(
        by=[
            "mean_test_neg_medical_cost",
            "mean_test_recall_condition_1",
            "mean_test_roc_auc",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarize_best_cv(grid_results: pd.DataFrame) -> pd.DataFrame:
    best = grid_results.iloc[0]
    rows = [
        {
            "metric": "recall_condition_1",
            "mean": best["mean_test_recall_condition_1"],
            "std": best["std_test_recall_condition_1"],
        },
        {
            "metric": "medical_cost",
            "mean": best["mean_test_medical_cost"],
            "std": best["std_test_medical_cost"],
        },
        {
            "metric": "roc_auc",
            "mean": best["mean_test_roc_auc"],
            "std": best["std_test_roc_auc"],
        },
    ]
    return pd.DataFrame(rows)


def original_feature_from_transformed(feature_name: str) -> str:
    raw = feature_name.split("__", 1)[-1]
    for feature in ORIGINAL_FEATURES:
        if raw == feature or raw.startswith(f"{feature}_"):
            return feature
    return raw


def get_transformed_feature_names(model: Any) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def logistic_coefficient_importance(model: Any) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    transformed_names = get_transformed_feature_names(model)
    rows = []
    for name, coefficient in zip(transformed_names, coefficients):
        rows.append(
            {
                "transformed_feature": name,
                "feature": original_feature_from_transformed(name),
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
            }
        )
    detailed = pd.DataFrame(rows)
    return (
        detailed.groupby("feature", as_index=False)
        .agg(
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
            signed_coefficient_sum=("coefficient", "sum"),
        )
        .sort_values("mean_abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def permutation_feature_importance(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="recall",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    rows = []
    for index, feature in enumerate(x_test.columns):
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(importance.importances_mean[index]),
                "importance_std": float(importance.importances_std[index]),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_df = df.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"], color="#4C78A8")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Caida media de recall al permutar la variable")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_logistic_coefficients_plot(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_df = df.sort_values("mean_abs_coefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["mean_abs_coefficient"], color="#F58518")
    ax.set_title("Regresion Logistica - Magnitud media de coeficientes")
    ax.set_xlabel("Coeficiente absoluto medio")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_model_report(
    output_path: Path,
    model_name: str,
    best_params: dict[str, Any],
    cv_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    fn_cost: float,
    fp_cost: float,
) -> Path:
    recall = float(cv_summary.loc[cv_summary["metric"] == "recall_condition_1", "mean"].iloc[0])
    cost = float(cv_summary.loc[cv_summary["metric"] == "medical_cost", "mean"].iloc[0])
    roc_auc = float(cv_summary.loc[cv_summary["metric"] == "roc_auc", "mean"].iloc[0])
    test_row = test_metrics.iloc[0]
    top_features = feature_importance.head(5)
    feature_lines = "\n".join(
        f"- `{row.feature}`: importancia media {row.importance_mean:.4f}"
        for row in top_features.itertuples()
    )
    params_json = json.dumps(best_params, ensure_ascii=False, indent=2, default=str)
    content = f"""# Task 4 - {model_name}

## Objetivo

Este experimento optimiza hiperparametros para predecir `condition`, guiado por el coste medico normalizado con FN={fn_cost:g} y FP={fp_cost:g}. En el contexto medico del proyecto se prioriza detectar pacientes con `condition=1`, pero se evita seleccionar configuraciones degeneradas que maximizan recall a costa de muchos falsos positivos o mala generalizacion.

## Mejor configuracion

```json
{params_json}
```

## Validacion cruzada

- Recall condition=1: {recall:.4f}
- Coste medico normalizado: {cost:.4f}
- ROC-AUC: {roc_auc:.4f}

## Test final reservado

- Recall condition=1: {test_row["recall_condition_1"]:.4f}
- Falsos negativos: {int(test_row["false_negatives"])}
- Falsos positivos: {int(test_row["false_positives"])}
- Coste medico normalizado: {test_row["medical_cost"]:.4f}
- ROC-AUC: {test_row["roc_auc"]:.4f}

## Variables influyentes

La importancia se calculo con `permutation_importance` sobre el pipeline completo y usando recall como scoring. Las variables con mayor caida de recall al permutarse son:

{feature_lines}
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def run_optimization_experiment(
    estimator: Any,
    param_grid: list[dict[str, Any]] | dict[str, Any],
    experiment_name: str,
    experiment_slug: str,
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = task3_common.DEFAULT_TEST_SIZE,
    n_splits: int = task3_common.DEFAULT_N_SPLITS,
    n_repeats: int = 10,
    fn_cost: float = task3_common.DEFAULT_FN_COST,
    fp_cost: float = task3_common.DEFAULT_FP_COST,
    n_jobs: int = 1,
    permutation_repeats: int = 30,
) -> dict[str, Path | pd.DataFrame | Any]:
    selected_output = (output_dir or DEFAULT_OUTPUT_DIR).resolve() / experiment_slug
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = task3_common.load_dataset(csv_path)
    x, y = task3_common.split_features_target(df)
    x_train, x_test, y_train, y_test = task3_common.stratified_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=task3_common.RANDOM_STATE,
    )

    search = grid_search(
        estimator=estimator,
        param_grid=param_grid,
        x_train=x_train,
        y_train=y_train,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=task3_common.RANDOM_STATE,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        n_jobs=n_jobs,
    )
    best_model = search.best_estimator_
    grid_results = tidy_grid_results(search)
    cv_summary = summarize_best_cv(grid_results)
    _, test_metrics, y_score = task3_common.fit_and_evaluate_test(
        best_model,
        x_train,
        y_train,
        x_test,
        y_test,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    feature_importance = permutation_feature_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=permutation_repeats,
        random_state=task3_common.RANDOM_STATE,
    )

    best_params = {key: value for key, value in search.best_params_.items()}
    pd.DataFrame([best_params]).to_csv(selected_output / "mejores_parametros.csv", index=False)
    grid_results.to_csv(selected_output / "grid_search_resultados.csv", index=False)
    cv_summary.to_csv(selected_output / "cv_metricas_resumen.csv", index=False)
    test_metrics.to_csv(selected_output / "test_metricas.csv", index=False)
    feature_importance.to_csv(selected_output / "feature_importance.csv", index=False)
    pd.crosstab(
        pd.Series(y_test.to_numpy(), name="real"),
        pd.Series(y_pred, name="predicho"),
    ).to_csv(selected_output / "test_matriz_confusion.csv")

    task3_common.save_confusion_matrix_plot(
        y_test,
        y_pred,
        selected_output / "test_matriz_confusion.png",
        f"{experiment_name} optimizado - Matriz de confusion en test",
    )
    task3_common.save_roc_curve_plot(
        y_test,
        y_score,
        selected_output / "test_roc_curve.png",
        f"{experiment_name} optimizado - Curva ROC en test",
    )
    save_feature_importance_plot(
        feature_importance,
        selected_output / "feature_importance.png",
        f"{experiment_name} optimizado - Influencia por permutacion",
    )

    if experiment_slug == "logistic_regression":
        coefficients = logistic_coefficient_importance(best_model)
        coefficients.to_csv(selected_output / "logistic_coefficients.csv", index=False)
        save_logistic_coefficients_plot(coefficients, selected_output / "logistic_coefficients.png")

    report_path = write_model_report(
        selected_output / "reporte.md",
        experiment_name,
        best_params,
        cv_summary,
        test_metrics,
        feature_importance,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )

    print(f"CSV usado: {selected_csv}")
    print(f"Resultados guardados en: {selected_output}")
    print("Mejores parametros:")
    print(best_params)
    print(cv_summary.to_string(index=False))
    print("Metricas de test final:")
    print(test_metrics.to_string(index=False))
    print(f"Reporte guardado en: {report_path}")

    return {
        "output_dir": selected_output,
        "grid_results": grid_results,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "best_estimator": best_model,
        "report_path": report_path,
    }
