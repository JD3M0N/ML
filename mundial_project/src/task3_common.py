from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "foot_clean.csv"
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"
TARGET = "condition"
CONTINUOUS_FEATURES = ["feature_5", "feature_6", "feature_10", "feature_11", "feature_13"]
CATEGORICAL_FEATURES = ["feature_2", "feature_3", "feature_8", "feature_9", "feature_12"]
BINARY_FEATURES = ["feature_1", "feature_4", "feature_7"]
RANDOM_STATE = 42
POSITIVE_LABEL = 1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 30
DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0


def sklearn_is_available() -> bool:
    return importlib.util.find_spec("sklearn") is not None


def build_preprocessor() -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV de entrada. Si se omite usa data/processed/foot_clean.csv o data/foot.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta base donde se guardan los resultados de Task 3.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Proporcion estratificada reservada como test final.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Numero de folds estratificados para validacion cruzada.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Numero de repeticiones del K-fold estratificado.",
    )
    parser.add_argument(
        "--fn-cost",
        type=float,
        default=DEFAULT_FN_COST,
        help="Costo medico asignado a cada falso negativo.",
    )
    parser.add_argument(
        "--fp-cost",
        type=float,
        default=DEFAULT_FP_COST,
        help="Costo medico asignado a cada falso positivo.",
    )


def load_dataset(csv_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    selected_path = csv_path or (DEFAULT_CLEAN_CSV if DEFAULT_CLEAN_CSV.exists() else DEFAULT_RAW_CSV)
    df = pd.read_csv(selected_path)
    if TARGET not in df.columns:
        raise ValueError(f"No existe la columna objetivo {TARGET!r}.")
    return df, selected_path


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    if sorted(y.dropna().unique().tolist()) != [0, 1]:
        raise ValueError("Task 3 espera una variable objetivo binaria con valores 0 y 1.")
    return x, y


def stratified_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(x, y))
    return (
        x.iloc[train_index].copy(),
        x.iloc[test_index].copy(),
        y.iloc[train_index].copy(),
        y.iloc[test_index].copy(),
    )


def prediction_scores(model: BaseEstimator, x: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return None


def medical_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> float:
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fn_cost * fn + fp_cost * fp) / (tn + fp + fn + tp))


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: np.ndarray | None,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_condition_1": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_condition_1": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_condition_1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "medical_cost": medical_cost(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_positives": float(tp),
        }
    )
    return metrics


def repeated_stratified_cv(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = RANDOM_STATE,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> pd.DataFrame:
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []
    for fold_id, (fit_index, val_index) in enumerate(cv.split(x_train, y_train), start=1):
        model = clone(estimator)
        x_fit = x_train.iloc[fit_index]
        y_fit = y_train.iloc[fit_index]
        x_val = x_train.iloc[val_index]
        y_val = y_train.iloc[val_index]

        model.fit(x_fit, y_fit)
        y_pred = model.predict(x_val)
        y_score = prediction_scores(model, x_val)
        metrics = classification_metrics(y_val, y_pred, y_score, fn_cost=fn_cost, fp_cost=fp_cost)
        metrics.update(
            {
                "fold": fold_id,
                "repeat": int((fold_id - 1) // n_splits + 1),
                "fold_in_repeat": int((fold_id - 1) % n_splits + 1),
                "train_condition_1_pct": float(y_fit.mean() * 100),
                "validation_condition_1_pct": float(y_val.mean() * 100),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_cv_metrics(cv_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "precision_condition_1",
        "recall_condition_1",
        "f1_condition_1",
        "roc_auc",
        "medical_cost",
        "false_negatives",
        "false_positives",
    ]
    rows = []
    for metric in metric_columns:
        rows.append(
            {
                "metric": metric,
                "mean": cv_results[metric].mean(),
                "std": cv_results[metric].std(ddof=1),
                "min": cv_results[metric].min(),
                "max": cv_results[metric].max(),
            }
        )
    return pd.DataFrame(rows)


def fit_and_evaluate_test(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> tuple[BaseEstimator, pd.DataFrame, np.ndarray | None]:
    from sklearn.base import clone

    model = clone(estimator)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_score = prediction_scores(model, x_test)
    metrics = classification_metrics(y_test, y_pred, y_score, fn_cost=fn_cost, fp_cost=fp_cost)
    return model, pd.DataFrame([metrics]), y_score


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["condition=0", "condition=1"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(
    y_true: pd.Series,
    y_score: np.ndarray | None,
    output_path: Path,
    title: str,
) -> None:
    if y_score is None:
        return
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, color="#E45756")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_learning_curve_plot(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Path,
    title: str,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = RANDOM_STATE,
    ) -> tuple[pd.DataFrame, str]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from sklearn.model_selection import RepeatedStratifiedKFold, learning_curve

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        x_train,
        y_train,
        cv=cv,
        scoring="recall",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1,
    )
    curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_recall_mean": train_scores.mean(axis=1),
            "train_recall_std": train_scores.std(axis=1, ddof=1),
            "validation_recall_mean": val_scores.mean(axis=1),
            "validation_recall_std": val_scores.std(axis=1, ddof=1),
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["train_size"], curve_df["train_recall_mean"], marker="o", label="Train recall")
    ax.fill_between(
        curve_df["train_size"],
        curve_df["train_recall_mean"] - curve_df["train_recall_std"],
        curve_df["train_recall_mean"] + curve_df["train_recall_std"],
        alpha=0.15,
    )
    ax.plot(
        curve_df["train_size"],
        curve_df["validation_recall_mean"],
        marker="o",
        label="Validation recall",
        color="#E45756",
    )
    ax.fill_between(
        curve_df["train_size"],
        curve_df["validation_recall_mean"] - curve_df["validation_recall_std"],
        curve_df["validation_recall_mean"] + curve_df["validation_recall_std"],
        alpha=0.15,
        color="#E45756",
    )
    ax.set_title(title)
    ax.set_xlabel("Tamano de entrenamiento")
    ax.set_ylabel("Recall condition=1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    # Preparar diagnostico automatizado para anadirlo bajo la figura
    try:
        last_train = float(curve_df["train_recall_mean"].iloc[-1])
        last_val = float(curve_df["validation_recall_mean"].iloc[-1])
        gap = last_train - last_val
        trend = "estable"
        if curve_df["validation_recall_mean"].is_monotonic_increasing:
            trend = "mejorando con mas datos"
        elif curve_df["validation_recall_mean"].is_monotonic_decreasing:
            trend = "empeorando con mas datos"

        if last_train >= 0.8 and gap > 0.05:
            diagnosis = "Posible sobreajuste: el recall en entrenamiento es significativamente mayor que en validacion."
        elif last_train < 0.6 and last_val < 0.6:
            diagnosis = "Posible infraajuste: tanto entrenamiento como validacion muestran recall bajo."
        elif gap <= 0.05 and last_val >= 0.6:
            diagnosis = "Buen ajuste: poca diferencia entre entrenamiento y validacion y recall razonable en validacion."
        else:
            diagnosis = "Comportamiento mixto: revisar curvas y varianza entre folds para mas contexto."

        diag_text = (
            f"Diagnostico automatizado:\nTrain recall (ult.): {last_train:.3f}  |  Val recall (ult.): {last_val:.3f}  |  Gap: {gap:.3f}  |  Tendencia: {trend}\nInterpretacion: {diagnosis}"
        )
    except Exception:
        diag_text = "No fue posible generar un diagnostico automatico de la curva de aprendizaje."

    # Reservar espacio inferior para el texto y añadirlo
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.01, 0.02, diag_text, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return curve_df, diag_text


def make_supervised_pipeline(classifier: BaseEstimator) -> BaseEstimator:
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", classifier),
        ]
    )


def run_supervised_experiment(
    estimator: BaseEstimator,
    experiment_name: str,
    experiment_slug: str,
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> dict[str, Path | pd.DataFrame]:
    selected_output = (output_dir or DEFAULT_OUTPUT_DIR).resolve() / experiment_slug
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = load_dataset(csv_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    cv_results = repeated_stratified_cv(
        estimator,
        x_train,
        y_train,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    cv_summary = summarize_cv_metrics(cv_results)
    fitted_model, test_metrics, y_score = fit_and_evaluate_test(
        estimator,
        x_train,
        y_train,
        x_test,
        y_test,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    y_pred = fitted_model.predict(x_test)

    cv_results.to_csv(selected_output / "cv_metricas_por_fold.csv", index=False)
    cv_summary.to_csv(selected_output / "cv_metricas_resumen.csv", index=False)
    test_metrics.to_csv(selected_output / "test_metricas.csv", index=False)
    pd.crosstab(
        pd.Series(y_test.to_numpy(), name="real"),
        pd.Series(y_pred, name="predicho"),
    ).to_csv(selected_output / "test_matriz_confusion.csv")

    save_confusion_matrix_plot(
        y_test,
        y_pred,
        selected_output / "test_matriz_confusion.png",
        f"{experiment_name} - Matriz de confusion en test",
    )
    save_roc_curve_plot(
        y_test,
        y_score,
        selected_output / "test_roc_curve.png",
        f"{experiment_name} - Curva ROC en test",
    )
    learning_curve_df, learning_curve_diag = save_learning_curve_plot(
        estimator,
        x_train,
        y_train,
        selected_output / "learning_curve_recall.png",
        f"{experiment_name} - Curva de aprendizaje",
        n_splits=n_splits,
        random_state=RANDOM_STATE,
    )
    learning_curve_df.to_csv(selected_output / "learning_curve_recall.csv", index=False)
    # Guardar diagnostico en un archivo de texto para referencia
    (selected_output / "learning_curve_recall_diagnostico.txt").write_text(learning_curve_diag, encoding="utf-8")

    report_path = write_markdown_report(
        selected_output / "reporte.md",
        experiment_name,
        selected_csv.resolve(),
        cv_summary,
        test_metrics,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        n_splits=n_splits,
        n_repeats=n_repeats,
        learning_curve_diagnosis=learning_curve_diag,
    )

    print(f"CSV usado: {selected_csv}")
    print(f"Resultados guardados en: {selected_output}")
    print(cv_summary.to_string(index=False))
    print("Metricas de test final:")
    print(test_metrics.to_string(index=False))
    print(f"Reporte guardado en: {report_path}")

    return {
        "output_dir": selected_output,
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "report_path": report_path,
    }


def write_markdown_report(
    output_path: Path,
    experiment_name: str,
    csv_path: Path,
    cv_summary: pd.DataFrame,
    test_metrics: pd.DataFrame,
    fn_cost: float,
    fp_cost: float,
    n_splits: int,
    n_repeats: int,
    learning_curve_diagnosis: str | None = None,
) -> Path:
    def metric_value(df: pd.DataFrame, metric: str, column: str = "mean") -> float:
        return float(df.loc[df["metric"] == metric, column].iloc[0])

    test_row = test_metrics.iloc[0]
    content = f"""# Task 3 - {experiment_name}

## Objetivo

Este experimento entrena un modelo supervisado para predecir `condition` y evalua su utilidad en un contexto medico, priorizando la reduccion de falsos negativos mediante `recall_condition_1`.

## Datos y validacion

- CSV usado: `{csv_path.relative_to(PROJECT_ROOT) if csv_path.is_relative_to(PROJECT_ROOT) else csv_path}`.
- Test final: particion estratificada del 20%, reservada hasta el cierre del experimento.
- Validacion interna: `RepeatedStratifiedKFold` con {n_splits} folds y {n_repeats} repeticiones.
- Preprocesamiento: estandarizacion de variables continuas, one-hot encoding de discretas y paso directo de binarias dentro del `Pipeline`.
- Coste medico: FN={fn_cost:g}, FP={fp_cost:g}, normalizado por numero de pacientes.

## Resultados de validacion cruzada

| Metrica | Media | Desviacion |
|---|---:|---:|
| Accuracy | {metric_value(cv_summary, "accuracy"):.4f} | {metric_value(cv_summary, "accuracy", "std"):.4f} |
| Precision condition=1 | {metric_value(cv_summary, "precision_condition_1"):.4f} | {metric_value(cv_summary, "precision_condition_1", "std"):.4f} |
| Recall condition=1 | {metric_value(cv_summary, "recall_condition_1"):.4f} | {metric_value(cv_summary, "recall_condition_1", "std"):.4f} |
| F1 condition=1 | {metric_value(cv_summary, "f1_condition_1"):.4f} | {metric_value(cv_summary, "f1_condition_1", "std"):.4f} |
| ROC-AUC | {metric_value(cv_summary, "roc_auc"):.4f} | {metric_value(cv_summary, "roc_auc", "std"):.4f} |
| Coste medico | {metric_value(cv_summary, "medical_cost"):.4f} | {metric_value(cv_summary, "medical_cost", "std"):.4f} |

## Resultados en test final

- Accuracy: {test_row["accuracy"]:.4f}
- Precision condition=1: {test_row["precision_condition_1"]:.4f}
- Recall condition=1: {test_row["recall_condition_1"]:.4f}
- F1 condition=1: {test_row["f1_condition_1"]:.4f}
- ROC-AUC: {test_row["roc_auc"]:.4f}
- Falsos negativos: {int(test_row["false_negatives"])}
- Falsos positivos: {int(test_row["false_positives"])}
- Coste medico normalizado: {test_row["medical_cost"]:.4f}

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
"""
    if learning_curve_diagnosis:
        content = content + f"\n## Diagnostico de la curva de aprendizaje\n\n{learning_curve_diagnosis}\n"
        content = content + (
            "\n## Posibles causas de bajo accuracy en redes neuronales y recomendaciones\n\n"
            "- Desequilibrio de clases: si la clase positiva es rara, la accuracy global puede ser alta aunque el modelo falle en detectar la condicion.\n"
            "- Infraajuste: arquitectura o capacidad insuficiente, learning rate inapropiado, o pocas iteraciones.\n"
            "- Sobreajuste: modelo demasiado complejo sin regularizacion adecuada; gap grande entre entrenamiento y validacion.\n"
            "- Preprocesado: features irrelevantes o mal escaladas afectan la convergencia; revisar estandarizacion y encoding.\n"
            "- Hiperparametros: `alpha` (regularizacion), `learning_rate_init`, `hidden_layer_sizes` y `max_iter` influyen fuertemente.\n"
            "- Early stopping: si esta activado puede detener antes de convergencia si la validacion es ruidosa.\n"
            "\nRecomendaciones:\n- Revisar balance de clases y usar `class_weight='balanced'` o re-muestreo si procede.\n- Probar aumentar `max_iter`, ajustar `learning_rate_init` y `alpha`, y explorar diferentes `hidden_layer_sizes`.\n- Usar validacion cruzada estable y observar curvas de aprendizaje (ya generadas) para decidir si mas datos ayudarian.\n- Priorizar metrics de interes (recall para la clase positiva) en lugar de accuracy.\n"
        )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostico rapido de la configuracion comun para Task 3."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV a revisar. Si se omite usa data/processed/foot_clean.csv o data/foot.csv.",
    )
    args = parser.parse_args()

    print("Task 3 common: modulo de utilidades cargado correctamente.")
    print(f"Python puede importar scikit-learn: {'si' if sklearn_is_available() else 'no'}")

    df, csv_path = load_dataset(args.csv)
    x, y = split_features_target(df)
    print(f"CSV usado: {csv_path}")
    print(f"Filas: {len(df)}")
    print(f"Features: {x.shape[1]}")
    print("Balance de condition:")
    print(y.value_counts().sort_index().to_string())

    if not sklearn_is_available():
        print()
        print("Aviso: este entorno no tiene scikit-learn instalado.")
        print("Este archivo ya puede ejecutarse como diagnostico, pero los experimentos supervisados")
        print("necesitan scikit-learn. Para correr el modelo usa un entorno con requirements.txt instalado")
        print("y ejecuta: python mundial_project\\src\\task3_logistic_regression.py")


if __name__ == "__main__":
    main()
