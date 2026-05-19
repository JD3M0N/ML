from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from task3_common import (
    DEFAULT_OUTPUT_DIR,
    add_common_args,
    fit_and_evaluate_test,
    load_dataset,
    make_supervised_pipeline,
    repeated_stratified_cv,
    save_confusion_matrix_plot,
    save_learning_curve_plot,
    save_roc_curve_plot,
    split_features_target,
    stratified_train_test_split,
    summarize_cv_metrics,
    write_markdown_report,
)


EXPERIMENT_NAME = "Regresion Logistica"
EXPERIMENT_SLUG = "logistic_regression"


def build_estimator() -> object:
    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=None,
        random_state=42,
    )
    return make_supervised_pipeline(classifier)


def run(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_repeats: int = 30,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict[str, Path | pd.DataFrame]:
    selected_output = (output_dir or DEFAULT_OUTPUT_DIR).resolve() / EXPERIMENT_SLUG
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = load_dataset(csv_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
    )

    estimator = build_estimator()
    cv_results = repeated_stratified_cv(
        estimator,
        x_train,
        y_train,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
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

    cv_results_path = selected_output / "cv_metricas_por_fold.csv"
    cv_summary_path = selected_output / "cv_metricas_resumen.csv"
    test_metrics_path = selected_output / "test_metricas.csv"
    confusion_csv_path = selected_output / "test_matriz_confusion.csv"
    learning_curve_csv_path = selected_output / "learning_curve_recall.csv"

    cv_results.to_csv(cv_results_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    test_metrics.to_csv(test_metrics_path, index=False)
    pd.crosstab(
        pd.Series(y_test.to_numpy(), name="real"),
        pd.Series(y_pred, name="predicho"),
    ).to_csv(confusion_csv_path)

    save_confusion_matrix_plot(
        y_test,
        y_pred,
        selected_output / "test_matriz_confusion.png",
        "Regresion Logistica - Matriz de confusion en test",
    )
    save_roc_curve_plot(
        y_test,
        y_score,
        selected_output / "test_roc_curve.png",
        "Regresion Logistica - Curva ROC en test",
    )
    learning_curve_df = save_learning_curve_plot(
        estimator,
        x_train,
        y_train,
        selected_output / "learning_curve_recall.png",
        "Regresion Logistica - Curva de aprendizaje",
        n_splits=n_splits,
        random_state=42,
    )
    learning_curve_df.to_csv(learning_curve_csv_path, index=False)

    report_path = write_markdown_report(
        selected_output / "reporte.md",
        EXPERIMENT_NAME,
        selected_csv.resolve(),
        cv_summary,
        test_metrics,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        n_splits=n_splits,
        n_repeats=n_repeats,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: experimento de regresion logistica.")
    add_common_args(parser)
    args = parser.parse_args()
    run(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
    )


if __name__ == "__main__":
    main()
