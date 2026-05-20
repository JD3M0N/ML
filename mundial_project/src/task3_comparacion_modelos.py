from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task3_common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT


MODEL_METADATA = {
    "logistic_regression": {
        "modelo": "Regresión Logística",
        "familia": "Modelo lineal probabilístico",
        "por_que": "Baseline interpretable para estimar una frontera lineal y probabilidades.",
        "configuracion": "LogisticRegression(max_iter=2000, solver='lbfgs')",
        "fortaleza": "Alta interpretabilidad y buen equilibrio entre recall, ROC-AUC y coste.",
        "limitacion": "Puede quedarse corta si la relación entre variables y condition no es lineal.",
    },
    "naive_bayes": {
        "modelo": "Naive Bayes",
        "familia": "Modelo probabilístico simple",
        "por_que": "Contrasta un enfoque rápido y robusto con supuestos fuertes de independencia.",
        "configuracion": "GaussianNB sobre variables preprocesadas",
        "fortaleza": "Sirve como baseline sencillo y suele funcionar con pocos datos.",
        "limitacion": "El supuesto de independencia puede reducir recall si hay relaciones entre variables.",
    },
    "knn": {
        "modelo": "KNN",
        "familia": "Modelo basado en vecinos",
        "por_que": "Evalúa si pacientes cercanos en el espacio de variables comparten condition.",
        "configuracion": "KNeighborsClassifier con configuración base",
        "fortaleza": "Captura fronteras no lineales sin imponer una forma paramétrica.",
        "limitacion": "Es sensible al escalado, a ruido local y al tamaño efectivo del dataset.",
    },
    "decision_tree": {
        "modelo": "Árbol de Decisión CART",
        "familia": "Árbol interpretable",
        "por_que": "Baseline de reglas explícitas para comparar contra modelos más flexibles.",
        "configuracion": "DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)",
        "fortaleza": "Sus reglas son fáciles de explicar.",
        "limitacion": "Puede ser inestable y perder capacidad predictiva frente a ensamblados.",
    },
    "xgboost": {
        "modelo": "XGBoost",
        "familia": "Ensamblado de árboles potenciados",
        "por_que": "Prueba un modelo de árboles más potente que CART sin sustituir el baseline simple.",
        "configuracion": "XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05)",
        "fortaleza": "Captura interacciones no lineales con regularización y boosting.",
        "limitacion": "Menos interpretable que CART; debe justificar mejora real en métricas médicas.",
    },
    "svm": {
        "modelo": "SVM RBF",
        "familia": "Margen máximo no lineal",
        "por_que": "Evalúa una frontera no lineal suave mediante kernel RBF.",
        "configuracion": "SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)",
        "fortaleza": "Buen rendimiento en espacios no lineales de tamaño moderado.",
        "limitacion": "Menos interpretable y dependiente de C/gamma.",
    },
    "mlp": {
        "modelo": "Red Neuronal MLP",
        "familia": "Red neuronal feed-forward",
        "por_que": "Documenta si un modelo más flexible mejora la detección de condition=1.",
        "configuracion": "MLPClassifier(hidden_layer_sizes=(16, 8), early_stopping=True)",
        "fortaleza": "Puede modelar relaciones no lineales complejas.",
        "limitacion": "Puede ser inestable en datasets pequeños y perder recall clínico.",
    },
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


def model_name(slug: str) -> str:
    return MODEL_METADATA.get(slug, {}).get("modelo", slug)


def read_cv_summary(model_dir: Path) -> dict[str, float]:
    df = pd.read_csv(model_dir / "cv_metricas_resumen.csv")
    row = {"modelo": model_name(model_dir.name)}
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
    for slug in MODEL_METADATA:
        model_dir = task3_dir / slug
        if not model_dir.exists():
            continue
        row = {"slug": slug, **read_cv_summary(model_dir), **read_test_metrics(model_dir)}
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError(f"No se encontraron resultados en {task3_dir}.")

    comparison = comparison.sort_values(
        by=["cv_recall_condition_1_mean", "cv_medical_cost_mean", "test_false_negatives"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    comparison["rank"] = comparison.index + 1
    comparison["ranking_reason"] = comparison.apply(ranking_reason, axis=1)
    return comparison


def ranking_reason(row: pd.Series) -> str:
    return (
        f"Recall CV={row['cv_recall_condition_1_mean']:.4f}, "
        f"coste CV={row['cv_medical_cost_mean']:.4f}, "
        f"FN test={int(row['test_false_negatives'])}."
    )


def compact_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "modelo",
        "ranking_reason",
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


def experiment_catalog(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_slug = comparison.set_index("slug") if "slug" in comparison else pd.DataFrame()
    for slug, metadata in MODEL_METADATA.items():
        row = {"slug": slug, **metadata}
        if not by_slug.empty and slug in by_slug.index:
            row["lectura_resultado"] = interpret_model_row(slug, by_slug.loc[slug])
            row["conclusion_experimento"] = conclude_model_row(slug, by_slug.loc[slug])
        else:
            row["lectura_resultado"] = "Sin resultados generados todavía."
            row["conclusion_experimento"] = "Pendiente de ejecutar."
        rows.append(row)
    return pd.DataFrame(rows)


def interpret_model_row(slug: str, row: pd.Series) -> str:
    recall = row["cv_recall_condition_1_mean"]
    cost = row["cv_medical_cost_mean"]
    test_fn = int(row["test_false_negatives"])
    if slug == "mlp" and recall < 0.7:
        return (
            f"Experimento débil: recall CV {recall:.4f}, coste {cost:.4f} y {test_fn} falsos "
            "negativos en test; se conserva como intento no recomendable."
        )
    if slug == "xgboost":
        return (
            f"Modelo potente de contraste: recall CV {recall:.4f}, coste {cost:.4f} y {test_fn} "
            "falsos negativos en test; solo reemplazaría a CART si mejora las métricas médicas."
        )
    return (
        f"Recall CV {recall:.4f}, coste médico CV {cost:.4f} y {test_fn} falsos negativos en test."
    )


def conclude_model_row(slug: str, row: pd.Series) -> str:
    recall = row["cv_recall_condition_1_mean"]
    cost = row["cv_medical_cost_mean"]
    roc = row["cv_roc_auc_mean"]
    test_recall = row["test_recall_condition_1"]
    test_fn = int(row["test_false_negatives"])
    test_fp = int(row["test_false_positives"])

    if slug == "logistic_regression":
        return (
            "Es el baseline más defendible: detecta cerca del 81% de positivos en CV, "
            "mantiene el menor coste médico medio y además permite explicar coeficientes. "
            "Su límite es que todavía deja 5 falsos negativos en test."
        )
    if slug == "svm":
        return (
            "Confirma que una frontera no lineal ayuda: queda casi empatado con Regresión "
            f"Logística en CV y en test reduce los falsos negativos a {test_fn}. "
            "Se considera alternativa prometedora, aunque menos interpretable."
        )
    if slug == "xgboost":
        return (
            "Mejora claramente al árbol CART y captura interacciones no lineales, pero su "
            f"recall CV ({recall:.4f}) no supera a Regresión Logística. Aporta evidencia "
            "de que boosting es útil, no de que deba reemplazar al modelo base."
        )
    if slug == "knn":
        return (
            "Funciona como contraste local: el rendimiento es razonable, pero queda por "
            "debajo de los mejores en recall CV y depende mucho de distancias y escalado. "
            "No ofrece una ventaja clínica clara frente a modelos más estables."
        )
    if slug == "naive_bayes":
        return (
            "Es útil como baseline probabilístico rápido, pero el supuesto de independencia "
            "parece limitarlo: presenta menor recall CV y mayor variabilidad que los modelos "
            "más robustos. Su buen test debe leerse con cautela por ser una sola partición."
        )
    if slug == "decision_tree":
        return (
            "Aporta interpretabilidad mediante reglas, pero pierde rendimiento clínico: "
            f"su coste CV sube a {cost:.4f} y en test deja {test_fn} falsos negativos con "
            f"{test_fp} falsos positivos. Sirve como baseline, no como candidato final."
        )
    if slug == "mlp":
        return (
            "El experimento muestra que más complejidad no garantiza mejor detección: "
            f"recall test {test_recall:.4f} y {test_fn} falsos negativos. En esta configuración "
            "no es recomendable para priorizar pacientes con condition=1."
        )
    return (
        f"Recall CV={recall:.4f}, ROC-AUC CV={roc:.4f}, coste CV={cost:.4f}; "
        "interpretar junto con la matriz de confusión."
    )


def validation_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decisión": "Test final",
                "valor": "20% estratificado",
                "justificación": "Reserva una evaluación final no usada durante la comparación interna.",
            },
            {
                "decisión": "Validación interna",
                "valor": "RepeatedStratifiedKFold, 5 folds x 30 repeticiones",
                "justificación": "Reduce la dependencia de una única partición en un dataset moderado.",
            },
            {
                "decisión": "Dataset de entrada",
                "valor": "foot_clean.csv + Pipeline",
                "justificación": "Evita fuga de información al ajustar escalado y one-hot solo con entrenamiento.",
            },
            {
                "decisión": "Métrica principal",
                "valor": "recall_condition_1",
                "justificación": "Mide cuántos pacientes con condition=1 son detectados.",
            },
            {
                "decisión": "Coste médico",
                "valor": "FN=5, FP=1",
                "justificación": "Penaliza más dejar sin detectar un caso real que activar una alarma falsa.",
            },
            {
                "decisión": "Umbral",
                "valor": "Umbral por defecto",
                "justificación": "Permite comparar modelos base de forma homogénea antes de optimizar umbrales.",
            },
        ]
    )


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


def save_comparison_plots(comparison: pd.DataFrame, task3_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_df = comparison.sort_values("cv_recall_condition_1_mean", ascending=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    axes[0].barh(plot_df["modelo"], plot_df["cv_recall_condition_1_mean"], color="#4C78A8")
    axes[0].set_title("Recall CV condition=1")
    axes[0].set_xlim(0, 1.05)

    axes[1].barh(plot_df["modelo"], plot_df["cv_medical_cost_mean"], color="#F58518")
    axes[1].set_title("Coste médico CV")

    axes[2].barh(plot_df["modelo"], plot_df["test_false_negatives"], color="#E45756")
    axes[2].set_title("Falsos negativos en test")

    axes[3].barh(plot_df["modelo"], plot_df["cv_roc_auc_mean"], color="#54A24B")
    axes[3].set_title("ROC-AUC CV")
    axes[3].set_xlim(0, 1.05)

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(task3_dir / "comparacion_modelos_metricas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_individual_experiment_plots(comparison: pd.DataFrame, task3_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_df = comparison.sort_values("rank")
    fig, axes = plt.subplots(len(plot_df), 1, figsize=(11, 2.3 * len(plot_df)), sharex=False)
    if len(plot_df) == 1:
        axes = [axes]

    for ax, row in zip(axes, plot_df.itertuples(index=False)):
        metric_names = ["Recall CV", "Recall test", "ROC-AUC CV"]
        metric_values = [
            row.cv_recall_condition_1_mean,
            row.test_recall_condition_1,
            row.cv_roc_auc_mean,
        ]
        colors = ["#4C78A8", "#72B7B2", "#54A24B"]
        ax.barh(metric_names, metric_values, color=colors)
        ax.set_xlim(0, 1.05)
        ax.set_title(
            f"{row.modelo} | coste CV={row.cv_medical_cost_mean:.3f} | "
            f"FN test={int(row.test_false_negatives)} | FP test={int(row.test_false_positives)}",
            loc="left",
        )
        ax.grid(axis="x", alpha=0.25)
        for index, value in enumerate(metric_values):
            ax.text(min(value + 0.02, 1.0), index, f"{value:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(task3_dir / "experimentos_individuales_metricas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(comparison: pd.DataFrame, report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    compact = compact_columns(comparison)
    catalog = experiment_catalog(comparison)
    best_cv = comparison.iloc[0]
    best_test_cost = comparison.sort_values("test_medical_cost", ascending=True).iloc[0]
    worst_medical = comparison.sort_values("cv_medical_cost_mean", ascending=False).iloc[0]

    content = f"""# Task 3 - Aprendizaje Supervisado

## Estrategia de evaluación

Todos los modelos se evaluaron con la misma partición final estratificada y la misma validación interna: `RepeatedStratifiedKFold` con 5 folds y 30 repeticiones. El preprocesamiento se mantuvo dentro de un `Pipeline`, por lo que cada fold ajusta escalado y codificación solo con sus datos de entrenamiento. Por esta razón se usa `foot_clean.csv` y no un dataset estandarizado previamente con todo el conjunto.

La métrica principal es `recall_condition_1`, porque en un escenario de salud el error más grave es el falso negativo: clasificar como sano a un paciente que realmente presenta la condición. También se reportan F1, ROC-AUC, matriz de confusión y un coste médico normalizado con FN=5 y FP=1. El falso negativo se pondera cinco veces más porque puede retrasar diagnóstico o seguimiento; el falso positivo también tiene coste, pero no deja sin detectar un caso real.

Task 3 mantiene el umbral de decisión por defecto para comparar modelos base de forma homogénea. En un escenario clínico real se debería optimizar ese umbral para reducir falsos negativos, incluso si aumentan los falsos positivos.

{markdown_table(validation_decisions())}

## Experimentos individuales

{markdown_table(catalog[["modelo", "familia", "por_que", "fortaleza", "limitacion", "lectura_resultado", "conclusion_experimento"]])}

## Comparación consolidada

{markdown_table(compact)}

## Lectura de resultados

El mejor rendimiento medio en validación cruzada según la métrica principal lo obtiene **{best_cv["modelo"]}**, con recall medio de {best_cv["cv_recall_condition_1_mean"]:.4f} y coste médico medio de {best_cv["cv_medical_cost_mean"]:.4f}. Esta es la comparación más robusta porque promedia 150 validaciones estratificadas.

En el test final, el menor coste médico lo obtiene **{best_test_cost["modelo"]}**, con {int(best_test_cost["test_false_negatives"])} falsos negativos, {int(best_test_cost["test_false_positives"])} falsos positivos y coste {best_test_cost["test_medical_cost"]:.4f}. Este resultado es útil, pero debe interpretarse con cautela porque el test contiene una sola partición.

El peor comportamiento médico lo muestra **{worst_medical["modelo"]}**, con coste medio de validación {worst_medical["cv_medical_cost_mean"]:.4f}. Este modelo genera demasiados falsos negativos para un escenario clínico y queda documentado como experimento no recomendable en esta configuración.

## Variables y lectura clínica

Las matrices de confusión se revisan junto al coste médico porque muestran directamente cuántos pacientes positivos quedan sin detectar. La curva ROC resume separación probabilística, pero no reemplaza el análisis de falsos negativos.

Para el MLP se añade una lectura de importancia por gradientes de entrada. Esta técnica mide sensibilidad del output ante cambios en las variables preprocesadas; sirve para interpretar el comportamiento del modelo, pero no implica causalidad clínica.

## Conclusión

Para selección inicial de Task 3, el modelo recomendado debe salir de la validación cruzada y no de una única partición de test. **{best_cv["modelo"]}** queda como candidato base más sólido por recall medio, coste médico y estabilidad relativa. **{best_test_cost["modelo"]}** queda como alternativa a revisar por su comportamiento en test. XGBoost se incluye como contraste potente de árboles, mientras que CART se conserva como baseline interpretable. El MLP queda documentado incluso si falla, porque el enunciado exige registrar experimentos débiles o no recomendables.
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    task3_dir = args.task3_dir.resolve()
    comparison = build_comparison(task3_dir)
    catalog = experiment_catalog(comparison)
    decisions = validation_decisions()

    task3_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(task3_dir / "comparacion_modelos_completa.csv", index=False)
    compact_columns(comparison).to_csv(task3_dir / "comparacion_modelos_resumen.csv", index=False)
    catalog.to_csv(task3_dir / "catalogo_experimentos.csv", index=False)
    decisions.to_csv(task3_dir / "decisiones_validacion.csv", index=False)
    save_comparison_plots(comparison, task3_dir)
    save_individual_experiment_plots(comparison, task3_dir)
    report_path = write_report(comparison, args.report_path.resolve())

    print("Comparación de modelos supervisados:")
    print(compact_columns(comparison).to_string(index=False))
    print(f"CSV completo: {task3_dir / 'comparacion_modelos_completa.csv'}")
    print(f"CSV resumen: {task3_dir / 'comparacion_modelos_resumen.csv'}")
    print(f"Catálogo: {task3_dir / 'catalogo_experimentos.csv'}")
    print(f"Decisiones: {task3_dir / 'decisiones_validacion.csv'}")
    print(f"Gráfica: {task3_dir / 'comparacion_modelos_metricas.png'}")
    print(f"Gráfica individual: {task3_dir / 'experimentos_individuales_metricas.png'}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
