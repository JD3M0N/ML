from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "REPORT.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


TASK3_CELLS = [
    markdown_cell(
        """
# 3. Aprendizaje Supervisado

Esta sección compara modelos supervisados para predecir `condition`. La lectura se hace desde un contexto médico: no basta con maximizar accuracy, porque el error crítico es clasificar como sano a un paciente que realmente tiene `condition=1`.

La estructura de esta parte es deliberadamente experimental. Primero se fija la estrategia de validación y la métrica clínica, después se revisa cada modelo por separado y finalmente se comparan todos con la misma regla.
"""
    ),
    markdown_cell(
        """
## 3.1. Estrategia experimental y métrica médica

Todos los modelos usan un 20% de test final estratificado, reservado hasta el cierre de cada experimento. La comparación interna usa `RepeatedStratifiedKFold` con 5 folds y 30 repeticiones, es decir, 150 validaciones por modelo. Esta repetición reduce la dependencia de una única partición, algo importante en un dataset médico de tamaño moderado.

El preprocesamiento se integra dentro de un `Pipeline`: las variables continuas se estandarizan, las discretas se codifican con one-hot y las binarias pasan directamente. Así, cada fold ajusta sus transformaciones solo con datos de entrenamiento. Por eso en Task 3 se usa `foot_clean.csv` y no un dataset previamente estandarizado con todo el conjunto, ya que eso podría introducir fuga de información.

La métrica principal es `recall_condition_1`, porque mide qué proporción de pacientes con la condición real fueron detectados. El coste médico se define como `(5·FN + 1·FP) / N`: el falso negativo pesa más porque puede retrasar diagnóstico, seguimiento o tratamiento; el falso positivo también importa, pero suele representar alarma o pruebas adicionales, no dejar sin detectar un caso real.

En esta tarea se mantiene el umbral de decisión por defecto para comparar modelos base de forma homogénea. En un uso clínico real habría que optimizar ese umbral para reducir falsos negativos, incluso si aumentan falsos positivos.
"""
    ),
    code_cell(
        """
validation_summary = read_csv(OUTPUTS_DIR / "task3" / "decisiones_validacion.csv")
show_table("Decisiones de validación y métrica médica", validation_summary, n=None)
"""
    ),
    markdown_cell(
        """
## 3.2. Experimentos uno a uno

Antes de comparar modelos, se registra por qué se prueba cada enfoque y qué se puede extraer de sus resultados. Esto evita que la sección sea solo una tabla: cada modelo responde una pregunta distinta sobre los datos, desde un baseline lineal interpretable hasta modelos no lineales y un ensamblado de árboles.

La gráfica resume por experimento tres señales comparables entre 0 y 1: recall medio en validación cruzada, recall en test y ROC-AUC medio. En el título de cada panel se añaden coste médico medio, falsos negativos y falsos positivos en test, que son las señales clínicas más importantes.
"""
    ),
    code_cell(
        """
experiment_catalog = read_csv(OUTPUTS_DIR / "task3" / "catalogo_experimentos.csv")
catalog_columns = [
    "modelo",
    "familia",
    "por_que",
    "fortaleza",
    "limitacion",
    "lectura_resultado",
    "conclusion_experimento",
]
show_table("Registro de experimentos supervisados", experiment_catalog[catalog_columns], n=None)
show_image(OUTPUTS_DIR / "task3" / "experimentos_individuales_metricas.png", width=950)

model_slugs = [
    ("logistic_regression", "Regresión Logística"),
    ("naive_bayes", "Naive Bayes"),
    ("knn", "KNN"),
    ("decision_tree", "Árbol de Decisión CART"),
    ("xgboost", "XGBoost"),
    ("svm", "SVM RBF"),
    ("mlp", "Red Neuronal MLP"),
]

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

for slug, name in model_slugs:
    display(Markdown(f"### {name}"))
    catalog_row = experiment_catalog.loc[experiment_catalog["slug"] == slug]
    if not catalog_row.empty:
        display(Markdown(f"**Conclusión del experimento:** {catalog_row['conclusion_experimento'].iloc[0]}"))
    cv_path = OUTPUTS_DIR / "task3" / slug / "cv_metricas_resumen.csv"
    test_path = OUTPUTS_DIR / "task3" / slug / "test_metricas.csv"
    if cv_path.exists():
        cv_summary = read_csv(cv_path)
        display(Markdown("**Validación cruzada**"))
        display(cv_summary.loc[cv_summary["metric"].isin(metric_columns), ["metric", "mean", "std", "min", "max"]])
    if test_path.exists():
        display(Markdown("**Test final reservado**"))
        display(read_csv(test_path)[metric_columns])
"""
    ),
    markdown_cell(
        """
## 3.3. Comparación consolidada

La comparación se ordena principalmente por `recall_condition_1` medio en validación cruzada. Como desempate se revisa el coste médico y después los falsos negativos en test. El test final ayuda a confirmar comportamiento, pero no se usa como única razón para declarar ganador.
"""
    ),
    code_cell(
        """
task3_summary = read_csv(OUTPUTS_DIR / "task3" / "comparacion_modelos_resumen.csv")
show_table("Comparación de modelos supervisados", task3_summary, n=None)
show_image(OUTPUTS_DIR / "task3" / "comparacion_modelos_metricas.png", width=950)
"""
    ),
    markdown_cell(
        """
Regresión Logística queda como modelo base más sólido por validación cruzada: combina el mayor recall medio, buen ROC-AUC y el menor coste médico medio. SVM RBF logra el menor coste en el test final y reduce falsos negativos en esa partición, por lo que queda como alternativa prometedora, pero no reemplaza automáticamente al ganador de validación.

XGBoost se añade como contraste de árboles potenciados. Su rendimiento es competitivo y mejora claramente al CART simple, pero no justifica desplazar a Regresión Logística si la decisión principal se apoya en validación cruzada, coste médico e interpretabilidad.

El MLP documenta un intento débil: su recall en test es bajo y deja demasiados falsos negativos. Esto es valioso para el reporte porque muestra que un modelo más complejo no necesariamente es mejor en un dataset pequeño o mediano.
"""
    ),
    markdown_cell(
        """
## 3.4. Matrices de confusión, ROC y curvas de aprendizaje

Las matrices de confusión muestran directamente falsos negativos y falsos positivos. Las curvas ROC resumen separación probabilística, pero no sustituyen la revisión del error clínico. Las curvas de aprendizaje ayudan a detectar si el recall mejora al añadir datos o si el modelo muestra inestabilidad entre entrenamiento y validación.
"""
    ),
    code_cell(
        """
for slug, name in model_slugs:
    display(Markdown(f"### {name}"))
    show_image(OUTPUTS_DIR / "task3" / slug / "test_matriz_confusion.png", width=430)
    show_image(OUTPUTS_DIR / "task3" / slug / "test_roc_curve.png", width=430)

for slug, name in [
    ("logistic_regression", "Regresión Logística"),
    ("svm", "SVM RBF"),
    ("xgboost", "XGBoost"),
    ("knn", "KNN"),
    ("mlp", "MLP"),
]:
    display(Markdown(f"### Curva de aprendizaje - {name}"))
    show_image(OUTPUTS_DIR / "task3" / slug / "learning_curve_recall.png", width=700)
"""
    ),
    markdown_cell(
        """
## 3.5. Variables relevantes y gradientes del MLP

Para la red neuronal se calcula una importancia por gradientes de entrada sobre el MLP entrenado. La idea es medir cuánto cambia la salida del modelo ante pequeñas variaciones de cada variable preprocesada. Esta lectura ayuda a interpretar sensibilidad del modelo, pero no implica causalidad clínica.
"""
    ),
    code_cell(
        """
gradient_path = OUTPUTS_DIR / "task3" / "mlp" / "gradient_feature_importance.csv"
if gradient_path.exists():
    gradients = read_csv(gradient_path)
    show_table("MLP - importancia por gradientes de entrada", gradients, n=None)
    show_image(OUTPUTS_DIR / "task3" / "mlp" / "gradient_feature_importance.png", width=700)
else:
    display(Markdown("No se encontró la importancia por gradientes del MLP. Ejecuta `task3_mlp_gradients.py`."))
"""
    ),
    markdown_cell(
        """
## 3.6. Conclusión médica y limitaciones

La conclusión de Task 3 no debe basarse en accuracy ni en una única partición de test. Con la evidencia actual, Regresión Logística es el modelo base más sólido por validación cruzada e interpretabilidad. SVM RBF queda como alternativa prometedora porque en test reduce falsos negativos, y XGBoost queda como contraste potente de árboles que mejora al CART simple pero no domina la comparación global.

La limitación principal es que todos los modelos se comparan con el umbral por defecto. En medicina, el umbral debería ajustarse explícitamente para controlar falsos negativos y falsos positivos según el coste real del escenario. Por eso Task 4 continúa con optimización de hiperparámetros y selección guiada por coste médico.

Estos resultados son una comparación experimental sobre datos anonimizados; no constituyen una herramienta diagnóstica clínica.
"""
    ),
]


def replace_task3_section() -> None:
    notebook = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    start = next(
        index for index, cell in enumerate(cells) if "".join(cell.get("source", [])).startswith("# 3.")
    )
    end = next(
        index for index, cell in enumerate(cells[start + 1 :], start=start + 1)
        if "".join(cell.get("source", [])).startswith("# 4.")
    )
    notebook["cells"] = cells[:start] + TASK3_CELLS + cells[end:]
    REPORT_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    replace_task3_section()
    print(f"Se actualizó la sección 3 de {REPORT_PATH}")
