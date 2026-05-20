from __future__ import annotations

import json
from pathlib import Path


REPORT_PATH = Path(__file__).resolve().parents[2] / "REPORT.ipynb"


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


def main() -> None:
    notebook = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    marker = "## 3.3. Comparación consolidada"
    insert_at = next(
        index for index, cell in enumerate(cells)
        if "".join(cell.get("source", [])).startswith(marker)
    )

    # Avoid duplicating the section if the script is run more than once.
    if any(
        "".join(cell.get("source", [])).startswith("## 3.3. Visualización de modelos basados en árboles")
        for cell in cells
    ):
        print("La sección de árboles ya existe en REPORT.ipynb")
        return

    new_cells = [
        markdown_cell(
            """
## 3.3. Visualización de modelos basados en árboles

Sí es posible modelar visualmente los árboles, pero CART y XGBoost se leen de forma distinta. CART es un único árbol de reglas, por lo que puede mostrarse completo. XGBoost es un ensamblado de muchos árboles pequeños: por eso se muestra un árbol representativo del ensamblado y una gráfica de importancia de variables.

Esta visualización ayuda a explicar cómo se toman algunas decisiones, pero no reemplaza las métricas médicas. Un árbol puede ser fácil de leer y aun así dejar demasiados falsos negativos.
"""
        ),
        code_cell(
            """
tree_viz_dir = OUTPUTS_DIR / "task3" / "tree_visualizations"

display(Markdown("### CART: árbol completo de reglas"))
show_image(tree_viz_dir / "decision_tree_rules.png", width=950)

display(Markdown("### XGBoost: un árbol representativo del ensamblado"))
show_image(tree_viz_dir / "xgboost_first_tree.png", width=950)

display(Markdown("### XGBoost: importancia de variables"))
show_image(tree_viz_dir / "xgboost_feature_importance.png", width=750)

xgb_importance_path = tree_viz_dir / "xgboost_feature_importance.csv"
if xgb_importance_path.exists():
    show_table("Importancia XGBoost por variable original", read_csv(xgb_importance_path), n=None)
"""
        ),
        markdown_cell(
            """
La lectura visual refuerza la conclusión de la comparación: CART aporta reglas claras, pero su rendimiento clínico es inferior. XGBoost aprovecha mejor interacciones entre variables y supera al árbol simple, aunque su interpretabilidad baja porque la predicción final combina muchos árboles. Por eso se conserva CART como explicación simple y XGBoost como contraste potente, no como sustituto automático del modelo recomendado.
"""
        ),
    ]

    notebook["cells"] = cells[:insert_at] + new_cells + cells[insert_at:]

    # Renumber following headings from 3.3 onward.
    replacements = {
        "## 3.3. Comparación consolidada": "## 3.4. Comparación consolidada",
        "## 3.4. Matrices de confusión, ROC y curvas de aprendizaje": "## 3.5. Matrices de confusión, ROC y curvas de aprendizaje",
        "## 3.5. Variables relevantes y gradientes del MLP": "## 3.6. Variables relevantes y gradientes del MLP",
        "## 3.6. Conclusión médica y limitaciones": "## 3.7. Conclusión médica y limitaciones",
    }
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            if source.startswith(old):
                source = source.replace(old, new, 1)
        cell["source"] = source.splitlines(keepends=True)

    REPORT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Se añadió la visualización de árboles a REPORT.ipynb")


if __name__ == "__main__":
    main()
