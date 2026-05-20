from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import export_text, plot_tree

import task3_common
import task3_decision_tree
import task3_xgboost


def prepared_train_test(csv_path: Path | None, test_size: float) -> tuple[pd.DataFrame, pd.Series]:
    df, _ = task3_common.load_dataset(csv_path)
    x, y = task3_common.split_features_target(df)
    x_train, _, y_train, _ = task3_common.stratified_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=task3_common.RANDOM_STATE,
    )
    return x_train, y_train


def feature_names(model: Any) -> list[str]:
    return list(model.named_steps["preprocess"].get_feature_names_out())


def save_cart_tree(model: Any, output_dir: Path) -> None:
    names = feature_names(model)
    tree_model = model.named_steps["model"]

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        tree_model,
        feature_names=names,
        class_names=["condition=0", "condition=1"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Árbol de Decisión CART - reglas aprendidas")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree_rules.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rules = export_text(tree_model, feature_names=names, decimals=3)
    (output_dir / "decision_tree_rules.txt").write_text(rules, encoding="utf-8")


def save_xgboost_importance(model: Any, output_dir: Path) -> None:
    booster = model.named_steps["model"]
    names = feature_names(model)
    importances = pd.DataFrame(
        {
            "transformed_feature": names,
            "importance": booster.feature_importances_,
        }
    )
    importances["feature"] = importances["transformed_feature"].map(original_feature_from_transformed)
    grouped = (
        importances.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)

    plot_df = grouped.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4C78A8")
    ax.set_title("XGBoost - importancia de variables")
    ax.set_xlabel("Importancia acumulada por variable original")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def original_feature_from_transformed(feature_name: str) -> str:
    raw = feature_name.split("__", 1)[-1]
    for feature in (
        task3_common.CONTINUOUS_FEATURES
        + task3_common.CATEGORICAL_FEATURES
        + task3_common.BINARY_FEATURES
    ):
        if raw == feature or raw.startswith(f"{feature}_"):
            return feature
    return raw


def save_xgboost_first_tree(model: Any, output_dir: Path, max_depth: int = 3) -> None:
    booster = model.named_steps["model"].get_booster()
    trees = booster.trees_to_dataframe()
    first_tree = trees.loc[trees["Tree"] == 0].copy()
    first_tree.to_csv(output_dir / "xgboost_first_tree_nodes.csv", index=False)

    nodes = {row.ID: row for row in first_tree.itertuples(index=False)}
    root_id = "0-0"
    positions: dict[str, tuple[float, float]] = {}
    labels: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []

    def walk(node_id: str, depth: int, x_left: float, x_right: float) -> None:
        if node_id not in nodes or depth > max_depth:
            return
        node = nodes[node_id]
        x = (x_left + x_right) / 2
        y = -depth
        positions[node_id] = (x, y)
        if node.Feature == "Leaf":
            labels[node_id] = f"Leaf\nvalor={node.Gain:.3f}\ncover={node.Cover:.1f}"
            return

        split_feature = original_feature_from_transformed(str(node.Feature))
        labels[node_id] = (
            f"{split_feature}\n<= {node.Split:.3f}\n"
            f"gain={node.Gain:.3f}"
        )
        if isinstance(node.Yes, str):
            edges.append((node_id, node.Yes, "sí"))
            walk(node.Yes, depth + 1, x_left, x)
        if isinstance(node.No, str):
            edges.append((node_id, node.No, "no"))
            walk(node.No, depth + 1, x, x_right)

    walk(root_id, 0, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    for parent, child, label in edges:
        if parent not in positions or child not in positions:
            continue
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, fontsize=8, ha="center", va="center")

    for node_id, (x, y) in positions.items():
        is_leaf = labels[node_id].startswith("Leaf")
        ax.text(
            x,
            y,
            labels[node_id],
            ha="center",
            va="center",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#F2CF5B" if is_leaf else "#B7D7F0",
                "edgecolor": "#333333",
                "linewidth": 0.8,
            },
        )

    ax.set_title("XGBoost - primer árbol del ensamblado")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_dir / "xgboost_first_tree.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = task3_common.DEFAULT_TEST_SIZE,
) -> Path:
    selected_output = (output_dir or task3_common.DEFAULT_OUTPUT_DIR).resolve() / "tree_visualizations"
    selected_output.mkdir(parents=True, exist_ok=True)

    x_train, y_train = prepared_train_test(csv_path, test_size)

    cart = task3_decision_tree.build_estimator()
    cart.fit(x_train, y_train)
    save_cart_tree(cart, selected_output)

    xgb = task3_xgboost.build_estimator()
    xgb.fit(x_train, y_train)
    save_xgboost_importance(xgb, selected_output)
    save_xgboost_first_tree(xgb, selected_output)

    print(f"Visualizaciones de árboles guardadas en: {selected_output}")
    return selected_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: visualizaciones de CART y XGBoost.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=task3_common.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=task3_common.DEFAULT_TEST_SIZE)
    args = parser.parse_args()
    run(csv_path=args.csv, output_dir=args.output_dir, test_size=args.test_size)


if __name__ == "__main__":
    main()
