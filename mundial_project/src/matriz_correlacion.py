from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "correlacion" / "matriz_correlacion.csv"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "outputs" / "correlacion" / "matriz_correlacion.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula y guarda la matriz de correlacion de las columnas numericas."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Ruta al archivo CSV. Por defecto usa mundial_project/data/foot.csv.",
    )
    parser.add_argument(
        "--method",
        choices=("pearson", "spearman", "kendall"),
        default="pearson",
        help="Metodo de correlacion a calcular.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta para guardar la matriz en CSV.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Ruta para guardar el heatmap como imagen PNG.",
    )
    return parser.parse_args()


def build_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No hay columnas numericas para calcular correlacion.")
    return numeric_df.corr(method=method)


def save_correlation_heatmap(correlation: pd.DataFrame, output_path: Path, method: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlacion")

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.index)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation.index)

    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]
            color = "white" if abs(value) >= 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(f"Matriz de correlacion ({method})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()

    df = pd.read_csv(csv_path)
    correlation = build_correlation_matrix(df, args.method)

    print(f"CSV analizado: {csv_path}")
    print(correlation.round(4).to_string())

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correlation.to_csv(output_path)
    print(f"Matriz guardada en: {output_path}")

    image_output_path = args.image_output.resolve()
    save_correlation_heatmap(correlation, image_output_path, args.method)
    print(f"Imagen guardada en: {image_output_path}")


if __name__ == "__main__":
    main()
