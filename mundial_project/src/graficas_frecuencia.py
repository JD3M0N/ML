from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graficas_frecuencia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficas de frecuencia para cada feature de un CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Ruta al archivo CSV. Por defecto usa mundial_project/data/foot.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardaran las graficas.",
    )
    parser.add_argument(
        "--target",
        default="condition",
        help="Columna objetivo a excluir de las features. Usa '' para no excluir ninguna.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Maximo de valores unicos para usar grafica de barras en columnas numericas.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Numero de bins para histogramas de columnas numericas continuas.",
    )
    return parser.parse_args()


def clean_filename(name: str) -> str:
    valid = []
    for char in name:
        valid.append(char if char.isalnum() or char in ("-", "_") else "_")
    return "".join(valid).strip("_") or "feature"


def plot_feature_frequency(
    series: pd.Series,
    feature_name: str,
    output_path: Path,
    max_categories: int,
    bins: int,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    current_ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    clean_series = series.dropna()
    unique_count = clean_series.nunique()

    if pd.api.types.is_numeric_dtype(clean_series) and unique_count > max_categories:
        current_ax.hist(clean_series, bins=bins, edgecolor="black", color="#4C78A8")
        current_ax.set_xlabel(feature_name)
        current_ax.set_ylabel("Frecuencia")
    else:
        counts = clean_series.value_counts().sort_index()
        counts.plot(kind="bar", ax=current_ax, color="#59A14F", edgecolor="black")
        current_ax.set_xlabel(feature_name)
        current_ax.set_ylabel("Frecuencia")
        current_ax.tick_params(axis="x", rotation=45)

    current_ax.set_title(f"Frecuencia de {feature_name}")
    current_ax.grid(axis="y", alpha=0.25)

    if ax is None:
        plt.tight_layout()
        current_ax.figure.savefig(output_path, dpi=160)
        plt.close(current_ax.figure)

    return current_ax


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()

    df = pd.read_csv(csv_path)

    target = args.target.strip()
    features = [column for column in df.columns if column != target]
    if not features:
        raise ValueError("No hay features para graficar. Revisa --target o el CSV.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        output_path = output_dir / f"{clean_filename(feature)}_frecuencia.png"
        plot_feature_frequency(
            df[feature],
            feature,
            output_path,
            args.max_categories,
            args.bins,
        )

    cols = 3
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.8))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, feature in zip(flat_axes, features):
        plot_feature_frequency(
            df[feature],
            feature,
            output_dir / f"{clean_filename(feature)}_frecuencia.png",
            args.max_categories,
            args.bins,
            ax=ax,
        )

    for ax in flat_axes[len(features) :]:
        ax.set_visible(False)

    fig.suptitle("Graficas de frecuencia por feature", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    combined_path = output_dir / "todas_las_frecuencias.png"
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    print(f"CSV analizado: {csv_path}")
    print(f"Graficas guardadas en: {output_dir}")
    print(f"Grafica combinada: {combined_path.resolve()}")


if __name__ == "__main__":
    main()
