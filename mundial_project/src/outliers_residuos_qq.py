from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "outliers_residuos_qq.csv"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "outliers_residuos_qq.png"
DEFAULT_QQ_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "qq_plots.png"
MIN_UNIQUE_VALUES = 6
MIN_VALUE_RANGE = 3


def is_outlier_candidate(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    clean_series = series.dropna()
    if clean_series.nunique() < MIN_UNIQUE_VALUES:
        return False
    return (clean_series.max() - clean_series.min()) > MIN_VALUE_RANGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta outliers con residuos estandarizados y genera QQ-plots."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Ruta al archivo CSV. Por defecto usa mundial_project/data/foot.csv.",
    )
    parser.add_argument(
        "--target",
        default="condition",
        help="Columna objetivo a excluir de las features. Usa '' para no excluir ninguna.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Umbral absoluto para marcar residuos estandarizados como outliers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta para guardar el resumen en CSV.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Ruta para guardar la tabla de residuos como imagen PNG.",
    )
    parser.add_argument(
        "--qq-output",
        type=Path,
        default=DEFAULT_QQ_OUTPUT,
        help="Ruta para guardar los QQ-plots como imagen PNG.",
    )
    return parser.parse_args()


def normal_quantiles(size: int) -> list[float]:
    normal = NormalDist()
    return [normal.inv_cdf((rank - 0.5) / size) for rank in range(1, size + 1)]


def build_residual_outliers(
    df: pd.DataFrame,
    target: str,
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    features = [
        column
        for column in df.columns
        if column != target and is_outlier_candidate(df[column])
    ]
    if not features:
        raise ValueError("No hay features continuas o de rango amplio para analizar.")

    rows = []
    outlier_indices: dict[str, set[int]] = {}
    for feature in features:
        series = df[feature]
        std = series.std()
        if std == 0 or pd.isna(std):
            z_scores = pd.Series(0.0, index=series.index)
        else:
            z_scores = (series - series.mean()) / std

        mask = z_scores.abs() > z_threshold
        indices = set(df.index[mask].tolist())
        outlier_indices[feature] = indices

        sorted_values = series.dropna().sort_values().reset_index(drop=True)
        theoretical = pd.Series(normal_quantiles(len(sorted_values)))
        qq_correlation = sorted_values.corr(theoretical) if len(sorted_values) > 1 else None

        rows.append(
            {
                "feature": feature,
                "media": series.mean(),
                "desviacion_std": std,
                "umbral_z": z_threshold,
                "outliers_zscore": len(indices),
                "porcentaje_zscore": len(indices) / len(df) * 100,
                "qq_correlacion": qq_correlation,
            }
        )

    return pd.DataFrame(rows), outlier_indices


def save_table_image(table_df: pd.DataFrame, output_path: Path, title: str) -> None:
    display_df = table_df.copy()
    numeric_columns = display_df.select_dtypes(include="number").columns
    for column in numeric_columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    fig_height = max(3.5, len(display_df) * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4C78A8")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F4F7")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_qq_plots(df: pd.DataFrame, target: str, output_path: Path) -> None:
    features = [
        column
        for column in df.columns
        if column != target and is_outlier_candidate(df[column])
    ]
    cols = 3
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.0))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, feature in zip(flat_axes, features):
        values = df[feature].dropna().sort_values().reset_index(drop=True)
        theoretical = pd.Series(normal_quantiles(len(values)))
        ax.scatter(theoretical, values, s=12, alpha=0.7, color="#4C78A8")
        ax.set_title(feature)
        ax.set_xlabel("Cuantiles teoricos normales")
        ax.set_ylabel("Valores observados")
        ax.grid(alpha=0.25)

    for ax in flat_axes[len(features) :]:
        ax.set_visible(False)

    fig.suptitle("QQ-plots por feature", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    summary, _ = build_residual_outliers(df, target, args.z_threshold)

    print(f"CSV analizado: {csv_path}")
    print(summary.to_string(index=False))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Resumen guardado en: {output_path}")

    image_output_path = args.image_output.resolve()
    save_table_image(summary, image_output_path, "Outliers por residuos estandarizados")
    print(f"Imagen guardada en: {image_output_path}")

    qq_output_path = args.qq_output.resolve()
    save_qq_plots(df, target, qq_output_path)
    print(f"QQ-plots guardados en: {qq_output_path}")


if __name__ == "__main__":
    main()
