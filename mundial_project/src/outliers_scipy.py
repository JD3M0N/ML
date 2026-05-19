from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "outliers_scipy.csv"
DEFAULT_COMPARISON_OUTPUT = (
    PROJECT_ROOT / "outputs" / "outliers" / "comparacion_scipy_manual.csv"
)
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "outliers_scipy.png"
DEFAULT_COMPARISON_IMAGE_OUTPUT = (
    PROJECT_ROOT / "outputs" / "outliers" / "comparacion_scipy_manual.png"
)
DEFAULT_QQ_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "qq_plots_scipy.png"
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
        description="Detecta outliers con SciPy usando z-score y QQ-plots."
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
        help="Umbral absoluto para marcar z-scores como outliers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta para guardar el resumen SciPy en CSV.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT,
        help="Ruta para guardar la comparacion SciPy vs calculo manual en CSV.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Ruta para guardar la tabla SciPy como imagen PNG.",
    )
    parser.add_argument(
        "--comparison-image-output",
        type=Path,
        default=DEFAULT_COMPARISON_IMAGE_OUTPUT,
        help="Ruta para guardar la comparacion como imagen PNG.",
    )
    parser.add_argument(
        "--qq-output",
        type=Path,
        default=DEFAULT_QQ_OUTPUT,
        help="Ruta para guardar los QQ-plots de SciPy como imagen PNG.",
    )
    return parser.parse_args()


def numeric_features(df: pd.DataFrame, target: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column != target and is_outlier_candidate(df[column])
    ]


def build_scipy_outliers(
    df: pd.DataFrame,
    target: str,
    z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    features = numeric_features(df, target)
    if not features:
        raise ValueError("No hay features continuas o de rango amplio para analizar.")

    rows = []
    outlier_indices: dict[str, set[int]] = {}
    for feature in features:
        series = df[feature]
        z_scores = pd.Series(
            stats.zscore(series, nan_policy="omit", ddof=1),
            index=series.index,
        ).fillna(0.0)

        mask = z_scores.abs() > z_threshold
        indices = set(df.index[mask].tolist())
        outlier_indices[feature] = indices

        (_, _), (_, _, qq_r) = stats.probplot(series.dropna(), dist="norm")
        rows.append(
            {
                "feature": feature,
                "media": series.mean(),
                "desviacion_std": series.std(),
                "umbral_z": z_threshold,
                "outliers_scipy": len(indices),
                "porcentaje_scipy": len(indices) / len(df) * 100,
                "qq_r_scipy": qq_r,
            }
        )

    return pd.DataFrame(rows), outlier_indices


def build_manual_comparison(
    df: pd.DataFrame,
    target: str,
    scipy_indices: dict[str, set[int]],
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    rows = []
    for feature in numeric_features(df, target):
        series = df[feature]
        std = series.std()
        if std == 0 or pd.isna(std):
            manual_z_scores = pd.Series(0.0, index=series.index)
        else:
            manual_z_scores = (series - series.mean()) / std

        manual_set = set(df.index[manual_z_scores.abs() > z_threshold].tolist())
        scipy_set = scipy_indices.get(feature, set())
        both = manual_set & scipy_set
        union = manual_set | scipy_set
        agreement = len(both) / len(union) * 100 if union else 100.0

        rows.append(
            {
                "feature": feature,
                "outliers_manual": len(manual_set),
                "outliers_scipy": len(scipy_set),
                "coinciden": len(both),
                "solo_manual": len(manual_set - scipy_set),
                "solo_scipy": len(scipy_set - manual_set),
                "acuerdo_porcentaje": agreement,
            }
        )

    return pd.DataFrame(rows)


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


def save_scipy_qq_plots(df: pd.DataFrame, target: str, output_path: Path) -> None:
    features = numeric_features(df, target)
    cols = 3
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.0))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, feature in zip(flat_axes, features):
        stats.probplot(df[feature].dropna(), dist="norm", plot=ax)
        ax.set_title(feature)
        ax.grid(alpha=0.25)

    for ax in flat_axes[len(features) :]:
        ax.set_visible(False)

    fig.suptitle("QQ-plots con SciPy por feature", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    summary, scipy_indices = build_scipy_outliers(df, target, args.z_threshold)
    comparison = build_manual_comparison(df, target, scipy_indices, args.z_threshold)

    print(f"CSV analizado: {csv_path}")
    print(summary.to_string(index=False))
    print("\nComparacion SciPy vs calculo manual:")
    print(comparison.to_string(index=False))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Resumen SciPy guardado en: {output_path}")

    comparison_output_path = args.comparison_output.resolve()
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_output_path, index=False)
    print(f"Comparacion guardada en: {comparison_output_path}")

    image_output_path = args.image_output.resolve()
    save_table_image(summary, image_output_path, "Outliers con SciPy")
    print(f"Imagen guardada en: {image_output_path}")

    comparison_image_output_path = args.comparison_image_output.resolve()
    save_table_image(
        comparison,
        comparison_image_output_path,
        "Comparacion SciPy vs calculo manual",
    )
    print(f"Imagen de comparacion guardada en: {comparison_image_output_path}")

    qq_output_path = args.qq_output.resolve()
    save_scipy_qq_plots(df, target, qq_output_path)
    print(f"QQ-plots SciPy guardados en: {qq_output_path}")


if __name__ == "__main__":
    main()
