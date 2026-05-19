from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from outliers_iqr import build_iqr_outliers
from outliers_residuos_qq import build_residual_outliers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "comparacion_outliers.csv"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "outputs" / "outliers" / "comparacion_outliers.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara outliers detectados con IQR y residuos estandarizados."
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
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="Multiplicador del IQR para definir limites.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Umbral absoluto para residuos estandarizados.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta para guardar la comparacion en CSV.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Ruta para guardar la tabla de comparacion como imagen PNG.",
    )
    return parser.parse_args()


def build_outlier_comparison(
    df: pd.DataFrame,
    target: str,
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    iqr_summary, iqr_indices = build_iqr_outliers(df, target, iqr_multiplier)
    z_summary, z_indices = build_residual_outliers(df, target, z_threshold)

    iqr_counts = iqr_summary.set_index("feature")["outliers_iqr"].to_dict()
    z_counts = z_summary.set_index("feature")["outliers_zscore"].to_dict()
    features = sorted(set(iqr_indices) | set(z_indices))

    rows = []
    for feature in features:
        iqr_set = iqr_indices.get(feature, set())
        z_set = z_indices.get(feature, set())
        both = iqr_set & z_set
        union = iqr_set | z_set
        agreement = len(both) / len(union) * 100 if union else 100.0

        rows.append(
            {
                "feature": feature,
                "outliers_iqr": iqr_counts.get(feature, 0),
                "outliers_zscore": z_counts.get(feature, 0),
                "coinciden": len(both),
                "solo_iqr": len(iqr_set - z_set),
                "solo_zscore": len(z_set - iqr_set),
                "acuerdo_porcentaje": agreement,
            }
        )

    return pd.DataFrame(rows)


def save_table_image(table_df: pd.DataFrame, output_path: Path) -> None:
    display_df = table_df.copy()
    numeric_columns = display_df.select_dtypes(include="number").columns
    for column in numeric_columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    fig_height = max(3.5, len(display_df) * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(13, fig_height))
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

    fig.suptitle("Comparacion de outliers: IQR vs z-score", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    comparison = build_outlier_comparison(
        df,
        target,
        args.iqr_multiplier,
        args.z_threshold,
    )

    print(f"CSV analizado: {csv_path}")
    print(comparison.to_string(index=False))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(f"Comparacion guardada en: {output_path}")

    image_output_path = args.image_output.resolve()
    save_table_image(comparison, image_output_path)
    print(f"Imagen guardada en: {image_output_path}")


if __name__ == "__main__":
    main()
