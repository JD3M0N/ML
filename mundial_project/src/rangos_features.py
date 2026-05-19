from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "outputs" / "rangos_features.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestra los rangos de valores de las features de un CSV."
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
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional para guardar el resumen de rangos en CSV.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Ruta para guardar la tabla de rangos como imagen PNG.",
    )
    return parser.parse_args()


def build_feature_ranges(df: pd.DataFrame, target: str) -> pd.DataFrame:
    features = [column for column in df.columns if column != target]
    if not features:
        raise ValueError("No hay features para analizar. Revisa --target o el CSV.")

    rows = []
    for feature in features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            minimum = series.min()
            maximum = series.max()
            value_range = maximum - minimum
        else:
            minimum = None
            maximum = None
            value_range = None

        rows.append(
            {
                "feature": feature,
                "tipo_dato": str(series.dtype),
                "minimo": minimum,
                "maximo": maximum,
                "rango": value_range,
                "valores_unicos": series.nunique(dropna=True),
                "valores_nulos": series.isna().sum(),
            }
        )

    return pd.DataFrame(rows)


def save_ranges_image(ranges: pd.DataFrame, output_path: Path) -> None:
    display_ranges = ranges.copy()
    for column in ("minimo", "maximo", "rango"):
        display_ranges[column] = display_ranges[column].map(
            lambda value: f"{value:g}" if pd.notna(value) else ""
        )

    row_count = len(display_ranges)
    fig_height = max(3.5, row_count * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_ranges.values,
        colLabels=display_ranges.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4C78A8")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F4F7")

    fig.suptitle("Rangos de valores por feature", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    ranges = build_feature_ranges(df, target)

    print(f"CSV analizado: {csv_path}")
    print(ranges.to_string(index=False))

    image_output_path = args.image_output.resolve()
    save_ranges_image(ranges, image_output_path)
    print(f"Imagen guardada en: {image_output_path}")

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranges.to_csv(output_path, index=False)
        print(f"Resumen guardado en: {output_path}")


if __name__ == "__main__":
    main()
