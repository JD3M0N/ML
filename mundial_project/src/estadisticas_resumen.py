from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "outputs" / "estadisticas_resumen.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula media y varianza para cada feature de un CSV."
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
        help="Ruta opcional para guardar el resumen en CSV.",
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=DEFAULT_IMAGE_OUTPUT,
        help="Ruta para guardar la tabla de estadisticas como imagen PNG.",
    )
    return parser.parse_args()


def build_summary_stats(df: pd.DataFrame, target: str) -> pd.DataFrame:
    features = [column for column in df.columns if column != target]
    if not features:
        raise ValueError("No hay features para analizar. Revisa --target o el CSV.")

    rows = []
    for feature in features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            mean = series.mean()
            variance = series.var()
        else:
            mean = None
            variance = None

        rows.append(
            {
                "feature": feature,
                "tipo_dato": str(series.dtype),
                "media_mu": mean,
                "varianza_sigma2": variance,
                "valores_unicos": series.nunique(dropna=True),
                "valores_nulos": series.isna().sum(),
            }
        )

    return pd.DataFrame(rows)


def save_summary_image(summary: pd.DataFrame, output_path: Path) -> None:
    display_summary = summary.copy()
    for column in ("media_mu", "varianza_sigma2"):
        display_summary[column] = display_summary[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else ""
        )

    row_count = len(display_summary)
    fig_height = max(3.5, row_count * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_summary.values,
        colLabels=display_summary.columns,
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

    fig.suptitle("Estadisticas de resumen por feature", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    summary = build_summary_stats(df, target)

    print(f"CSV analizado: {csv_path}")
    print(summary.to_string(index=False))

    image_output_path = args.image_output.resolve()
    save_summary_image(summary, image_output_path)
    print(f"Imagen guardada en: {image_output_path}")

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Resumen guardado en: {output_path}")


if __name__ == "__main__":
    main()
