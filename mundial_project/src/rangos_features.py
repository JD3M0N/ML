from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"


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


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    ranges = build_feature_ranges(df, target)

    print(f"CSV analizado: {csv_path}")
    print(ranges.to_string(index=False))

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ranges.to_csv(output_path, index=False)
        print(f"Resumen guardado en: {output_path}")


if __name__ == "__main__":
    main()
