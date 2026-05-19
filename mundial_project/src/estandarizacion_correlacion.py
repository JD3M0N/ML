from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "estandarizacion_correlacion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estandariza las features y genera matriz de correlacion, heatmap "
            "y matriz de dispersion."
        )
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
        help="Columna objetivo usada para colorear la matriz de dispersion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardan CSVs e imagenes.",
    )
    return parser.parse_args()


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    features = [
        column
        for column in df.select_dtypes(include="number").columns
        if column != target
    ]
    if not features:
        raise ValueError("No hay features numericas para estandarizar.")
    return features


def standardize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    feature_df = df[features]
    std = feature_df.std()
    if (std == 0).any():
        zero_std = std[std == 0].index.tolist()
        raise ValueError(f"No se pueden estandarizar columnas con sigma=0: {zero_std}")
    return (feature_df - feature_df.mean()) / std


def build_standardization_check(standardized_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": standardized_df.columns,
            "media_estandarizada": standardized_df.mean().values,
            "varianza_estandarizada": standardized_df.var().values,
            "desviacion_estandarizada": standardized_df.std().values,
        }
    )


def save_table_image(table_df: pd.DataFrame, output_path: Path, title: str) -> None:
    display_df = table_df.copy()
    numeric_columns = display_df.select_dtypes(include="number").columns
    for column in numeric_columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    fig_height = max(3.5, len(display_df) * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
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


def save_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8.5))
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

    ax.set_title("Matriz R de correlacion Pearson sobre features estandarizadas")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scatter_matrix(
    standardized_df: pd.DataFrame,
    target_series: pd.Series,
    output_path: Path,
) -> None:
    columns = standardized_df.columns.tolist()
    count = len(columns)
    target_values = sorted(target_series.dropna().unique())
    palette = {
        target_values[0]: "#4C78A8",
        target_values[-1]: "#E45756",
    } if len(target_values) >= 2 else {target_values[0]: "#4C78A8"}

    fig, axes = plt.subplots(count, count, figsize=(count * 1.55, count * 1.55))

    for row, y_column in enumerate(columns):
        for col, x_column in enumerate(columns):
            ax = axes[row, col]
            if row == col:
                ax.hist(standardized_df[x_column], bins=18, color="#72B7B2", edgecolor="white")
            else:
                for target_value in target_values:
                    mask = target_series == target_value
                    ax.scatter(
                        standardized_df.loc[mask, x_column],
                        standardized_df.loc[mask, y_column],
                        s=8,
                        alpha=0.55,
                        color=palette[target_value],
                        label=f"condition={target_value}",
                    )

            if row == count - 1:
                ax.set_xlabel(x_column, fontsize=7, rotation=45)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(y_column, fontsize=7)
            else:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=6)
            ax.grid(alpha=0.15)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)

    fig.suptitle("Matriz de dispersion de features estandarizadas", fontsize=16)
    fig.tight_layout(rect=(0, 0, 0.98, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()

    df = pd.read_csv(csv_path)
    features = get_feature_columns(df, args.target)
    standardized = standardize_features(df, features)
    standardization_check = build_standardization_check(standardized)
    correlation = standardized.corr(method="pearson")

    output_dir.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_dir / "features_estandarizadas.csv", index=False)
    standardization_check.to_csv(output_dir / "comprobacion_estandarizacion.csv", index=False)
    correlation.to_csv(output_dir / "matriz_correlacion_pearson_estandarizada.csv")

    save_table_image(
        standardization_check,
        output_dir / "comprobacion_estandarizacion.png",
        "Comprobacion de estandarizacion",
    )
    save_correlation_heatmap(
        correlation,
        output_dir / "heatmap_correlacion_pearson_estandarizada.png",
    )
    save_scatter_matrix(
        standardized,
        df[args.target],
        output_dir / "matriz_dispersion_estandarizada.png",
    )

    print(f"CSV analizado: {csv_path}")
    print(f"Features estandarizadas: {features}")
    print(standardization_check.round(4).to_string(index=False))
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
