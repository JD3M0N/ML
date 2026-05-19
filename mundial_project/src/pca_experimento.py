from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica PCA sobre las 13 features estandarizadas y genera vistas 2D/3D."
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
        help="Columna objetivo usada para colorear las proyecciones.",
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
        raise ValueError("No hay features numericas para aplicar PCA.")
    return features


def standardize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    feature_df = df[features]
    std = feature_df.std()
    if (std == 0).any():
        zero_std = std[std == 0].index.tolist()
        raise ValueError(f"No se pueden estandarizar columnas con sigma=0: {zero_std}")
    return (feature_df - feature_df.mean()) / std


def run_pca(standardized_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = standardized_df.to_numpy()
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    components = vt
    scores = x @ components.T
    eigenvalues = (singular_values**2) / (len(standardized_df) - 1)
    explained_ratio = eigenvalues / eigenvalues.sum()
    cumulative_ratio = np.cumsum(explained_ratio)

    component_names = [f"PC{i}" for i in range(1, len(components) + 1)]
    scores_df = pd.DataFrame(scores, columns=component_names, index=standardized_df.index)
    explained_df = pd.DataFrame(
        {
            "componente": component_names,
            "autovalor": eigenvalues,
            "varianza_explicada": explained_ratio,
            "varianza_acumulada": cumulative_ratio,
        }
    )
    loadings_df = pd.DataFrame(
        components.T,
        index=standardized_df.columns,
        columns=component_names,
    )

    return scores_df, explained_df, loadings_df


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


def target_palette(target_series: pd.Series) -> dict[object, str]:
    target_values = sorted(target_series.dropna().unique())
    if len(target_values) >= 2:
        return {target_values[0]: "#4C78A8", target_values[-1]: "#E45756"}
    return {target_values[0]: "#4C78A8"}


def save_pca_2d(
    scores_df: pd.DataFrame,
    explained_df: pd.DataFrame,
    target_series: pd.Series,
    output_path: Path,
) -> None:
    palette = target_palette(target_series)
    fig, ax = plt.subplots(figsize=(8, 6))

    for target_value, color in palette.items():
        mask = target_series == target_value
        ax.scatter(
            scores_df.loc[mask, "PC1"],
            scores_df.loc[mask, "PC2"],
            s=38,
            alpha=0.75,
            color=color,
            label=f"condition={target_value}",
            edgecolor="white",
            linewidth=0.4,
        )

    pc1 = explained_df.loc[0, "varianza_explicada"] * 100
    pc2 = explained_df.loc[1, "varianza_explicada"] * 100
    ax.set_xlabel(f"PC1 ({pc1:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% varianza)")
    ax.set_title("Proyeccion PCA 2D")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pca_3d(
    scores_df: pd.DataFrame,
    explained_df: pd.DataFrame,
    target_series: pd.Series,
    output_path: Path,
) -> None:
    palette = target_palette(target_series)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for target_value, color in palette.items():
        mask = target_series == target_value
        ax.scatter(
            scores_df.loc[mask, "PC1"],
            scores_df.loc[mask, "PC2"],
            scores_df.loc[mask, "PC3"],
            s=32,
            alpha=0.75,
            color=color,
            label=f"condition={target_value}",
        )

    pc1 = explained_df.loc[0, "varianza_explicada"] * 100
    pc2 = explained_df.loc[1, "varianza_explicada"] * 100
    pc3 = explained_df.loc[2, "varianza_explicada"] * 100
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_zlabel(f"PC3 ({pc3:.1f}%)")
    ax.set_title("Proyeccion PCA 3D")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scree_plot(explained_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(explained_df) + 1)
    ax.bar(x, explained_df["varianza_explicada"] * 100, color="#4C78A8", label="Individual")
    ax.plot(
        x,
        explained_df["varianza_acumulada"] * 100,
        color="#E45756",
        marker="o",
        label="Acumulada",
    )
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title("Varianza explicada por PCA")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
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
    scores, explained, loadings = run_pca(standardized)
    scores_with_target = scores.copy()
    scores_with_target[args.target] = df[args.target]

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_with_target.to_csv(output_dir / "pca_scores.csv", index=False)
    explained.to_csv(output_dir / "pca_varianza_explicada.csv", index=False)
    loadings.to_csv(output_dir / "pca_cargas_componentes.csv")

    save_table_image(
        explained,
        output_dir / "pca_varianza_explicada.png",
        "Varianza explicada por PCA",
    )
    save_scree_plot(explained, output_dir / "pca_scree_plot.png")
    save_pca_2d(scores, explained, df[args.target], output_dir / "pca_2d.png")
    save_pca_3d(scores, explained, df[args.target], output_dir / "pca_3d.png")

    print(f"CSV analizado: {csv_path}")
    print(f"Features usadas en PCA: {features}")
    print(explained.head(5).round(4).to_string(index=False))
    print(f"Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
