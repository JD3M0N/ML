from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from task2_common import TARGET, add_common_args, load_dataset, prepare_features, save_processed_feature_names


def save_pca_outputs(x_processed: np.ndarray, y: pd.Series, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca_2d = PCA(n_components=2, random_state=42)
    x_pca = pca_2d.fit_transform(x_processed)
    pca_df = pd.DataFrame({"PC1": x_pca[:, 0], "PC2": x_pca[:, 1], TARGET: y.values})
    pca_df.to_csv(output_dir / "pca_scores.csv", index=False)

    pca_full = PCA()
    pca_full.fit(x_processed)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    variance_df = pd.DataFrame(
        {
            "componente": [f"PC{i}" for i in range(1, len(cumulative) + 1)],
            "varianza_explicada": pca_full.explained_variance_ratio_,
            "varianza_acumulada": cumulative,
        }
    )
    variance_df.to_csv(output_dir / "pca_varianza.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=TARGET, palette="Set2", ax=ax)
    ax.set_title("Visualizacion PCA coloreada por condition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_condition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="#4C78A8")
    ax.axhline(0.80, linestyle="--", color="#E45756", label="80%")
    ax.axhline(0.90, linestyle="--", color="#59A14F", label="90%")
    ax.set_xlabel("Numero de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_title("Varianza explicada acumulada por PCA")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_varianza_acumulada.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pca_df, variance_df


def run(csv_path: Path | None = None, output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    parser = argparse.ArgumentParser(add_help=False)
    add_common_args(parser)
    default_output = parser.parse_args([])
    selected_output = (output_dir or default_output.output_dir).resolve()
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = load_dataset(csv_path)
    _, y, x_processed, feature_names = prepare_features(df)
    save_processed_feature_names(feature_names, selected_output)
    pca_df, variance_df = save_pca_outputs(x_processed, y, selected_output)
    print(f"CSV usado: {selected_csv}")
    print(f"PCA guardado en: {selected_output}")
    return pca_df, variance_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta PCA para la Task 2.")
    add_common_args(parser)
    args = parser.parse_args()
    run(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
