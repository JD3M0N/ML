from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from task2_common import TARGET, add_common_args, load_dataset, prepare_features


def save_tsne_outputs(x_processed, y: pd.Series, kmeans_labels, output_dir: Path) -> pd.DataFrame:
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    x_tsne = tsne.fit_transform(x_processed)
    tsne_df = pd.DataFrame(
        {
            "TSNE1": x_tsne[:, 0],
            "TSNE2": x_tsne[:, 1],
            TARGET: y.values,
            "cluster_kmeans": kmeans_labels,
        }
    )
    tsne_df.to_csv(output_dir / "tsne_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue=TARGET, palette="Set2", ax=ax)
    ax.set_title("Visualizacion t-SNE coloreada por condition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_condition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="cluster_kmeans", palette="Set1", ax=ax)
    ax.set_title("Visualizacion t-SNE coloreada por cluster K-Means")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_kmeans.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue=TARGET, palette="Set2", ax=axes[0])
    axes[0].set_title("t-SNE coloreado por condition")
    axes[0].grid(alpha=0.25)
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="cluster_kmeans", palette="Set1", ax=axes[1])
    axes[1].set_title("t-SNE coloreado por K-Means")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Comparacion visual t-SNE: condition vs clusters", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_condition_kmeans.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return tsne_df


def run(csv_path: Path | None = None, output_dir: Path | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(add_help=False)
    add_common_args(parser)
    default_output = parser.parse_args([])
    selected_output = (output_dir or default_output.output_dir).resolve()
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = load_dataset(csv_path)
    _, y, x_processed, _ = prepare_features(df)
    kmeans_labels = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(x_processed)
    tsne_df = save_tsne_outputs(x_processed, y, kmeans_labels, selected_output)
    print(f"CSV usado: {selected_csv}")
    print(f"t-SNE guardado en: {selected_output}")
    return tsne_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta t-SNE para la Task 2.")
    add_common_args(parser)
    args = parser.parse_args()
    run(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
