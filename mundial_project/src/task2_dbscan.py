from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from task2_common import add_common_args, clustering_metrics, load_dataset, plot_clusters_on_pca, prepare_features
from task2_pca import save_pca_outputs


def save_dbscan_outputs(x_processed, pca_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    best_labels = None
    best_silhouette = -np.inf

    for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for min_samples in [3, 5, 10]:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(x_processed)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            metrics = (
                clustering_metrics(x_processed, labels)
                if n_clusters > 1
                else {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
            )
            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    **metrics,
                }
            )
            if metrics["silhouette"] is not None and metrics["silhouette"] > best_silhouette:
                best_silhouette = metrics["silhouette"]
                best_labels = labels

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "dbscan_metricas.csv", index=False)
    if best_labels is None:
        best_labels = DBSCAN(eps=3.0, min_samples=3).fit_predict(x_processed)
    plot_clusters_on_pca(
        pca_df,
        best_labels,
        "cluster_dbscan",
        "DBSCAN visualizado con PCA",
        output_dir / "dbscan_clusters_pca.png",
        palette="tab10",
    )
    return results_df


def run(csv_path: Path | None = None, output_dir: Path | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(add_help=False)
    add_common_args(parser)
    default_output = parser.parse_args([])
    selected_output = (output_dir or default_output.output_dir).resolve()
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = load_dataset(csv_path)
    _, y, x_processed, _ = prepare_features(df)
    pca_df, _ = save_pca_outputs(x_processed, y, selected_output)
    results_df = save_dbscan_outputs(x_processed, pca_df, selected_output)
    print(f"CSV usado: {selected_csv}")
    print(f"DBSCAN guardado en: {selected_output}")
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta DBSCAN para la Task 2.")
    add_common_args(parser)
    args = parser.parse_args()
    run(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
