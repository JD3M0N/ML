from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from task2_common import (
    TARGET,
    add_common_args,
    clustering_metrics,
    load_dataset,
    plot_clusters_on_pca,
    prepare_features,
)
from task2_pca import save_pca_outputs


def save_kmeans_outputs(
    df: pd.DataFrame,
    x_processed,
    y: pd.Series,
    pca_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, int]:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x_processed)
        rows.append({"k": k, "inertia": model.inertia_, **clustering_metrics(x_processed, labels)})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "kmeans_metricas.csv", index=False)
    best_k = int(results_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

    best_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = best_model.fit_predict(x_processed)
    cluster_df = df.copy()
    cluster_df["cluster_kmeans"] = labels
    cluster_df.to_csv(output_dir / "kmeans_clusters.csv", index=False)

    profile = cluster_df.groupby("cluster_kmeans").mean(numeric_only=True).T
    if profile.shape[1] == 2:
        profile["diferencia_cluster_1_menos_0"] = profile.iloc[:, 1] - profile.iloc[:, 0]
    profile.to_csv(output_dir / "kmeans_perfil_clusters.csv")

    pd.crosstab(cluster_df["cluster_kmeans"], cluster_df[TARGET]).to_csv(
        output_dir / "kmeans_condition_crosstab.csv"
    )
    (pd.crosstab(cluster_df["cluster_kmeans"], cluster_df[TARGET], normalize="index") * 100).to_csv(
        output_dir / "kmeans_condition_crosstab_pct.csv"
    )
    pd.DataFrame(
        [
            {
                "modelo": f"KMeans_k{best_k}",
                "adjusted_rand_index": adjusted_rand_score(y, labels),
                "normalized_mutual_information": normalized_mutual_info_score(y, labels),
            }
        ]
    ).to_csv(output_dir / "kmeans_metricas_externas_condition.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df["inertia"], marker="o", color="#4C78A8")
    ax.set_xlabel("Numero de clusters k")
    ax.set_ylabel("Inercia")
    ax.set_title("Metodo del codo para K-Means")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_elbow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df["silhouette"], marker="o", color="#E45756")
    ax.set_xlabel("Numero de clusters k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score para K-Means")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "kmeans_silhouette.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_clusters_on_pca(
        pca_df,
        labels,
        "cluster_kmeans",
        f"Clusters K-Means k={best_k} visualizados con PCA",
        output_dir / "kmeans_clusters_pca.png",
    )
    return results_df, best_k


def run(csv_path: Path | None = None, output_dir: Path | None = None) -> tuple[pd.DataFrame, int]:
    parser = argparse.ArgumentParser(add_help=False)
    add_common_args(parser)
    default_output = parser.parse_args([])
    selected_output = (output_dir or default_output.output_dir).resolve()
    selected_output.mkdir(parents=True, exist_ok=True)

    df, selected_csv = load_dataset(csv_path)
    _, y, x_processed, _ = prepare_features(df)
    pca_df, _ = save_pca_outputs(x_processed, y, selected_output)
    results_df, best_k = save_kmeans_outputs(df, x_processed, y, pca_df, selected_output)
    print(f"CSV usado: {selected_csv}")
    print(f"Mejor K-Means k={best_k}")
    return results_df, best_k


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta K-Means para la Task 2.")
    add_common_args(parser)
    args = parser.parse_args()
    run(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
