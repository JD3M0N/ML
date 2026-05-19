from __future__ import annotations

import argparse
from pathlib import Path

from task2_agglomerative import run as run_agglomerative
from task2_common import DEFAULT_OUTPUT_DIR, add_common_args, load_dataset, prepare_features
from task2_dbscan import run as run_dbscan
from task2_kmeans import run as run_kmeans
from task2_pca import run as run_pca
from task2_reporte import write_summary_report
from task2_tsne import run as run_tsne


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta todos los scripts independientes de aprendizaje no supervisado."
    )
    add_common_args(parser)
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df, csv_path = load_dataset(args.csv)
    _, _, x_processed, _ = prepare_features(df)

    run_pca(args.csv, output_dir)
    _, best_kmeans_k = run_kmeans(args.csv, output_dir)
    _, best_agg_k = run_agglomerative(args.csv, output_dir)
    run_dbscan(args.csv, output_dir)
    run_tsne(args.csv, output_dir)
    report_path = write_summary_report(output_dir, csv_path)

    print(f"CSV usado: {csv_path}")
    print(f"X procesado: {x_processed.shape}")
    print(f"Mejor K-Means k={best_kmeans_k}")
    print(f"Mejor jerarquico k={best_agg_k}")
    print(f"Resultados guardados en: {output_dir}")
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
