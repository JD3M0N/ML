from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task2_common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, TARGET


def write_summary_report(output_dir: Path, csv_path: Path | None = None) -> Path:
    pca_variance = pd.read_csv(output_dir / "pca_varianza.csv")
    kmeans_results = pd.read_csv(output_dir / "kmeans_metricas.csv")
    agg_results = pd.read_csv(output_dir / "agglomerative_metricas.csv")
    dbscan_results = pd.read_csv(output_dir / "dbscan_metricas.csv")
    profile = pd.read_csv(output_dir / "kmeans_perfil_clusters.csv", index_col=0)
    kmeans_pct = pd.read_csv(output_dir / "kmeans_condition_crosstab_pct.csv", index_col=0)

    best_kmeans = kmeans_results.sort_values("silhouette", ascending=False).iloc[0]
    best_agg = agg_results.sort_values("silhouette", ascending=False).iloc[0]
    valid_dbscan = dbscan_results.dropna(subset=["silhouette"])
    best_dbscan = valid_dbscan.sort_values("silhouette", ascending=False).head(1)

    pc2_total = pca_variance.loc[1, "varianza_acumulada"]
    components_80 = int((pca_variance["varianza_acumulada"] >= 0.80).idxmax() + 1)
    components_90 = int((pca_variance["varianza_acumulada"] >= 0.90).idxmax() + 1)

    difference_column = "diferencia_cluster_1_menos_0"
    top_features = []
    if difference_column in profile.columns:
        top_features = (
            profile.drop(index=[TARGET], errors="ignore")[difference_column]
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

    source = csv_path or (PROJECT_ROOT / "data" / "processed" / "foot_clean.csv")
    try:
        source_text = source.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_text = str(source)

    lines = [
        "# Task 2 - Aprendizaje No Supervisado",
        "",
        "## Objetivo",
        "",
        "El objetivo fue explorar estructuras ocultas en los pacientes sin usar `condition` para entrenar los modelos no supervisados. La variable objetivo se uso solo despues, para interpretar si los grupos descubiertos se relacionan con la condicion medica.",
        "",
        "## Preparacion",
        "",
        f"Fuente de datos usada: `{source_text}`.",
        "Las variables continuas se estandarizaron, las discretas se codificaron con one-hot encoding y las binarias se conservaron sin cambios.",
        "",
        "## PCA",
        "",
        f"Las dos primeras componentes explicaron {pc2_total * 100:.2f}% de la varianza acumulada.",
        f"Se necesitaron {components_80} componentes para alcanzar al menos 80% de varianza y {components_90} componentes para alcanzar al menos 90%.",
        "",
        "## K-Means",
        "",
        f"El mejor valor evaluado por silhouette fue k={int(best_kmeans['k'])}, con silhouette={best_kmeans['silhouette']:.4f}, Davies-Bouldin={best_kmeans['davies_bouldin']:.4f} y Calinski-Harabasz={best_kmeans['calinski_harabasz']:.4f}.",
        "Los perfiles de clusters y el cruce posterior con `condition` se guardaron para interpretar que variables caracterizan cada grupo.",
        f"Las variables con mayor diferencia media entre clusters fueron: {', '.join(top_features)}.",
        f"En el cruce posterior, el cluster 0 contiene {kmeans_pct.loc[0, '0']:.1f}% de pacientes con `condition=0`, mientras que el cluster 1 contiene {kmeans_pct.loc[1, '1']:.1f}% de pacientes con `condition=1`.",
        "",
        "## Clustering jerarquico",
        "",
        f"El mejor valor evaluado fue k={int(best_agg['k'])}, con silhouette={best_agg['silhouette']:.4f}. Esta tecnica sirve como contraste frente a K-Means.",
        "",
        "## DBSCAN",
        "",
    ]
    if best_dbscan.empty:
        lines.append("DBSCAN no encontro una configuracion con mas de un grupo evaluable por silhouette; esto sugiere que los datos no forman grupos densos claramente separados.")
    else:
        row = best_dbscan.iloc[0]
        lines.append(
            f"La mejor configuracion evaluable fue eps={row['eps']} y min_samples={int(row['min_samples'])}, con {int(row['n_clusters'])} clusters, {int(row['n_noise'])} puntos de ruido y silhouette={row['silhouette']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Los experimentos sugieren una estructura latente parcialmente relacionada con `condition`, especialmente al comparar los clusters de K-Means con la variable objetivo despues del agrupamiento. Aun asi, las metricas internas deben interpretarse con cautela: los clusters son perfiles exploratorios de pacientes y no diagnosticos.",
        ]
    )

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "task2_aprendizaje_no_supervisado.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el reporte de Task 2 desde outputs/task2.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    report_path = write_summary_report(args.output_dir.resolve(), args.csv)
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
