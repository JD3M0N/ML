from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from task2_common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, TARGET


def write_summary_report(output_dir: Path, csv_path: Path | None = None) -> Path:
    pca_variance = pd.read_csv(output_dir / "pca_varianza.csv")
    kmeans_results = pd.read_csv(output_dir / "kmeans_metricas.csv")
    kmeans_extended = pd.read_csv(output_dir / "kmeans_metricas_extendido.csv")
    kmeans_k235 = pd.read_csv(output_dir / "kmeans_comparacion_k_2_3_5.csv")
    agg_results = pd.read_csv(output_dir / "agglomerative_metricas.csv")
    agg_disparity = pd.read_csv(output_dir / "agglomerative_disparidad.csv")
    dbscan_results = pd.read_csv(output_dir / "dbscan_metricas.csv")
    profile = pd.read_csv(output_dir / "kmeans_perfil_clusters.csv", index_col=0)
    kmeans_pct = pd.read_csv(output_dir / "kmeans_condition_crosstab_pct.csv", index_col=0)

    best_kmeans = kmeans_results.sort_values("silhouette", ascending=False).iloc[0]
    best_agg = agg_results.sort_values("silhouette", ascending=False).iloc[0]
    dbscan_two_cluster = dbscan_results.loc[dbscan_results["n_clusters"] == 2].sort_values("n_noise").head(1)

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
        "El metodo del codo extendido se evaluo para k=1..15. La inercia baja al aumentar k, como es esperable, pero no aparece una rodilla posterior que justifique una segmentacion mucho mas fina.",
        "Se conserva k=2 como particion principal porque separa dos perfiles generales, mantiene una interpretacion clara y se relaciona de forma marcada con `condition` en el cruce posterior.",
        "Tambien se compararon k=3 y k=5 como lecturas mas granulares; generan subgrupos mas especificos, pero fragmentan la muestra y se dejan como exploracion secundaria.",
        "Los perfiles de clusters y el cruce posterior con `condition` se guardaron para interpretar que variables caracterizan cada grupo.",
        f"Las variables con mayor diferencia media entre clusters fueron: {', '.join(top_features)}.",
        f"En el cruce posterior, el cluster 0 contiene {kmeans_pct.loc[0, '0']:.1f}% de pacientes con `condition=0`, mientras que el cluster 1 contiene {kmeans_pct.loc[1, '1']:.1f}% de pacientes con `condition=1`.",
        "",
        "## Clustering jerarquico",
        "",
        f"El analisis de disparidad del dendrograma muestra su mayor salto de altura en la fusion {int(agg_disparity.iloc[0]['fusion'])}, con salto={agg_disparity.iloc[0]['salto_altura']:.4f}.",
        "Ese salto ocurre al final del proceso, antes de fusionar los dos macrogrupos restantes, lo que apoya una lectura de dos grupos principales.",
        "Esta tecnica sirve como contraste frente a K-Means porque llega a una lectura global compatible desde otra familia de clustering.",
        "",
        "## DBSCAN",
        "",
    ]
    if dbscan_two_cluster.empty:
        lines.append("DBSCAN no encontro una configuracion limpia con dos grupos; esto sugiere que los datos no forman regiones densas claramente separadas.")
    else:
        row = dbscan_two_cluster.iloc[0]
        lines.append(
            f"La configuracion con dos clusters y menos ruido fue eps={row['eps']} y min_samples={int(row['min_samples'])}, pero aun conserva {int(row['n_noise'])} puntos de ruido."
        )
    lines.extend(
        [
            "",
            "## Trabajo extra: silhouette y t-SNE",
            "",
            f"Silhouette se conserva como metrica complementaria: en la evaluacion base el mayor valor fue k={int(best_kmeans['k'])}, con silhouette={best_kmeans['silhouette']:.4f}.",
            f"En clustering jerarquico, la mejor lectura complementaria por silhouette fue k={int(best_agg['k'])}, con silhouette={best_agg['silhouette']:.4f}.",
            "t-SNE se conserva como visualizacion adicional para mirar vecindarios locales, pero no se usa como criterio principal de seleccion.",
            "",
            "## Conclusion",
            "",
            "Los experimentos sugieren una estructura latente parcialmente relacionada con `condition`, especialmente al combinar K-Means con k=2, el metodo del codo, la disparidad jerarquica y el cruce posterior con la variable objetivo. Los clusters son perfiles exploratorios de pacientes y no diagnosticos.",
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
