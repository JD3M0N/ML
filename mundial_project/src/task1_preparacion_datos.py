from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task1"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MIN_CONTINUOUS_UNIQUE_VALUES = 10
IQR_MULTIPLIER = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita, limpia y prepara los datos para la Task 1."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Ruta al CSV original. Por defecto usa mundial_project/data/foot.csv.",
    )
    parser.add_argument(
        "--target",
        default="condition",
        help="Columna objetivo que se conserva sin transformar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardan tablas e imagenes de Task 1.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Carpeta donde se guardan los datasets preparados.",
    )
    return parser.parse_args()


def classify_feature(series: pd.Series) -> str:
    unique_count = series.dropna().nunique()
    if unique_count == 2:
        return "binaria"
    if pd.api.types.is_numeric_dtype(series) and unique_count >= MIN_CONTINUOUS_UNIQUE_VALUES:
        return "continua"
    return "discreta"


def feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    features = [column for column in df.columns if column != target]
    if not features:
        raise ValueError("No hay features para analizar. Revisa el nombre de --target.")
    return features


def build_quality_audit(df: pd.DataFrame, target: str) -> pd.DataFrame:
    target_values = sorted(df[target].dropna().unique().tolist()) if target in df else []
    return pd.DataFrame(
        [
            {"metrica": "filas", "valor": len(df)},
            {"metrica": "columnas", "valor": len(df.columns)},
            {"metrica": "features", "valor": len(feature_columns(df, target))},
            {"metrica": "columna_objetivo", "valor": target},
            {"metrica": "valores_objetivo", "valor": str(target_values)},
            {"metrica": "valores_nulos_totales", "valor": int(df.isna().sum().sum())},
            {"metrica": "filas_duplicadas", "valor": int(df.duplicated().sum())},
        ]
    )


def outlier_count_iqr(series: pd.Series) -> int:
    if not pd.api.types.is_numeric_dtype(series):
        return 0
    clean_series = series.dropna()
    if clean_series.nunique() < MIN_CONTINUOUS_UNIQUE_VALUES:
        return 0
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - IQR_MULTIPLIER * iqr
    upper_limit = q3 + IQR_MULTIPLIER * iqr
    return int(((clean_series < lower_limit) | (clean_series > upper_limit)).sum())


def build_feature_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for feature in feature_columns(df, target):
        series = df[feature]
        numeric = pd.api.types.is_numeric_dtype(series)
        by_target_means = (
            df.groupby(target)[feature].mean().to_dict() if numeric and target in df else {}
        )
        mean_0 = by_target_means.get(0)
        mean_1 = by_target_means.get(1)
        rows.append(
            {
                "feature": feature,
                "tipo_dato": str(series.dtype),
                "tipo_feature": classify_feature(series),
                "valores_nulos": int(series.isna().sum()),
                "valores_unicos": int(series.nunique(dropna=True)),
                "minimo": series.min() if numeric else None,
                "maximo": series.max() if numeric else None,
                "media": series.mean() if numeric else None,
                "desviacion_std": series.std() if numeric else None,
                "outliers_iqr_reportados": outlier_count_iqr(series),
                "media_condition_0": mean_0,
                "media_condition_1": mean_1,
                "diferencia_media_1_menos_0": (
                    mean_1 - mean_0 if mean_0 is not None and mean_1 is not None else None
                ),
            }
        )
    return pd.DataFrame(rows)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop_duplicates().copy()
    for column in cleaned.columns:
        if not cleaned[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            mode = cleaned[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "desconocido"
            cleaned[column] = cleaned[column].fillna(fill_value)
    return cleaned


def build_model_ready_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    features = feature_columns(df, target)
    model_ready = df.copy()
    for feature in features:
        series = model_ready[feature]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        std = series.std()
        if std == 0 or pd.isna(std):
            model_ready[feature] = 0.0
        else:
            model_ready[feature] = (series - series.mean()) / std
    return model_ready


def save_table_image(table_df: pd.DataFrame, output_path: Path, title: str) -> None:
    display_df = table_df.copy()
    for column in display_df.select_dtypes(include="number").columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    fig_height = max(3.5, len(display_df) * 0.35 + 1.2)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
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


def save_condition_balance(df: pd.DataFrame, target: str, output_path: Path) -> None:
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", color=["#4C78A8", "#E45756"], edgecolor="black", ax=ax)
    ax.set_title("Distribucion de condition")
    ax.set_xlabel("condition")
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=0)
    for index, value in enumerate(counts):
        ax.text(index, value + 2, str(value), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_continuous_boxplots(
    df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    continuous_features = feature_summary.loc[
        feature_summary["tipo_feature"] == "continua", "feature"
    ].tolist()
    cols = 3
    rows = math.ceil(len(continuous_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 4.0))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, feature in zip(flat_axes, continuous_features):
        grouped = [
            df.loc[df[target] == target_value, feature].dropna()
            for target_value in sorted(df[target].dropna().unique())
        ]
        ax.boxplot(
            grouped,
            tick_labels=[str(value) for value in sorted(df[target].dropna().unique())],
        )
        ax.set_title(feature)
        ax.set_xlabel("condition")
        ax.set_ylabel("valor")
        ax.grid(axis="y", alpha=0.25)

    for ax in flat_axes[len(continuous_features) :]:
        ax.set_visible(False)
    fig.suptitle("Features continuas por condition", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_discrete_bars(
    df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    discrete_features = feature_summary.loc[
        feature_summary["tipo_feature"].isin(["binaria", "discreta"]), "feature"
    ].tolist()
    cols = 3
    rows = math.ceil(len(discrete_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, feature in zip(flat_axes, discrete_features):
        normalized = pd.crosstab(df[feature], df[target], normalize="columns") * 100
        normalized.plot(kind="bar", ax=ax, color=["#4C78A8", "#E45756"], edgecolor="black")
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("% dentro de condition")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="condition", fontsize=8)

    for ax in flat_axes[len(discrete_features) :]:
        ax.set_visible(False)
    fig.suptitle("Features discretas por condition", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    target = args.target.strip()

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"No existe la columna objetivo {target!r} en el CSV.")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    quality_audit = build_quality_audit(df, target)
    feature_summary = build_feature_summary(df, target)
    cleaned = clean_dataset(df)
    model_ready = build_model_ready_dataset(cleaned, target)

    quality_audit.to_csv(output_dir / "calidad_datos.csv", index=False)
    feature_summary.to_csv(output_dir / "resumen_features.csv", index=False)
    cleaned.to_csv(processed_dir / "foot_clean.csv", index=False)
    model_ready.to_csv(processed_dir / "foot_model_ready.csv", index=False)

    save_table_image(quality_audit, output_dir / "calidad_datos.png", "Auditoria de calidad")
    save_table_image(feature_summary, output_dir / "resumen_features.png", "Resumen de features")
    save_condition_balance(df, target, output_dir / "balance_condition.png")
    save_continuous_boxplots(
        df,
        feature_summary,
        target,
        output_dir / "features_continuas_por_condition.png",
    )
    save_discrete_bars(
        df,
        feature_summary,
        target,
        output_dir / "features_discretas_por_condition.png",
    )

    print(f"CSV analizado: {csv_path}")
    print(quality_audit.to_string(index=False))
    print(f"Resumen de features guardado en: {output_dir / 'resumen_features.csv'}")
    print(f"Dataset limpio guardado en: {processed_dir / 'foot_clean.csv'}")
    print(f"Dataset model-ready guardado en: {processed_dir / 'foot_model_ready.csv'}")


if __name__ == "__main__":
    main()
