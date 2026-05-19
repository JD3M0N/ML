from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "foot_clean.csv"
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "foot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task2"
TARGET = "condition"
CONTINUOUS_FEATURES = ["feature_5", "feature_6", "feature_10", "feature_11", "feature_13"]
CATEGORICAL_FEATURES = ["feature_2", "feature_3", "feature_8", "feature_9", "feature_12"]
BINARY_FEATURES = ["feature_1", "feature_4", "feature_7"]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV de entrada. Si se omite usa data/processed/foot_clean.csv o data/foot.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardan CSVs e imagenes.",
    )


def load_dataset(csv_path: Path | None) -> tuple[pd.DataFrame, Path]:
    selected_path = csv_path or (DEFAULT_CLEAN_CSV if DEFAULT_CLEAN_CSV.exists() else DEFAULT_RAW_CSV)
    df = pd.read_csv(selected_path)
    if TARGET not in df.columns:
        raise ValueError(f"No existe la columna objetivo {TARGET!r}.")
    return df, selected_path


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray, list[str]]:
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    preprocessor = build_preprocessor()
    x_processed = preprocessor.fit_transform(x)
    if hasattr(x_processed, "toarray"):
        x_processed = x_processed.toarray()
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x, y, np.asarray(x_processed), feature_names


def clustering_metrics(x_processed: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {
        "silhouette": silhouette_score(x_processed, labels),
        "davies_bouldin": davies_bouldin_score(x_processed, labels),
        "calinski_harabasz": calinski_harabasz_score(x_processed, labels),
    }


def save_processed_feature_names(feature_names: list[str], output_dir: Path) -> None:
    pd.Series(feature_names, name="feature_procesada").to_csv(
        output_dir / "features_procesadas.csv",
        index=False,
    )


def plot_clusters_on_pca(
    pca_df: pd.DataFrame,
    labels: np.ndarray,
    label_column: str,
    title: str,
    output_path: Path,
    palette: str = "Set1",
) -> None:
    import seaborn as sns

    plot_df = pca_df.copy()
    plot_df[label_column] = labels
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=label_column, palette=palette, ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
