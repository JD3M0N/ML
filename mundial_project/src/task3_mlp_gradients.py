from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import task3_common
import task3_mlp


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def relu_derivative(values: np.ndarray) -> np.ndarray:
    return (values > 0).astype(float)


def transformed_to_original_feature(feature_name: str) -> str:
    raw = feature_name.split("__", 1)[-1]
    for feature in (
        task3_common.CONTINUOUS_FEATURES
        + task3_common.CATEGORICAL_FEATURES
        + task3_common.BINARY_FEATURES
    ):
        if raw == feature or raw.startswith(f"{feature}_"):
            return feature
    return raw


def get_feature_names(model: object) -> list[str]:
    return list(model.named_steps["preprocess"].get_feature_names_out())


def input_gradients_for_mlp(model: object, x: pd.DataFrame) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    mlp = model.named_steps["model"]
    transformed = preprocess.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)

    if mlp.out_activation_ != "logistic":
        raise ValueError(f"Se esperaba salida logistica binaria, pero MLP usa {mlp.out_activation_!r}.")

    rows = []
    feature_names = get_feature_names(model)
    for sample in transformed:
        activations = [sample]
        pre_activations = []
        current = sample
        for weights, bias in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
            z = current @ weights + bias
            pre_activations.append(z)
            current = np.maximum(z, 0.0)
            activations.append(current)

        output_logit = current @ mlp.coefs_[-1] + mlp.intercepts_[-1]
        output_prob = sigmoid(output_logit)
        grad = (output_prob * (1.0 - output_prob)) @ mlp.coefs_[-1].T

        for layer_idx in range(len(mlp.coefs_) - 2, -1, -1):
            grad = grad * relu_derivative(pre_activations[layer_idx])
            grad = grad @ mlp.coefs_[layer_idx].T

        rows.append(np.abs(grad))

    gradient_matrix = np.vstack(rows)
    detailed = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "feature": [transformed_to_original_feature(name) for name in feature_names],
            "mean_abs_gradient": gradient_matrix.mean(axis=0),
            "std_abs_gradient": gradient_matrix.std(axis=0, ddof=1),
        }
    )
    return (
        detailed.groupby("feature", as_index=False)
        .agg(
            importance_mean=("mean_abs_gradient", "mean"),
            importance_max=("mean_abs_gradient", "max"),
            importance_std=("std_abs_gradient", "mean"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_gradient_plot(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_df = df.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance_mean"], color="#4C78A8")
    ax.set_title("MLP - Importancia por gradientes de entrada")
    ax.set_xlabel("Gradiente absoluto medio")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = 0.2,
) -> pd.DataFrame:
    selected_output = (output_dir or task3_common.DEFAULT_OUTPUT_DIR).resolve() / task3_mlp.EXPERIMENT_SLUG
    selected_output.mkdir(parents=True, exist_ok=True)

    df, _ = task3_common.load_dataset(csv_path)
    x, y = task3_common.split_features_target(df)
    x_train, x_test, y_train, _ = task3_common.stratified_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=task3_common.RANDOM_STATE,
    )
    model = task3_mlp.build_estimator()
    model.fit(x_train, y_train)
    gradients = input_gradients_for_mlp(model, x_test)
    gradients.to_csv(selected_output / "gradient_feature_importance.csv", index=False)
    save_gradient_plot(gradients, selected_output / "gradient_feature_importance.png")
    print("Importancia por gradientes del MLP:")
    print(gradients.to_string(index=False))
    return gradients


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: importancia por gradientes del MLP.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=task3_common.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=task3_common.DEFAULT_TEST_SIZE)
    args = parser.parse_args()
    run(csv_path=args.csv, output_dir=args.output_dir, test_size=args.test_size)


if __name__ == "__main__":
    main()
