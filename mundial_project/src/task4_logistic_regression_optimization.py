from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

import task3_common
import task4_common


EXPERIMENT_NAME = "Regresion Logistica Optimizada"
EXPERIMENT_SLUG = "logistic_regression"


def build_estimator() -> object:
    classifier = LogisticRegression(max_iter=3000, random_state=task3_common.RANDOM_STATE)
    return task3_common.make_supervised_pipeline(classifier)


def param_grid() -> list[dict[str, object]]:
    return [
        {
            "model__solver": ["lbfgs"],
            "model__penalty": ["l2"],
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model__solver": ["liblinear"],
            "model__penalty": ["l1", "l2"],
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"],
        },
    ]


def run(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_repeats: int = 10,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    n_jobs: int = 1,
    permutation_repeats: int = 30,
) -> dict[str, Path | pd.DataFrame | object]:
    return task4_common.run_optimization_experiment(
        build_estimator(),
        param_grid(),
        EXPERIMENT_NAME,
        EXPERIMENT_SLUG,
        csv_path=csv_path,
        output_dir=output_dir,
        test_size=test_size,
        n_splits=n_splits,
        n_repeats=n_repeats,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        n_jobs=n_jobs,
        permutation_repeats=permutation_repeats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: optimizacion de Regresion Logistica.")
    task4_common.add_task4_args(parser)
    args = parser.parse_args()
    run(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
        n_jobs=args.n_jobs,
        permutation_repeats=args.permutation_repeats,
    )


if __name__ == "__main__":
    main()
