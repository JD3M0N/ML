from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import task3_common


EXPERIMENT_NAME = "K-Nearest Neighbors"
EXPERIMENT_SLUG = "knn"


def build_estimator() -> object:
    classifier = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2,
    )
    return task3_common.make_supervised_pipeline(classifier)


def run(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    test_size: float = 0.2,
    n_splits: int = 5,
    n_repeats: int = 30,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict[str, Path | pd.DataFrame]:
    return task3_common.run_supervised_experiment(
        build_estimator(),
        EXPERIMENT_NAME,
        EXPERIMENT_SLUG,
        csv_path=csv_path,
        output_dir=output_dir,
        test_size=test_size,
        n_splits=n_splits,
        n_repeats=n_repeats,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: experimento de KNN.")
    task3_common.add_common_args(parser)
    args = parser.parse_args()
    run(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
    )


if __name__ == "__main__":
    main()
