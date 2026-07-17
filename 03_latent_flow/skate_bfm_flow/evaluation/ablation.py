from __future__ import annotations

import itertools


def config_matrix(matrix: dict[str, list]) -> list[list[str]]:
    keys = sorted(matrix)
    return [[f"{key}={value}" for key, value in zip(keys, values, strict=True)] for values in itertools.product(*(matrix[key] for key in keys))]
