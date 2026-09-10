from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Sequence

import numpy as np
from threadpoolctl import threadpool_limits


class BalancedPartitionError(ValueError):
    pass


@dataclass(frozen=True)
class SplitAudit:
    path: str
    rows: int
    target_leaves: int
    left_rows: int
    right_rows: int
    left_target_leaves: int
    right_target_leaves: int
    axis_mode: str
    pivot_feature: int
    leading_eigenvalue_hex: str
    relative_eigengap_hex: str
    axis_digest: str
    member_ids_digest: str
    left_child_path: str
    right_child_path: str
    left_boundary_hex: str
    right_boundary_hex: str
    boundary_margin_hex: str


@dataclass(frozen=True)
class BalancedPartition:
    labels: np.ndarray
    leaf_paths: tuple[str, ...]
    leaf_sizes: tuple[int, ...]
    splits: tuple[SplitAudit, ...]


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def id_digest(ids: Sequence[str]) -> str:
    payload = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def midrank_transform(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Map per-column exact-tie midranks to [-1, 1]; constants become zero."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2 or not np.isfinite(source).all():
        raise BalancedPartitionError("rank-transform input must be a finite 2D array with two rows")
    rows, columns = source.shape
    transformed = np.zeros((rows, columns), dtype=np.float64)
    doubled_midranks = np.empty((rows, columns), dtype=np.int64)
    constant: list[int] = []
    for column in range(columns):
        data = source[:, column]
        order = np.argsort(data, kind="stable")
        ordered = data[order]
        if ordered[0] == ordered[-1]:
            constant.append(column)
            doubled_midranks[:, column] = rows + 1
            continue
        start = 0
        while start < rows:
            stop = start + 1
            while stop < rows and ordered[stop] == ordered[start]:
                stop += 1
            doubled_midrank = start + stop + 1
            doubled_midranks[order[start:stop], column] = doubled_midrank
            transformed[order[start:stop], column] = (
                doubled_midrank - (rows + 1)
            ) / (rows - 1)
            start = stop
    transformed[transformed == 0.0] = 0.0
    return transformed, doubled_midranks, tuple(constant)


def _oriented_axis(
    centered: np.ndarray, integer_scores: np.ndarray, *, relative_eigengap_tolerance: float,
) -> tuple[np.ndarray, str, int, float, float]:
    scatter = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    except np.linalg.LinAlgError as error:
        raise BalancedPartitionError("symmetric eigensolver did not converge") from error
    leading = float(eigenvalues[-1])
    second = float(eigenvalues[-2]) if len(eigenvalues) > 1 else 0.0
    relative_eigengap = (leading - second) / leading if leading > 0.0 else 0.0
    if not np.isfinite(leading) or leading <= 0.0:
        raise BalancedPartitionError("node has no non-constant structure feature")
    if relative_eigengap <= relative_eigengap_tolerance:
        rows = len(integer_scores)
        exact_ss = []
        for column in range(integer_scores.shape[1]):
            data = integer_scores[:, column]
            total = sum(map(int, data))
            squares = sum(int(value) * int(value) for value in data)
            exact_ss.append(rows * squares - total * total)
        feature = max(range(len(exact_ss)), key=lambda column: (exact_ss[column], -column))
        if exact_ss[feature] <= 0:
            raise BalancedPartitionError("node fallback has no non-constant feature")
        axis = np.zeros(centered.shape[1], dtype=np.float64)
        axis[feature] = 1.0
        return axis, "exact_variance_fallback", feature, leading, relative_eigengap
    axis = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    if not np.isfinite(axis).all():
        raise BalancedPartitionError("leading eigenvector is non-finite")
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
    return axis, "leading_pca", pivot, leading, relative_eigengap


def balanced_recursive_partition(
    transformed: np.ndarray,
    doubled_midranks: np.ndarray,
    ids: Sequence[str],
    *,
    leaves: int,
    relative_eigengap_tolerance: float = 1e-10,
    boundary_tolerance: float = 1e-10,
) -> BalancedPartition:
    values = np.asarray(transformed, dtype=np.float64)
    integer_values = np.asarray(doubled_midranks)
    if values.ndim != 2 or len(values) != len(ids) or len(set(ids)) != len(ids):
        raise BalancedPartitionError("partition inputs differ")
    if not np.isfinite(values).all():
        raise BalancedPartitionError("partition vectors must be finite")
    if integer_values.shape != values.shape \
            or not np.issubdtype(integer_values.dtype, np.integer):
        raise BalancedPartitionError("integer midrank matrix differs")
    if not 1 <= leaves <= len(values) \
            or relative_eigengap_tolerance < 0.0 or boundary_tolerance < 0.0:
        raise BalancedPartitionError("partition limits differ")
    _ranked, reconstructed_integers, _constant = midrank_transform(values)
    if not np.array_equal(_ranked, values) \
            or not np.array_equal(reconstructed_integers, integer_values):
        raise BalancedPartitionError("partition input is not the canonical midrank transform")

    leaf_members: list[tuple[str, tuple[int, ...]]] = []
    splits: list[SplitAudit] = []

    def recurse(members: tuple[int, ...], target: int, path: str) -> None:
        if target == 1:
            leaf_members.append((path or "root", members))
            return
        rows = len(members)
        if rows < target:
            raise BalancedPartitionError("node has fewer rows than target leaves")
        left_target = target // 2
        right_target = target - left_target
        left_rows = rows * left_target // target
        if left_rows < left_target or rows - left_rows < right_target:
            raise BalancedPartitionError("balanced allocation cannot populate every leaf")
        node = values[np.asarray(members, dtype=np.int64)]
        node_integer = integer_values[np.asarray(members, dtype=np.int64)]
        centered = node - np.mean(node, axis=0)
        with threadpool_limits(limits=1, user_api="blas"):
            axis, mode, pivot, leading, relative_gap = _oriented_axis(
                centered, node_integer,
                relative_eigengap_tolerance=relative_eigengap_tolerance,
            )
        projection = np.asarray([
            math.fsum(float(row[column]) * float(axis[column]) for column in range(len(axis)))
            for row in centered
        ], dtype=np.float64)
        projection[projection == 0.0] = 0.0
        ordered = tuple(sorted(
            range(rows), key=lambda index: (float(projection[index]), ids[members[index]]),
        ))
        left = tuple(members[index] for index in ordered[:left_rows])
        right = tuple(members[index] for index in ordered[left_rows:])
        left_boundary = float(projection[ordered[left_rows - 1]])
        right_boundary = float(projection[ordered[left_rows]])
        margin = right_boundary - left_boundary
        scale = max(1.0, float(np.max(np.abs(projection))))
        if left_boundary.hex() != right_boundary.hex() \
                and not margin > boundary_tolerance * scale:
            raise BalancedPartitionError("projection split boundary is numerically unresolved")
        left_path = f"{path}0"
        right_path = f"{path}1"
        splits.append(SplitAudit(
            path=path or "root", rows=rows, target_leaves=target,
            left_rows=len(left), right_rows=len(right),
            left_target_leaves=left_target, right_target_leaves=right_target,
            axis_mode=mode, pivot_feature=pivot,
            leading_eigenvalue_hex=leading.hex(),
            relative_eigengap_hex=relative_gap.hex(), axis_digest=array_digest(axis),
            member_ids_digest=id_digest([ids[index] for index in members]),
            left_child_path=left_path, right_child_path=right_path,
            left_boundary_hex=left_boundary.hex(), right_boundary_hex=right_boundary.hex(),
            boundary_margin_hex=margin.hex(),
        ))
        recurse(left, left_target, left_path)
        recurse(right, right_target, right_path)

    recurse(tuple(range(len(values))), leaves, "")
    labels = np.full(len(values), -1, dtype=np.int32)
    paths: list[str] = []
    sizes: list[int] = []
    for label, (path, members) in enumerate(leaf_members):
        labels[np.asarray(members, dtype=np.int64)] = label
        paths.append(path)
        sizes.append(len(members))
    if np.any(labels < 0) or len(set(map(int, labels))) != leaves:
        raise BalancedPartitionError("partition did not assign exactly the requested leaves")
    floor_size = len(values) // leaves
    ceil_size = (len(values) + leaves - 1) // leaves
    if any(size not in {floor_size, ceil_size} for size in sizes):
        raise BalancedPartitionError("leaf sizes are not globally balanced")
    return BalancedPartition(labels, tuple(paths), tuple(sizes), tuple(splits))
