from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FusionResult:
    """A deterministic candidate pool assembled from independent distance views."""

    selected: pd.DataFrame
    rankings: pd.DataFrame


def preselect_per_group(
    candidates: pd.DataFrame,
    view_columns: tuple[str, ...],
    *,
    group_column: str = "symbol",
    id_column: str = "episode_id",
    per_view: int = 5,
) -> pd.DataFrame:
    """Union each view's local top candidates, matching the streaming scan bound."""
    if per_view < 1:
        raise ValueError("per_view must be positive")
    required = {group_column, id_column, *view_columns}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"missing preselection columns: {sorted(missing)}")
    selected_ids: set[str] = set()
    for _, group in candidates.groupby(group_column, sort=True):
        ids = group[id_column].astype(str).to_numpy()
        for view in view_columns:
            distances = pd.to_numeric(group[view], errors="coerce").to_numpy(dtype=float)
            distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf, neginf=np.inf)
            order = np.lexsort((ids, distances))[:per_view]
            selected_ids.update(ids[order])
    return candidates[candidates[id_column].astype(str).isin(selected_ids)].copy()


def reciprocal_rank_fusion(
    candidates: pd.DataFrame,
    view_columns: tuple[str, ...],
    *,
    pool_size: int,
    id_column: str = "episode_id",
    rank_constant: float = 60.0,
    minimum_per_view: int = 1,
) -> FusionResult:
    """Fuse smaller-is-better views without mixing their incompatible scales.

    Each view is converted to a stable rank and contributes ``1 / (k + rank)``.
    At least ``minimum_per_view`` candidates from every view are retained before
    the remaining pool is filled by the aggregate score.  Ties always resolve by
    the stable candidate identifier, so input ordering cannot affect the result.
    """
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    if minimum_per_view < 0:
        raise ValueError("minimum_per_view must be non-negative")
    if not view_columns:
        raise ValueError("at least one view is required")
    required = {id_column, *view_columns}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"missing fusion columns: {sorted(missing)}")
    if candidates[id_column].astype(str).duplicated().any():
        raise ValueError("candidate identifiers must be unique")
    if not len(candidates):
        empty = candidates.copy()
        empty["fusion_score"] = pd.Series(dtype=float)
        empty["fusion_rank"] = pd.Series(dtype=int)
        return FusionResult(empty, empty)

    work = candidates.copy().reset_index(drop=True)
    ids = work[id_column].astype(str).to_numpy()
    fusion_score = np.zeros(len(work), dtype=float)
    forced: set[int] = set()
    for view in view_columns:
        distances = pd.to_numeric(work[view], errors="coerce").to_numpy(dtype=float)
        distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf, neginf=np.inf)
        order = np.lexsort((ids, distances))
        ranks = np.empty(len(work), dtype=int)
        ranks[order] = np.arange(1, len(work) + 1)
        work[f"rank_{view}"] = ranks
        fusion_score += 1.0 / (rank_constant + ranks)
        forced.update(int(position) for position in order[:minimum_per_view])

    work["fusion_score"] = fusion_score
    fused_order = np.lexsort((ids, -fusion_score))
    limit = min(pool_size, len(work))
    # If the mandatory union is ever larger than the requested pool, the same
    # deterministic fused ordering decides which mandatory candidates survive.
    selected_positions = sorted(forced, key=lambda i: (-fusion_score[i], ids[i]))[:limit]
    already = set(selected_positions)
    selected_positions.extend(
        int(position) for position in fused_order
        if position not in already and len(selected_positions) < limit
    )

    fusion_rank = np.empty(len(work), dtype=int)
    fusion_rank[fused_order] = np.arange(1, len(work) + 1)
    work["fusion_rank"] = fusion_rank
    selected = work.iloc[selected_positions].copy().reset_index(drop=True)
    selected["pool_rank"] = np.arange(1, len(selected) + 1)
    return FusionResult(selected, work)


def oracle_pool_recall(
    ranking: pd.DataFrame,
    view_columns: tuple[str, ...],
    pool_sizes: tuple[int, ...],
    *,
    selected_column: str = "oracle_selected",
) -> dict[str, float]:
    """Measure fused-pool coverage of the exact, overlap-deduplicated oracle."""
    if selected_column not in ranking:
        raise ValueError(f"missing oracle membership column: {selected_column}")
    target = set(ranking.loc[ranking[selected_column], "episode_id"].astype(str))
    recalls: dict[str, float] = {}
    for pool_size in sorted(set(pool_sizes)):
        fused = reciprocal_rank_fusion(
            ranking, view_columns, pool_size=pool_size,
        ).selected
        found = set(fused.episode_id.astype(str)).intersection(target)
        recalls[str(pool_size)] = len(found) / len(target) if target else 0.0
    return recalls
