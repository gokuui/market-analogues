"""Deterministic primitives for descriptive fixed-neighbor future-path modes."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any, Mapping, Sequence


class FutureModeError(ValueError):
    """Raised when a future-mode input violates the frozen contract."""


@dataclass(frozen=True)
class Member:
    match_rank: int
    episode_id: str
    symbol: str
    cutoff: str
    source_fingerprint: str

    @property
    def key(self) -> tuple[int, str]:
        return self.match_rank, self.episode_id


@dataclass(frozen=True)
class Exclusion:
    member: Member
    reason: str


@dataclass(frozen=True)
class PreparedPath:
    member: Member
    values: tuple[float, ...]
    timestamps: tuple[str, ...]


@dataclass(frozen=True)
class PamResult:
    medoid_indices: tuple[int, ...]
    labels: tuple[int, ...]
    objective: float


def _member(value: Mapping[str, Any]) -> Member:
    try:
        member = Member(
            match_rank=int(value["match_rank"]),
            episode_id=str(value["matched_episode_id"]),
            symbol=str(value["matched_symbol"]),
            cutoff=str(value["matched_cutoff"]),
            source_fingerprint=str(value["source_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FutureModeError("invalid member identity") from error
    if member.match_rank < 1 or not all((
        member.episode_id, member.symbol, member.cutoff, member.source_fingerprint,
    )):
        raise FutureModeError("invalid member identity")
    return member


def select_primary_members(
    query_symbol: str, raw_members: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Member, ...], tuple[Exclusion, ...]]:
    """Apply only outcome-blind same/duplicate-symbol dependence controls."""
    members = sorted((_member(value) for value in raw_members), key=lambda value: value.key)
    ranks = [value.match_rank for value in members]
    episodes = [value.episode_id for value in members]
    if len(set(ranks)) != len(ranks):
        raise FutureModeError("duplicate match rank")
    if len(set(episodes)) != len(episodes):
        raise FutureModeError("duplicate matched episode")
    primary: list[Member] = []
    exclusions: list[Exclusion] = []
    seen_symbols: set[str] = set()
    for member in members:
        if member.symbol == query_symbol:
            exclusions.append(Exclusion(member, "query_symbol_memory"))
        elif member.symbol in seen_symbols:
            exclusions.append(Exclusion(member, "duplicate_matched_symbol"))
        else:
            primary.append(member)
            seen_symbols.add(member.symbol)
    return tuple(primary), tuple(exclusions)


def prepare_paths(
    members: Sequence[Member],
    path_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    value_field: str,
    horizon: int = 60,
) -> tuple[tuple[PreparedPath, ...], tuple[Exclusion, ...]]:
    """Validate a view without imputing, substituting or changing membership."""
    if horizon < 1 or not value_field:
        raise FutureModeError("invalid path request")
    prepared: list[PreparedPath] = []
    excluded: list[Exclusion] = []
    expected_steps = set(range(1, horizon + 1))
    for member in sorted(members, key=lambda value: value.key):
        rows = list(path_rows.get(member.episode_id, ()))
        selected: dict[int, Mapping[str, Any]] = {}
        reasons: set[str] = set()
        for row in rows:
            try:
                step = int(row["step"])
            except (KeyError, TypeError, ValueError):
                reasons.add("invalid_step")
                continue
            if step not in expected_steps:
                continue
            if step in selected:
                reasons.add("duplicate_step")
            else:
                selected[step] = row
        if set(selected) != expected_steps:
            reasons.add("missing_step")
        values: list[float] = []
        timestamps: list[str] = []
        contracts: set[str] = set()
        contents: set[str] = set()
        fingerprints: set[str] = set()
        for step in range(1, horizon + 1):
            row = selected.get(step)
            if row is None:
                continue
            if row.get("expected_session_match") is not True:
                reasons.add("unexpected_session")
            if str(row.get("episode_id", "")) != member.episode_id:
                reasons.add("episode_binding")
            if str(row.get("cutoff", "")) != member.cutoff:
                reasons.add("cutoff_binding")
            contracts.add(str(row.get("contract_digest", "")))
            contents.add(str(row.get("source_content_digest", "")))
            fingerprints.add(str(row.get("source_fingerprint", "")))
            timestamp = str(row.get("timestamp", ""))
            timestamps.append(timestamp)
            try:
                number = float(row[value_field])
            except (KeyError, TypeError, ValueError):
                reasons.add("invalid_value")
            else:
                if not isfinite(number):
                    reasons.add("nonfinite_value")
                values.append(number)
        if len(contracts) != 1 or "" in contracts:
            reasons.add("contract_binding")
        if len(contents) != 1 or "" in contents:
            reasons.add("source_content_binding")
        if fingerprints != {member.source_fingerprint}:
            reasons.add("source_fingerprint_binding")
        if len(timestamps) != horizon or any(not value for value in timestamps) \
                or len(set(timestamps)) != len(timestamps) \
                or timestamps != sorted(timestamps):
            reasons.add("timestamp_sequence")
        if reasons:
            excluded.append(Exclusion(member, "+".join(sorted(reasons))))
        else:
            prepared.append(PreparedPath(member, tuple(values), tuple(timestamps)))
    return tuple(prepared), tuple(excluded)


def pairwise_l1(paths: Sequence[PreparedPath]) -> tuple[tuple[float, ...], ...]:
    """Compute the frozen pointwise mean-L1 distance in a fixed operation order."""
    if not paths:
        return ()
    horizon = len(paths[0].values)
    if horizon < 1 or any(len(path.values) != horizon for path in paths):
        raise FutureModeError("path dimensions differ")
    matrix = [[0.0] * len(paths) for _ in paths]
    for left in range(len(paths)):
        if any(not isfinite(value) for value in paths[left].values):
            raise FutureModeError("nonfinite prepared path")
        for right in range(left):
            distance = fsum(
                abs(a - b) for a, b in zip(
                    paths[left].values, paths[right].values, strict=True,
                )
            ) / horizon
            matrix[left][right] = distance
            matrix[right][left] = distance
    return tuple(tuple(row) for row in matrix)


def _matrix(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    value = tuple(tuple(float(item) for item in row) for row in matrix)
    size = len(value)
    if size < 1 or any(len(row) != size for row in value):
        raise FutureModeError("distance matrix must be nonempty and square")
    for left, row in enumerate(value):
        for right, item in enumerate(row):
            if not isfinite(item) or item < 0:
                raise FutureModeError("invalid distance")
            if left == right and item != 0.0:
                raise FutureModeError("distance diagonal differs")
            if item != value[right][left]:
                raise FutureModeError("distance symmetry differs")
    return value


def _assignment(
    matrix: Sequence[Sequence[float]], medoids: Sequence[int],
    keys: Sequence[tuple[int, str]], weights: Sequence[float],
) -> tuple[tuple[int, ...], float]:
    ordered = tuple(sorted(medoids, key=lambda index: keys[index]))
    labels: list[int] = []
    costs: list[float] = []
    for row, weight in zip(matrix, weights, strict=True):
        label = min(range(len(ordered)), key=lambda item: (row[ordered[item]], keys[ordered[item]]))
        labels.append(label)
        costs.append(weight * row[ordered[label]])
    return tuple(labels), fsum(costs)


def pam(
    matrix: Sequence[Sequence[float]], keys: Sequence[tuple[int, str]], k: int,
    *, weights: Sequence[float] | None = None,
) -> PamResult:
    """Run deterministic PAM BUILD/SWAP with exact comparison and stable ties."""
    distances = _matrix(matrix)
    size = len(distances)
    stable_keys = tuple((int(rank), str(episode)) for rank, episode in keys)
    if len(stable_keys) != size or len(set(stable_keys)) != size:
        raise FutureModeError("medoid keys differ")
    if not 1 <= k <= size:
        raise FutureModeError("invalid medoid count")
    member_weights = tuple(1.0 for _ in range(size)) if weights is None else tuple(float(value) for value in weights)
    if len(member_weights) != size or any(not isfinite(value) or value < 0 for value in member_weights) \
            or not any(value > 0 for value in member_weights):
        raise FutureModeError("invalid member weights")
    positive = [index for index, value in enumerate(member_weights) if value > 0]
    if k > len(positive):
        raise FutureModeError("insufficient positive-weight medoid candidates")

    medoids: tuple[int, ...] = ()
    while len(medoids) < k:
        candidates = []
        for candidate in positive:
            if candidate in medoids:
                continue
            proposed = tuple(sorted((*medoids, candidate), key=lambda index: stable_keys[index]))
            _, objective = _assignment(distances, proposed, stable_keys, member_weights)
            candidates.append((objective, tuple(stable_keys[index] for index in proposed), proposed))
        medoids = min(candidates)[2]

    labels, objective = _assignment(distances, medoids, stable_keys, member_weights)
    while True:
        candidates = [(objective, tuple(stable_keys[index] for index in medoids), medoids)]
        for removed in medoids:
            for added in positive:
                if added in medoids:
                    continue
                proposed = tuple(sorted(
                    (index for index in (*medoids, added) if index != removed),
                    key=lambda index: stable_keys[index],
                ))
                _, proposed_objective = _assignment(
                    distances, proposed, stable_keys, member_weights,
                )
                if proposed_objective < objective:
                    candidates.append((
                        proposed_objective,
                        tuple(stable_keys[index] for index in proposed), proposed,
                    ))
        best_objective, _, best_medoids = min(candidates)
        if best_objective >= objective:
            break
        medoids = best_medoids
        labels, objective = _assignment(distances, medoids, stable_keys, member_weights)
    return PamResult(medoids, labels, objective)


def mean_silhouette(
    matrix: Sequence[Sequence[float]], labels: Sequence[int],
) -> float:
    """Return the unweighted mean silhouette; singleton members score zero."""
    distances = _matrix(matrix)
    groups: dict[int, list[int]] = {}
    if len(labels) != len(distances):
        raise FutureModeError("label count differs")
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    if len(groups) < 2:
        return 0.0
    scores: list[float] = []
    for index, label in enumerate(labels):
        own = groups[int(label)]
        if len(own) == 1:
            scores.append(0.0)
            continue
        a = fsum(distances[index][other] for other in own if other != index) / (len(own) - 1)
        b = min(
            fsum(distances[index][other] for other in members) / len(members)
            for other_label, members in groups.items() if other_label != int(label)
        )
        denominator = max(a, b)
        scores.append(0.0 if denominator == 0.0 else (b - a) / denominator)
    return fsum(scores) / len(scores)
