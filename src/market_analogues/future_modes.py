"""Deterministic primitives for descriptive fixed-neighbor future-path modes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
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


@dataclass(frozen=True)
class StabilityResult:
    valid_replicates: int
    adjusted_rand_indices: tuple[float, ...]
    median_adjusted_rand_index: float | None


@dataclass(frozen=True)
class ModeCandidate:
    k: int
    medoid_indices: tuple[int, ...]
    labels: tuple[int, ...]
    cluster_sizes: tuple[int, ...]
    mean_silhouette: float
    stability: StabilityResult | None
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ModeSelection:
    status: str
    selected_k: int
    medoid_indices: tuple[int, ...]
    labels: tuple[int, ...]
    candidates: tuple[ModeCandidate, ...]


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


def calendar_quarter(value: str) -> str:
    """Return an exact calendar-quarter block from an ISO cutoff date."""
    try:
        parsed = date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as error:
        raise FutureModeError("invalid matched cutoff") from error
    return f"{parsed.year:04d}-Q{(parsed.month - 1) // 3 + 1}"


def adjusted_rand_index(left: Sequence[int], right: Sequence[int]) -> float:
    """Compute the Hubert-Arabie adjusted Rand index with exact integer counts."""
    if len(left) != len(right) or len(left) < 2:
        raise FutureModeError("ARI label count differs")
    left_groups: dict[int, int] = {}
    right_groups: dict[int, int] = {}
    cells: dict[tuple[int, int], int] = {}
    for a, b in zip(left, right, strict=True):
        a, b = int(a), int(b)
        left_groups[a] = left_groups.get(a, 0) + 1
        right_groups[b] = right_groups.get(b, 0) + 1
        cells[(a, b)] = cells.get((a, b), 0) + 1
    choose2 = lambda count: count * (count - 1) // 2
    cell_pairs = sum(choose2(count) for count in cells.values())
    left_pairs = sum(choose2(count) for count in left_groups.values())
    right_pairs = sum(choose2(count) for count in right_groups.values())
    total_pairs = choose2(len(left))
    expected = left_pairs * right_pairs / total_pairs
    maximum = (left_pairs + right_pairs) / 2
    denominator = maximum - expected
    return 1.0 if denominator == 0.0 else (cell_pairs - expected) / denominator


class _HashStream:
    def __init__(self, seed: bytes):
        self.seed = seed
        self.counter = 0

    def integer(self, stop: int) -> int:
        if stop < 1:
            raise FutureModeError("invalid random range")
        limit = ((1 << 64) // stop) * stop
        while True:
            raw = sha256(self.seed + self.counter.to_bytes(8, "big")).digest()
            self.counter += 1
            value = int.from_bytes(raw[:8], "big")
            if value < limit:
                return value % stop


def bootstrap_stability(
    matrix: Sequence[Sequence[float]], keys: Sequence[tuple[int, str]],
    blocks: Sequence[str], base: PamResult, *, contract_digest: str,
    query_case_id: str, view_id: str, replicates: int = 256,
) -> StabilityResult:
    """Refit weighted PAM to deterministic calendar-block bootstrap samples."""
    distances = _matrix(matrix)
    if len(keys) != len(distances) or len(blocks) != len(distances):
        raise FutureModeError("stability input count differs")
    if replicates < 1 or len(base.medoid_indices) < 2:
        raise FutureModeError("invalid stability request")
    unique_blocks = tuple(sorted({str(value) for value in blocks}))
    k = len(base.medoid_indices)
    if len(unique_blocks) < max(4, k + 1):
        return StabilityResult(0, (), None)
    indices_by_block = {
        block: tuple(index for index, value in enumerate(blocks) if str(value) == block)
        for block in unique_blocks
    }
    scores: list[float] = []
    seed_prefix = "\0".join((
        contract_digest, query_case_id, view_id,
    )).encode("utf-8")
    for replicate in range(replicates):
        stream = _HashStream(seed_prefix + b"\0" + replicate.to_bytes(8, "big"))
        sampled = [unique_blocks[stream.integer(len(unique_blocks))]
                   for _ in unique_blocks]
        if len(set(sampled)) < k:
            continue
        counts = {block: sampled.count(block) for block in unique_blocks}
        weights = [float(counts[str(block)]) for block in blocks]
        if sum(weight > 0 for weight in weights) < k:
            continue
        fitted = pam(distances, keys, k, weights=weights)
        if len(set(fitted.labels)) != k:
            continue
        scores.append(adjusted_rand_index(base.labels, fitted.labels))
    if not scores:
        return StabilityResult(0, (), None)
    ordered = sorted(scores)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2
              else (ordered[middle - 1] + ordered[middle]) / 2)
    return StabilityResult(len(scores), tuple(scores), median)


def select_modes(
    matrix: Sequence[Sequence[float]], keys: Sequence[tuple[int, str]],
    blocks: Sequence[str], *, contract_digest: str, query_case_id: str,
    view_id: str, replicates: int = 256,
) -> ModeSelection:
    """Apply frozen size, separation and block-stability gates with safe fallback."""
    distances = _matrix(matrix)
    size = len(distances)
    if len(keys) != size or len(blocks) != size:
        raise FutureModeError("mode-selection input count differs")
    if size < 3:
        return ModeSelection("abstain_insufficient_complete_primary_members", 0, (), (), ())
    candidates: list[ModeCandidate] = []
    maximum = min(4, size // 3)
    for k in range(2, maximum + 1):
        fitted = pam(distances, keys, k)
        sizes = tuple(fitted.labels.count(label) for label in range(k))
        silhouette = mean_silhouette(distances, fitted.labels)
        reasons: list[str] = []
        if min(sizes) < 3:
            reasons.append("mode_smaller_than_3")
        if silhouette < 0.25:
            reasons.append("mean_silhouette_below_0.25")
        stability = None
        if not reasons:
            stability = bootstrap_stability(
                distances, keys, blocks, fitted, contract_digest=contract_digest,
                query_case_id=query_case_id, view_id=view_id, replicates=replicates,
            )
            minimum_valid = (replicates * 4 + 4) // 5
            if stability.valid_replicates < minimum_valid:
                reasons.append("fewer_than_80_percent_valid_block_bootstraps")
            if stability.median_adjusted_rand_index is None \
                    or stability.median_adjusted_rand_index < 0.8:
                reasons.append("median_adjusted_rand_index_below_0.8")
        candidates.append(ModeCandidate(
            k, fitted.medoid_indices, fitted.labels, sizes, silhouette, stability,
            not reasons, tuple(reasons),
        ))
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if accepted:
        chosen = min(accepted, key=lambda value: (-value.mean_silhouette, value.k))
        return ModeSelection(
            "stable_multiple_modes", chosen.k, chosen.medoid_indices,
            chosen.labels, tuple(candidates),
        )
    fallback = pam(distances, keys, 1)
    return ModeSelection(
        "one_mode_fallback", 1, fallback.medoid_indices, fallback.labels,
        tuple(candidates),
    )
