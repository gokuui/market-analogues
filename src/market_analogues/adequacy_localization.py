"""Pure, outcome-free R1-B conditional structural localization calculations.

Arrays use query order for rows and the *full recurrent cohort* for episode
columns. Primary-cohort filtering is applied only when aggregating episodes.
No file access, process scheduling, random state, or production retrieval code
is used here. The runner owns frozen identities, eligibility and provenance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from typing import Hashable, Sequence

import numpy as np


DIMENSIONS = 141
REPLICATES = 4096
PRIMARY_EPISODES = 357
PRIORITY_DOMAIN = b"R1B-B2-conditional-priority-v1"
PRIORITY_FAMILY = b"N0-N1-common"


class LocalizationError(ValueError):
    """An input does not satisfy the frozen numerical contract."""


def _finite(values: object, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise LocalizationError(f"{name} must be finite and {ndim}-dimensional")
    return array


def empirical_chart_transform(
    queries: np.ndarray, candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Map queries and candidates using exact query-column empirical midranks.

    Query constants map to zero; candidates equal to the constant map to zero,
    below to -1 and above to +1, following the same empirical clipping rule.
    Exact float64 equality defines ties;
    -0 and +0 tie and outputs canonicalize zero. Nonfinite values are refused.
    Candidate values outside a nonconstant query range clip to -1/+1.
    """
    query = _finite(queries, ndim=2, name="queries")
    candidate = _finite(candidates, ndim=2, name="candidates")
    if len(query) < 2 or query.shape[1] != DIMENSIONS or candidate.shape[1] != DIMENSIONS:
        raise LocalizationError("chart inputs require at least two queries and 141 columns")
    qz = np.zeros_like(query)
    ez = np.zeros_like(candidate)
    constants: list[int] = []
    count = len(query)
    for column in range(DIMENSIONS):
        ordered = np.sort(query[:, column], kind="stable")
        if ordered[0] == ordered[-1]:
            constants.append(column)
            ez[candidate[:, column] < ordered[0], column] = -1.0
            ez[candidate[:, column] > ordered[0], column] = 1.0
            continue
        for source, target in ((query, qz), (candidate, ez)):
            left = np.searchsorted(ordered, source[:, column], side="left")
            right = np.searchsorted(ordered, source[:, column], side="right")
            # Ties: a+b = (left+1)+right. Gaps: 2L+1.
            doubled_rank = left + right + 1
            target[:, column] = np.clip(
                (doubled_rank - (count + 1)) / (count - 1), -1.0, 1.0,
            )
    qz[qz == 0.0] = 0.0
    ez[ez == 0.0] = 0.0
    return qz, ez, tuple(constants)


def chart_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise 141D RMS using direct differences, avoiding dot-product cancellation.

    One left row at a time bounds temporary memory. Each pair is reduced with
    math.fsum in fixed feature order, then divided by 141 and square-rooted,
    independent of worker count and chunk boundaries.
    """
    a = _finite(left, ndim=2, name="left chart matrix")
    b = _finite(right, ndim=2, name="right chart matrix")
    if a.shape[1] != DIMENSIONS or b.shape[1] != DIMENSIONS:
        raise LocalizationError("chart distance requires exactly 141 columns")
    result = np.empty((len(a), len(b)), dtype=np.float64)
    for index, row in enumerate(a):
        with np.errstate(over="ignore", invalid="ignore"):
            delta = b - row
            squares = delta * delta
        if not np.isfinite(squares).all():
            raise LocalizationError("chart distance overflow")
        result[index] = [math.sqrt(math.fsum(map(float, pair)) / DIMENSIONS) for pair in squares]
    if not np.isfinite(result).all():
        raise LocalizationError("chart distance overflow")
    return result


def candidate_specificity_ranks(
    query_episode_distances: np.ndarray, eligibility: np.ndarray,
) -> np.ndarray:
    """Rank each eligible episode against every eligible full-cohort episode.

    Ineligible entries are NaN sentinels and can never be selected. For a tie
    occupying zero-based [a,b), (rank-.5)/n equals (a+b)/(2*n).
    """
    distances = _finite(query_episode_distances, ndim=2, name="query episode distances")
    eligible = np.asarray(eligibility)
    if eligible.dtype != np.bool_ or eligible.shape != distances.shape or (distances < 0).any():
        raise LocalizationError("eligibility must be a matching boolean matrix; distances nonnegative")
    result = np.full(distances.shape, np.nan, dtype=np.float64)
    for query_index in range(len(distances)):
        ids = np.flatnonzero(eligible[query_index])
        if not len(ids):
            continue
        values = distances[query_index, ids]
        ordered = np.sort(values, kind="stable")
        left = np.searchsorted(ordered, values, side="left")
        right = np.searchsorted(ordered, values, side="right")
        result[query_index, ids] = (left + right) / (2.0 * len(ids))
    return result


def _indices(values: Sequence[int], size: int, *, name: str) -> tuple[int, ...]:
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
        raise LocalizationError(f"{name} indices must be integers")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result) or any(value < 0 or value >= size for value in result):
        raise LocalizationError(f"{name} indices duplicate or outside matrix")
    return result


def query_cohesion(distances: np.ndarray, selected: Sequence[int]) -> float:
    matrix = _finite(distances, ndim=2, name="query distance matrix")
    if matrix.shape[0] != matrix.shape[1] or (matrix < 0).any() or not np.array_equal(matrix, matrix.T) or np.any(matrix.diagonal() != 0):
        raise LocalizationError("query distances must be symmetric, nonnegative, with zero diagonal")
    indices = sorted(_indices(selected, len(matrix), name="selected"))
    if len(indices) < 2:
        raise LocalizationError("cohesion requires two selected queries")
    return math.fsum(float(matrix[a, b]) for position, a in enumerate(indices) for b in indices[position + 1:]) / math.comb(len(indices), 2)


def episode_specificity(ranks: np.ndarray, selected: Sequence[int], episode: int) -> float:
    matrix = np.asarray(ranks, dtype=np.float64)
    if matrix.ndim != 2 or isinstance(episode, bool) or not isinstance(episode, (int, np.integer)) or not 0 <= episode < matrix.shape[1]:
        raise LocalizationError("specificity matrix or episode index invalid")
    indices = sorted(_indices(selected, len(matrix), name="selected"))
    if not indices:
        raise LocalizationError("specificity requires selected queries")
    values = matrix[indices, episode]
    if not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
        raise LocalizationError("selected episode must have valid eligible percentile ranks")
    return math.fsum(map(float, values)) / len(values)


def entropy_breadth(selected: Sequence[int], k8_labels: Sequence[int]) -> float:
    indices = _indices(selected, len(k8_labels), name="selected")
    if not indices or any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or not 0 <= label < 8 for label in k8_labels):
        raise LocalizationError("breadth requires nonempty membership and K8 labels")
    counts = Counter(int(k8_labels[index]) for index in indices)
    probabilities = [count / len(indices) for _, count in sorted(counts.items())]
    return math.exp(-math.fsum(p * math.log(p) for p in probabilities)) / min(8, len(indices))


def equal_episode_mean(values: Sequence[float]) -> float:
    array = _finite(values, ndim=1, name="episode values")
    if not len(array):
        raise LocalizationError("episode aggregate is empty")
    return math.fsum(map(float, array)) / len(array)


class PreparedLocalization:
    """Validate immutable geometry once, then evaluate batches of selected sets.

    The input matrices are copied so a caller cannot mutate validated state.
    Per-batch work is proportional to selected pairs, not the full query square.
    For one episode all selected sets have the same frozen cardinality.
    """

    def __init__(
        self, query_distances: np.ndarray, specificity_ranks: np.ndarray,
        eligibility: np.ndarray, k8_labels: Sequence[int],
    ) -> None:
        distances = _finite(query_distances, ndim=2, name="query distance matrix")
        if distances.shape[0] != distances.shape[1] or (distances < 0).any() or not np.array_equal(distances, distances.T) or np.any(distances.diagonal() != 0):
            raise LocalizationError("query distances must be symmetric, nonnegative, with zero diagonal")
        ranks = np.asarray(specificity_ranks, dtype=np.float64)
        eligible = np.asarray(eligibility)
        if ranks.ndim != 2 or ranks.shape[0] != len(distances) or eligible.shape != ranks.shape or eligible.dtype != np.bool_:
            raise LocalizationError("prepared ranks, eligibility and query dimensions differ")
        if not np.isnan(ranks[~eligible]).all() or not np.isfinite(ranks[eligible]).all() or ((ranks[eligible] <= 0) | (ranks[eligible] >= 1)).any():
            raise LocalizationError("prepared ranks must match eligible finite percentiles and ineligible NaN")
        if len(k8_labels) != len(distances) or any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or not 0 <= label < 8 for label in k8_labels):
            raise LocalizationError("prepared labels must be K8 query labels")
        self._distances = distances.copy()
        self._ranks = ranks.copy()
        self._eligible = eligible.copy()
        self._labels = np.asarray(k8_labels, dtype=np.int8).copy()
        for array in (self._distances, self._ranks, self._eligible, self._labels):
            array.flags.writeable = False

    def evaluate(self, episode: int, selections: np.ndarray) -> np.ndarray:
        """Return [cohesion, specificity] rows identical to the scalar oracle."""
        selected = np.asarray(selections)
        if isinstance(episode, bool) or not isinstance(episode, (int, np.integer)) or not 0 <= episode < self._ranks.shape[1]:
            raise LocalizationError("prepared episode index invalid")
        if selected.ndim != 2 or selected.dtype.kind not in "iu" or selected.shape[1] < 2 or not len(selected):
            raise LocalizationError("selections require nonempty integer rows with at least two queries")
        if (selected < 0).any() or (selected >= len(self._distances)).any():
            raise LocalizationError("selected query outside prepared geometry")
        selected = np.sort(selected, axis=1)
        if np.any(np.diff(selected, axis=1) == 0) or not self._eligible[selected, episode].all():
            raise LocalizationError("selected queries duplicate or causally ineligible")
        left, right = np.triu_indices(selected.shape[1], k=1)
        pairs = self._distances[selected[:, left], selected[:, right]]
        ranks = self._ranks[selected, episode]
        result = np.empty((len(selected), 2), dtype=np.float64)
        for index in range(len(selected)):
            result[index, 0] = math.fsum(map(float, pairs[index])) / len(left)
            result[index, 1] = math.fsum(map(float, ranks[index])) / selected.shape[1]
        return result

    def breadth(self, selections: np.ndarray) -> np.ndarray:
        """Descriptive effective K8 breadth; never part of the N1 gate."""
        selected = np.asarray(selections)
        if selected.ndim != 2 or selected.dtype.kind not in "iu" or not len(selected):
            raise LocalizationError("breadth selections require a nonempty integer matrix")
        return np.asarray([entropy_breadth(row, self._labels) for row in selected])


def _length_prefixed(parts: Sequence[bytes]) -> bytes:
    return b"".join(len(part).to_bytes(4, "big") + part for part in parts)


def conditional_priority(
    contract_digest: str, replicate: int, episode_id: str, query_id: str,
    *, shared_query: bool = False,
) -> bytes:
    """Full SHA256 priority common to N0/N1; domains separate sensitivity.

    All fields are uint32-length-prefixed bytes. Contract digest is decoded
    lowercase hex, replicate is uint32 big endian, IDs are UTF-8. N0 and N1
    deliberately share the family field; only their conditioning cells differ.
    """
    if not isinstance(contract_digest, str) or re.fullmatch(r"[0-9a-f]{64}", contract_digest) is None:
        raise LocalizationError("contract digest must be lowercase SHA256 hex")
    if isinstance(replicate, bool) or not isinstance(replicate, (int, np.integer)) or not 0 <= replicate < 2**32:
        raise LocalizationError("replicate must be uint32")
    if not isinstance(episode_id, str) or not episode_id or not isinstance(query_id, str) or not query_id or not isinstance(shared_query, bool):
        raise LocalizationError("priority identities or scheme invalid")
    parts = [PRIORITY_DOMAIN, bytes.fromhex(contract_digest), PRIORITY_FAMILY,
             b"shared-query" if shared_query else b"episode", int(replicate).to_bytes(4, "big")]
    if not shared_query:
        parts.append(episode_id.encode("utf-8"))
    parts.append(query_id.encode("utf-8"))
    return sha256(_length_prefixed(parts)).digest()


def conditional_draw(
    *, eligible: Sequence[int], observed: Sequence[int], cells: Sequence[Hashable],
    query_ids: Sequence[str], episode_id: str, contract_digest: str, replicate: int,
    shared_query: bool = False,
) -> tuple[int, ...]:
    """Uniform per-cell subset via complete cryptographic random priorities.

    Results are canonicalized by query ID, with query ID also resolving an
    exceptionally unlikely full-hash collision. The observed set is eligible.
    """
    if len(cells) != len(query_ids) or len(set(query_ids)) != len(query_ids) or any(not isinstance(value, str) or not value for value in query_ids):
        raise LocalizationError("query identities and cells differ")
    universe = _indices(eligible, len(query_ids), name="eligible")
    selected = _indices(observed, len(query_ids), name="observed")
    if not selected or not set(selected).issubset(universe):
        raise LocalizationError("observed membership must be a nonempty eligible subset")
    try:
        required = Counter(cells[index] for index in selected)
        available: dict[Hashable, list[int]] = defaultdict(list)
        for index in universe:
            available[cells[index]].append(index)
    except TypeError as error:
        raise LocalizationError("cells must be hashable") from error
    chosen: list[int] = []
    for cell, count in required.items():
        ordered = sorted(available[cell], key=lambda index: (
            conditional_priority(contract_digest, replicate, episode_id, query_ids[index], shared_query=shared_query),
            query_ids[index],
        ))
        chosen.extend(ordered[:count])
    return tuple(sorted(chosen, key=lambda index: query_ids[index]))


class PreparedConditionalDraw:
    """Prevalidate one episode/null and cache invariant priority bytes.

    Hashes and sampled sets are exactly those of ``conditional_draw``. Cells
    with zero observed count need no hash or sorting. N0 and N1 instances use
    the same byte family, preserving the frozen common priorities.
    """

    def __init__(
        self, *, eligible: Sequence[int], observed: Sequence[int], cells: Sequence[Hashable],
        query_ids: Sequence[str], episode_id: str, contract_digest: str,
        shared_query: bool = False,
    ) -> None:
        # Exercise all scalar validation once, including ID/priority contracts.
        conditional_draw(eligible=eligible, observed=observed, cells=cells,
                         query_ids=query_ids, episode_id=episode_id,
                         contract_digest=contract_digest, replicate=0, shared_query=shared_query)
        counts = Counter(cells[index] for index in observed)
        self._ids = tuple(query_ids)
        self._groups = tuple((count, tuple(index for index in eligible if cells[index] == cell)) for cell, count in counts.items())
        self._prefix = _length_prefixed([
            PRIORITY_DOMAIN, bytes.fromhex(contract_digest), PRIORITY_FAMILY,
            b"shared-query" if shared_query else b"episode",
        ])
        self._suffix = {
            index: _length_prefixed(([] if shared_query else [episode_id.encode("utf-8")]) + [query_ids[index].encode("utf-8")])
            for _, indices in self._groups for index in indices
        }
        self._size = len(observed)

    def draw(self, replicate: int) -> tuple[int, ...]:
        if isinstance(replicate, bool) or not isinstance(replicate, (int, np.integer)) or not 0 <= replicate < 2**32:
            raise LocalizationError("replicate must be uint32")
        prefix = self._prefix + _length_prefixed([int(replicate).to_bytes(4, "big")])
        chosen: list[int] = []
        for count, indices in self._groups:
            ordered = sorted(indices, key=lambda index: (
                sha256(prefix + self._suffix[index]).digest(), self._ids[index],
            ))
            chosen.extend(ordered[:count])
        return tuple(sorted(chosen, key=lambda index: self._ids[index]))

    def batch(self, start: int, stop: int) -> np.ndarray:
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (start, stop)) or not 0 <= start < stop <= 2**32:
            raise LocalizationError("replicate interval must be a nonempty uint32 range")
        return np.asarray([self.draw(replicate) for replicate in range(start, stop)], dtype=np.int64).reshape(stop - start, self._size)


def plus_one_lower_p(observed: float, null: Sequence[float]) -> float:
    values = _finite(null, ndim=1, name="null values")
    if not np.isfinite(observed) or not len(values):
        raise LocalizationError("p-value requires finite observation and replicates")
    return (1 + int(np.count_nonzero(values <= observed))) / (len(values) + 1)


def required_improved_count(episodes: int) -> int:
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)) or episodes < 1:
        raise LocalizationError("episode count must be a positive integer")
    return (3 * int(episodes) + 4) // 5


def episode_parities(episode_ids: Sequence[str]) -> np.ndarray:
    # EpisodeKey.id is the repository's canonical 96-bit (24-hex) SHA-256
    # prefix.  Do not confuse it with the separate 64-hex contract digest.
    if len(set(episode_ids)) != len(episode_ids) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{24}", value) is None
        for value in episode_ids
    ):
        raise LocalizationError("episode IDs must be unique lowercase 24-hex identifiers")
    return np.asarray([int(value, 16) & 1 for value in episode_ids], dtype=np.int8)


@dataclass(frozen=True)
class EffectSummary:
    observed_cohesion: float
    null_cohesion: float
    cohesion_relative_improvement: float | None
    specificity_improvement: float
    cohesion_improved_episodes: int
    specificity_improved_episodes: int
    required_improved_episodes: int
    practical_pass: bool


def effect_summary(observed: np.ndarray, null_means: np.ndarray) -> EffectSummary:
    """Two columns: cohesion and specificity; equal weight per episode."""
    obs = _finite(observed, ndim=2, name="observed statistics")
    means = _finite(null_means, ndim=2, name="null episode means")
    if obs.shape != means.shape or obs.shape[1] != 2 or len(obs) == 0 or (obs < 0).any() or (means < 0).any():
        raise LocalizationError("effect inputs must have matching nonnegative two-column shape")
    if (obs[:, 1] > 1).any() or (means[:, 1] > 1).any():
        raise LocalizationError("specificity effects require percentile values at most one")
    observed_cohesion = equal_episode_mean(obs[:, 0])
    null_cohesion = equal_episode_mean(means[:, 0])
    cohesion_effect = ((null_cohesion - observed_cohesion) / null_cohesion
                       if null_cohesion > 0 else None)
    specificity_effect = equal_episode_mean(means[:, 1]) - equal_episode_mean(obs[:, 1])
    cohesion_count = int(np.count_nonzero(obs[:, 0] < means[:, 0]))
    specificity_count = int(np.count_nonzero(obs[:, 1] < means[:, 1]))
    required = required_improved_count(len(obs))
    return EffectSummary(observed_cohesion, null_cohesion, cohesion_effect,
                         specificity_effect, cohesion_count, specificity_count, required,
                         cohesion_effect is not None and cohesion_effect >= 0.05 and specificity_effect >= 0.05
                         and cohesion_count >= required and specificity_count >= required)


def deterministic_null_summary(null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return replicate aggregates and episode means in fixed canonical order."""
    values = _finite(null, ndim=3, name="null table")
    if values.shape[2] != 2 or not values.shape[0] or not values.shape[1]:
        raise LocalizationError("null table shape must be replicates by episodes by two")
    if (values < 0).any() or (values[:, :, 1] > 1).any():
        raise LocalizationError("null statistics must be nonnegative and specificity at most one")
    aggregates = np.asarray([[equal_episode_mean(row[:, metric]) for metric in range(2)] for row in values])
    means = np.asarray([[equal_episode_mean(values[:, episode, metric]) for metric in range(2)] for episode in range(values.shape[1])])
    return aggregates, means


def assemble_replicate_chunks(
    chunks: Sequence[tuple[int, np.ndarray]], *, replicates: int, episodes: int,
) -> np.ndarray:
    """Reassemble restart/worker outputs, refusing holes, overlaps and bad values."""
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1 for value in (replicates, episodes)):
        raise LocalizationError("chunk dimensions must be positive")
    result = np.empty((replicates, episodes, 2), dtype=np.float64)
    covered = np.zeros(replicates, dtype=np.bool_)
    for start, chunk in chunks:
        values = _finite(chunk, ndim=3, name="replicate chunk")
        if isinstance(start, bool) or not isinstance(start, (int, np.integer)) or values.shape[1:] != (episodes, 2) or not len(values) or start < 0 or start + len(values) > replicates:
            raise LocalizationError("replicate chunk bounds or shape differ")
        stop = start + len(values)
        if covered[start:stop].any():
            raise LocalizationError("replicate chunks overlap")
        result[start:stop] = values
        covered[start:stop] = True
    if not covered.all():
        raise LocalizationError("replicate chunks have gaps")
    return result


@dataclass(frozen=True)
class LocalizationDecision:
    status: str
    reasons: tuple[str, ...]
    n1_pvalues: tuple[float, float] | None = None
    n1_effects: EffectSummary | None = None
    n0_effects: EffectSummary | None = None


def localization_decision(
    *, episode_ids: Sequence[str], observed: np.ndarray,
    n0_null: np.ndarray, n1_null: np.ndarray,
    shared_query_n0_null: np.ndarray, shared_query_n1_null: np.ndarray,
    prerequisite_verified: bool, primary_k: int = 8,
) -> LocalizationDecision:
    """Frozen primary decision; inferential failures cannot be rescued.

    Invalid/incomplete inputs and failed robustness are ``unresolved``.
    Valid calculations missing the scientific/effect gates are
    ``not_established``. The only success is ``structurally_localized``.
    """
    if prerequisite_verified is not True or primary_k != 8:
        return LocalizationDecision("unresolved", ("prerequisite verification or primary K8 differs",))
    try:
        parity = episode_parities(episode_ids)
        obs = _finite(observed, ndim=2, name="observed statistics")
        if len(episode_ids) != PRIMARY_EPISODES or obs.shape != (PRIMARY_EPISODES, 2):
            raise LocalizationError("primary population differs from 357 supported episodes")
        summaries = []
        for table in (
            n0_null, n1_null, shared_query_n0_null, shared_query_n1_null,
        ):
            array = _finite(table, ndim=3, name="null statistics")
            if array.shape != (REPLICATES, PRIMARY_EPISODES, 2):
                raise LocalizationError("exactly 4096 complete replicates required")
            summaries.append(deterministic_null_summary(array))
        ((_, n0_means), (n1_aggregates, n1_means),
         (_, shared_n0_means), (_, shared_n1_means)) = summaries
        n0_effects = effect_summary(obs, n0_means)
        n1_effects = effect_summary(obs, n1_means)
        shared_n0_effects = effect_summary(obs, shared_n0_means)
        shared_n1_effects = effect_summary(obs, shared_n1_means)
        pvalues = tuple(plus_one_lower_p(equal_episode_mean(obs[:, metric]), n1_aggregates[:, metric]) for metric in range(2))
        if n0_effects.cohesion_relative_improvement is None or n1_effects.cohesion_relative_improvement is None:
            return LocalizationDecision("not_established", ("zero null-mean cohesion leaves relative effect undefined",), pvalues, n1_effects, n0_effects)
        instability: list[str] = []
        for split in (0, 1):
            mask = parity == split
            if not mask.any():
                instability.append(f"episode parity {split} is empty")
                continue
            effects = effect_summary(obs[mask], n1_means[mask])
            if effects.cohesion_relative_improvement is None or effects.cohesion_relative_improvement < 0.05 or effects.specificity_improvement < 0.05:
                instability.append(f"episode parity {split} practical effects failed")
        for omitted in range(PRIMARY_EPISODES):
            mask = np.arange(PRIMARY_EPISODES) != omitted
            effects = effect_summary(obs[mask], n1_means[mask])
            if effects.cohesion_relative_improvement is None or effects.cohesion_relative_improvement <= 0 or effects.specificity_improvement <= 0:
                instability.append("leave-one-episode-out positive effects failed")
                break
        for name, effects in (
            ("N0", shared_n0_effects), ("N1", shared_n1_effects),
        ):
            if (effects.cohesion_relative_improvement is None
                    or effects.cohesion_relative_improvement <= 0
                    or effects.specificity_improvement <= 0):
                instability.append(f"shared-query {name} positive effects failed")
        if instability:
            return LocalizationDecision("unresolved", tuple(instability), pvalues, n1_effects, n0_effects)
        failures: list[str] = []
        if any(value > 0.01 for value in pvalues):
            failures.append("both N1 lower-tail p-values must be at most 0.01")
        if not n1_effects.practical_pass:
            failures.append("N1 practical effects or 60 percent episode counts failed")
        if not n0_effects.practical_pass:
            failures.append("N0 corroboration effects or 60 percent episode counts failed")
        return LocalizationDecision("not_established" if failures else "structurally_localized", tuple(failures), pvalues, n1_effects, n0_effects)
    except (LocalizationError, TypeError, ValueError, OverflowError) as error:
        return LocalizationDecision("unresolved", (str(error),))
