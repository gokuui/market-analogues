from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import resource
import shutil

import pandas as pd

from .adapters import OHLCVSource
from .exhaustive import build_exact_frontier, exhaustive_frontier_search
from .types import Episode, InstrumentKey, SearchQuery, stable_hash


@dataclass(frozen=True)
class ScaleRung:
    fraction: float
    selected_instruments: int
    total_eligible_instruments: int
    eligible_candidates: int
    exact_evaluated: int
    safely_pruned: int
    result_digest: str
    repeated_digest: str
    previous_top_k_overlap: float | None
    seeded_shards: int
    instruments_built: int
    instruments_reused: int
    build_seconds: float
    resume_seconds: float
    search_seconds: float
    repeat_search_seconds: float
    frontier_bytes: int
    projected_full_build_seconds: float | None
    projected_full_search_seconds: float
    peak_rss_mb: float
    free_disk_bytes: int
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ScaleLadderResult:
    dataset_id: str
    query_episode_id: str
    seed: str
    total_eligible_instruments: int
    fractions: tuple[float, ...]
    rungs: tuple[ScaleRung, ...]
    passed: bool
    failures: tuple[str, ...]


def collect_scale_history(
    gates: Path, dataset_id: str, query_episode_id: str, seed: str,
) -> dict[float, dict[str, object]]:
    records: dict[float, tuple[str, dict[str, object]]] = {}
    paths = list((gates / "history").glob(f"12e_exhaustive_scale_*_{dataset_id}/*.json"))
    paths.extend(gates.glob(f"12e_exhaustive_scale_*_{dataset_id}.json"))
    for path in sorted(set(paths)):
        try:
            payload = json.loads(path.read_text())
            metrics = payload["metrics"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if (
            not payload.get("passed")
            or metrics.get("dataset") != dataset_id
            or metrics.get("query_episode_id") != query_episode_id
            or metrics.get("seed") != seed
        ):
            continue
        for rung in metrics.get("rungs", []):
            if (
                not rung.get("passed")
                or rung.get("result_digest") != rung.get("repeated_digest")
            ):
                continue
            fraction = float(rung["fraction"])
            stamp = str(payload.get("created_at", ""))
            if fraction not in records or stamp > records[fraction][0]:
                records[fraction] = (stamp, rung)
    return {fraction: value[1] for fraction, value in sorted(records.items())}


def deterministic_instrument_prefixes(
    source: OHLCVSource,
    quality: pd.DataFrame,
    request: SearchQuery,
    fractions: tuple[float, ...],
    *,
    seed: str,
) -> tuple[tuple[InstrumentKey, ...], ...]:
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("fractions must contain values in (0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise ValueError("fractions must be unique and increasing")
    qmap = {str(row.symbol): str(row.tier) for row in quality.itertuples(index=False)}
    eligible = [
        key for key in source.instruments()
        if (not request.search_datasets or key.dataset_id in request.search_datasets)
        and qmap.get(key.source_symbol, "A") in request.quality_tiers
        and qmap.get(key.source_symbol, "A") != "QUARANTINED"
    ]
    eligible.sort(key=lambda key: (
        sha256(f"{seed}\0{key}".encode()).hexdigest(), str(key),
    ))
    return tuple(
        tuple(eligible[:max(1, math.ceil(len(eligible) * fraction))])
        for fraction in fractions
    )


def _matches_digest(matches) -> str:
    return stable_hash([
        {
            "episode_id": match.episode_key.id,
            "total": float(match.total_distance),
            "components": match.component_distances,
        }
        for match in matches
    ])


def _seed_from_largest_lower_scale(
    root: Path, rung_root: Path, query_episode_id: str, fraction: float,
) -> int:
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("scale-*"):
        try:
            candidate_fraction = int(path.name.removeprefix("scale-")) / 1_000_000
        except ValueError:
            continue
        if candidate_fraction < fraction:
            candidates.append((candidate_fraction, path))
    if not candidates:
        return 0
    source = max(candidates)[1] / query_episode_id / "shards"
    if not source.exists():
        return 0
    seeded = 0
    for path in source.rglob("*.npz"):
        destination = rung_root / query_episode_id / "shards" / path.relative_to(source)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
        seeded += 1
    return seeded


def run_scale_ladder(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    quality: pd.DataFrame,
    root: Path,
    *,
    fractions: tuple[float, ...],
    seed: str = "gate12-scale-v1",
    stride: int = 5,
    batch_size: int = 128,
    frontier_batch_rows: int = 256,
    representation_cache_shards: int = 2,
    tolerance: float = 1e-12,
    maximum_rss_mb: float = 1024,
    disk_reserve_bytes: int = 5 * 1024 ** 3,
    maximum_projected_hours: float = 2,
    rebuild_invalid: bool = False,
) -> ScaleLadderResult:
    prefixes = deterministic_instrument_prefixes(
        source, quality, request, fractions, seed=seed,
    )
    total = len(prefixes[-1]) if fractions[-1] == 1 else len(
        deterministic_instrument_prefixes(
            source, quality, request, (1.0,), seed=seed,
        )[0]
    )
    rungs: list[ScaleRung] = []
    failures: list[str] = []
    previous_ids: set[str] | None = None
    for fraction, keys in zip(fractions, prefixes):
        tag = f"scale-{round(fraction * 1_000_000):07d}"
        rung_root = root / tag
        seeded = _seed_from_largest_lower_scale(
            root, rung_root, query.key.id, fraction,
        )
        build = build_exact_frontier(
            query, source, request, quality, rung_root, stride=stride,
            batch_size=batch_size, instrument_keys=keys,
            rebuild_invalid=rebuild_invalid,
        )
        resume = build_exact_frontier(
            query, source, request, quality, rung_root, stride=stride,
            batch_size=batch_size, instrument_keys=keys,
        )
        rung_failures = list(build.failures) + list(resume.failures)
        first = exhaustive_frontier_search(
            query, source, request, rung_root, quality=quality,
            tolerance=tolerance, frontier_batch_rows=frontier_batch_rows,
            representation_cache_shards=representation_cache_shards,
        )
        repeated = exhaustive_frontier_search(
            query, source, request, rung_root, quality=quality,
            tolerance=tolerance, frontier_batch_rows=frontier_batch_rows,
            representation_cache_shards=representation_cache_shards,
        )
        digest = _matches_digest(first.matches)
        repeated_digest = _matches_digest(repeated.matches)
        ids = {match.episode_key.id for match in first.matches}
        overlap = (
            len(ids.intersection(previous_ids)) / request.top_k
            if previous_ids is not None else None
        )
        previous_ids = ids
        certificate = first.certificate
        if digest != repeated_digest:
            rung_failures.append("repeated exact result digest differs")
        if len(first.matches) != request.top_k:
            rung_failures.append(
                f"returned {len(first.matches)} matches; require {request.top_k}"
            )
        if certificate.exact_evaluated + certificate.safely_pruned != certificate.eligible_candidates:
            rung_failures.append("candidate accounting does not reconcile")
        if not certificate.stopped_early:
            rung_failures.append("strict early-stop certificate was not established")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if rss > maximum_rss_mb:
            rung_failures.append(
                f"peak RSS {rss:.2f} MB exceeds {maximum_rss_mb:.2f} MB"
            )
        frontier_bytes = sum(
            path.stat().st_size for path in rung_root.rglob("*") if path.is_file()
        )
        free = shutil.disk_usage(root.parent if root.parent.exists() else Path.cwd()).free
        projection_factor = total / len(keys)
        projected_frontier = frontier_bytes * projection_factor
        if free + frontier_bytes - projected_frontier < disk_reserve_bytes:
            rung_failures.append("projected full frontier violates the disk reserve")
        projected_build = (
            build.seconds * projection_factor
            if build.instruments_built and not build.instruments_reused else None
        )
        projected_search = certificate.elapsed_seconds * projection_factor
        if projected_search > maximum_projected_hours * 3600:
            rung_failures.append(
                f"projected search {projected_search / 3600:.2f} h exceeds "
                f"{maximum_projected_hours:.2f} h"
            )
        rung = ScaleRung(
            fraction, len(keys), total, certificate.eligible_candidates,
            certificate.exact_evaluated, certificate.safely_pruned,
            digest, repeated_digest, overlap, seeded,
            build.instruments_built, build.instruments_reused,
            build.seconds, resume.seconds,
            certificate.elapsed_seconds, repeated.certificate.elapsed_seconds,
            frontier_bytes, projected_build, projected_search, rss, free,
            not rung_failures, tuple(rung_failures),
        )
        rungs.append(rung)
        failures.extend(f"{fraction:.4f}:{failure}" for failure in rung_failures)
    return ScaleLadderResult(
        query.key.instrument.dataset_id, query.key.id, seed, total, fractions,
        tuple(rungs), not failures, tuple(failures),
    )


def write_scale_ladder_report(result: ScaleLadderResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        "<tr>"
        f"<td>{rung.fraction:.2%}</td><td>{rung.selected_instruments}</td>"
        f"<td>{rung.eligible_candidates:,}</td><td>{rung.exact_evaluated:,}</td>"
        f"<td>{rung.safely_pruned:,}</td><td>{rung.search_seconds:.2f}</td>"
        f"<td>{rung.projected_full_search_seconds / 3600:.2f} h</td>"
        f"<td>{rung.peak_rss_mb:.1f} MB</td>"
        f"<td>{'PASS' if rung.passed else 'FAIL'}</td></tr>"
        for rung in result.rungs
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    payload = escape(json.dumps(asdict(result), indent=2, default=str))
    status = "PASS" if result.passed else "FAIL"
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gate 12 scale ladder</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}</style></head><body><header><h1>T12-04 deterministic scale ladder: {status}</h1><p>Nested symbol prefixes are ordered by a locked hash. Each rung is atomically built, resumed, searched twice and projected independently; projections are planning evidence, not a full-universe certificate.</p></header><section><table><thead><tr><th>Scale</th><th>Symbols</th><th>Candidates</th><th>Exact</th><th>Pruned</th><th>Search</th><th>Projected full</th><th>RSS</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failures}</ul></section><section><h2>Machine detail</h2><pre>{payload}</pre></section></body></html>""")
    return path
