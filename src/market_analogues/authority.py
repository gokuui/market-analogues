from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .exhaustive import ExhaustiveResult, FrontierBuildResult
from .exhaustive import build_exact_frontier, exhaustive_frontier_search
from .types import Episode, InstrumentKey, SearchQuery, stable_hash


AUTHORITY_SCHEMA_VERSION = "gate12-authority-v1"


class AuthorityError(ValueError):
    pass


@dataclass(frozen=True)
class AuthorityArtifact:
    path: Path
    query_episode_id: str
    result_digest: str
    authority_digest: str
    eligible_candidates: int
    exact_evaluated: int
    safely_pruned: int


def _json_value(value: object) -> object:
    """Return the exact JSON-domain value used on disk and in content digests."""
    return json.loads(json.dumps(value, default=str))


def _certificate_failure(certificate: dict[str, object]) -> str | None:
    eligible = int(certificate.get("eligible_candidates", -1))
    evaluated = int(certificate.get("exact_evaluated", -1))
    pruned = int(certificate.get("safely_pruned", -1))
    if eligible < 0 or evaluated < 0 or pruned < 0 or evaluated + pruned != eligible:
        return "candidate accounting does not reconcile"
    if bool(certificate.get("stopped_early")):
        threshold = float(certificate.get("stop_threshold", "nan"))
        next_bound = certificate.get("next_lower_bound")
        if next_bound is None or not np.isfinite(threshold) or float(next_bound) <= threshold:
            return "strict stopping bound is invalid"
        return None
    if evaluated == eligible and pruned == 0 and certificate.get("next_lower_bound") is None:
        return None
    return "neither strict early stopping nor full exhaustion was established"


def _match_payload(match) -> dict[str, object]:
    return {
        "episode_id": match.episode_key.id,
        "dataset_id": match.episode_key.instrument.dataset_id,
        "symbol": match.episode_key.instrument.source_symbol,
        "cutoff": match.episode_key.cutoff.isoformat(),
        "lookback": match.episode_key.lookback,
        "representation_version": match.episode_key.representation_version,
        "total_distance": float(match.total_distance),
        "component_distances": {
            str(key): float(value)
            for key, value in sorted(match.component_distances.items())
        },
        "alignment": [[int(left), int(right)] for left, right in match.alignment],
        "quality_tier": match.quality_tier,
        "quality_issues": list(match.quality_issues),
    }


def authority_universe_digest(
    source: OHLCVSource, request: SearchQuery, quality: pd.DataFrame,
) -> str:
    qmap = {str(row.symbol): str(row.tier) for row in quality.itertuples(index=False)}
    records = []
    for key in sorted(source.instruments()):
        tier = qmap.get(key.source_symbol, "A")
        if (
            (not request.search_datasets or key.dataset_id in request.search_datasets)
            and tier in request.quality_tiers
            and tier != "QUARANTINED"
        ):
            records.append((str(key), source.fingerprint(key)))
    return stable_hash(records)


def _manifest_universe_digest(path: Path) -> str:
    try:
        manifest = json.loads(path.read_text())
        records = manifest["shards"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AuthorityError(f"cannot derive authority universe provenance: {exc}") from exc
    return stable_hash([
        (f"{record['dataset_id']}:{record['symbol']}", record["source_fingerprint"])
        for record in sorted(records, key=lambda value: (
            str(value["dataset_id"]), str(value["symbol"]),
        ))
    ])


def write_authority_artifact(
    query: Episode,
    request: SearchQuery,
    build: FrontierBuildResult,
    resume: FrontierBuildResult,
    result: ExhaustiveResult,
    repeated: ExhaustiveResult,
    path: Path,
    *,
    source_fingerprint: str,
    benchmark_fingerprint: str | None,
    registry_digest: str,
) -> AuthorityArtifact:
    matches = [_match_payload(match) for match in result.matches]
    repeated_matches = [_match_payload(match) for match in repeated.matches]
    result_digest = stable_hash(matches)
    repeated_digest = stable_hash(repeated_matches)
    certificate = asdict(result.certificate)
    repeated_certificate = asdict(repeated.certificate)
    failures: list[str] = []
    if not build.passed or not resume.passed:
        failures.append("frontier build or resume did not pass")
    if len(matches) != request.top_k:
        failures.append(f"authority contains {len(matches)} matches; require {request.top_k}")
    if result_digest != repeated_digest:
        failures.append("repeated authority digest differs")
    certificate_failure = _certificate_failure(certificate)
    if certificate_failure:
        failures.append(certificate_failure)
    repeated_failure = _certificate_failure(repeated_certificate)
    if repeated_failure:
        failures.append(f"repeated {repeated_failure}")
    if certificate["manifest_digest"] != repeated_certificate["manifest_digest"]:
        failures.append("repeated search used a different frontier manifest")
    if build.manifest_digest != resume.manifest_digest:
        failures.append("frontier changed during unchanged resume")
    if build.manifest_digest != certificate["manifest_digest"]:
        failures.append("search certificate does not identify the built frontier")
    if failures:
        raise AuthorityError("; ".join(failures))
    payload = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "query_episode_id": query.key.id,
        "query": {
            "dataset_id": query.key.instrument.dataset_id,
            "symbol": query.key.instrument.source_symbol,
            "cutoff": query.key.cutoff.isoformat(),
            "lookback": query.key.lookback,
            "representation_version": query.key.representation_version,
        },
        "request_digest": stable_hash(asdict(request)),
        "registry_digest": registry_digest,
        "source_fingerprint": source_fingerprint,
        "universe_source_digest": _manifest_universe_digest(build.manifest_path),
        "benchmark_fingerprint": benchmark_fingerprint,
        "frontier_manifest_digest": result.certificate.manifest_digest,
        "frontier_build": asdict(build),
        "frontier_resume": asdict(resume),
        "certificate": certificate,
        "repeated_certificate": repeated_certificate,
        "matches": matches,
        "result_digest": result_digest,
        "repeated_digest": repeated_digest,
    }
    payload = _json_value(payload)
    assert isinstance(payload, dict)
    payload["authority_digest"] = stable_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return load_authority_artifact(path)


def load_authority_artifact(path: Path) -> AuthorityArtifact:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"cannot load authority {path}: {exc}") from exc
    if payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise AuthorityError("unsupported authority schema")
    claimed = payload.get("authority_digest")
    content = dict(payload)
    content.pop("authority_digest", None)
    if claimed != stable_hash(content):
        raise AuthorityError("authority content digest mismatch")
    if payload.get("result_digest") != stable_hash(payload.get("matches")):
        raise AuthorityError("authority result digest mismatch")
    if payload.get("result_digest") != payload.get("repeated_digest"):
        raise AuthorityError("authority repeated digest mismatch")
    certificate = payload.get("certificate") or {}
    repeated_certificate = payload.get("repeated_certificate") or {}
    eligible = int(certificate.get("eligible_candidates", -1))
    evaluated = int(certificate.get("exact_evaluated", -1))
    pruned = int(certificate.get("safely_pruned", -1))
    certificate_failure = _certificate_failure(certificate)
    if certificate_failure:
        raise AuthorityError(f"authority certificate is invalid: {certificate_failure}")
    repeated_failure = _certificate_failure(repeated_certificate)
    if repeated_failure:
        raise AuthorityError(
            f"authority repeated certificate is invalid: {repeated_failure}"
        )
    if certificate.get("manifest_digest") != repeated_certificate.get("manifest_digest"):
        raise AuthorityError("authority repeated frontier manifest differs")
    frontier_digest = payload.get("frontier_manifest_digest")
    if frontier_digest != certificate.get("manifest_digest"):
        raise AuthorityError("authority frontier manifest binding is invalid")
    build = payload.get("frontier_build") or {}
    resume = payload.get("frontier_resume") or {}
    if build.get("manifest_digest") != frontier_digest:
        raise AuthorityError("authority build manifest binding is invalid")
    if resume.get("manifest_digest") != frontier_digest:
        raise AuthorityError("authority resume manifest binding is invalid")
    return AuthorityArtifact(
        path, str(payload["query_episode_id"]), str(payload["result_digest"]),
        str(claimed), eligible, evaluated, pruned,
    )


def validate_authority_artifact(
    path: Path,
    *,
    query: Episode,
    request: SearchQuery,
    source_fingerprint: str,
    benchmark_fingerprint: str | None,
    registry_digest: str,
    universe_source_digest: str,
) -> AuthorityArtifact:
    artifact = load_authority_artifact(path)
    payload = json.loads(path.read_text())
    expected = {
        "query_episode_id": query.key.id,
        "request_digest": stable_hash(asdict(request)),
        "source_fingerprint": source_fingerprint,
        "benchmark_fingerprint": benchmark_fingerprint,
        "registry_digest": registry_digest,
        "universe_source_digest": universe_source_digest,
    }
    stale = [name for name, value in expected.items() if payload.get(name) != value]
    if stale:
        raise AuthorityError(f"authority provenance is stale: {', '.join(stale)}")
    return artifact


def seed_frontier_shards(
    source_root: Path, target_root: Path, query_episode_id: str,
) -> int:
    """Seed a new authority root; the frontier builder remains the validator."""
    source = source_root / query_episode_id / "shards"
    if not source.exists():
        return 0
    seeded = 0
    for path in source.rglob("*.npz"):
        destination = target_root / query_episode_id / "shards" / path.relative_to(source)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
        seeded += 1
    return seeded


def run_authority_case(
    query: Episode,
    source: OHLCVSource,
    request: SearchQuery,
    quality: pd.DataFrame,
    frontier_root: Path,
    artifact_path: Path,
    *,
    registry_digest: str,
    seed_frontier_root: Path | None = None,
    stride: int = 5,
    batch_size: int = 128,
    frontier_batch_rows: int = 256,
    representation_cache_shards: int = 2,
    tolerance: float = 1e-12,
    rebuild_invalid: bool = False,
) -> tuple[AuthorityArtifact, int, FrontierBuildResult, FrontierBuildResult]:
    source_fingerprint = source.fingerprint(query.key.instrument)
    benchmark_fingerprint = source.benchmark_fingerprint()
    if artifact_path.exists():
        universe_digest = authority_universe_digest(source, request, quality)
        artifact = validate_authority_artifact(
            artifact_path, query=query, request=request,
            source_fingerprint=source_fingerprint,
            benchmark_fingerprint=benchmark_fingerprint,
            registry_digest=registry_digest,
            universe_source_digest=universe_digest,
        )
        # A validated artifact is a completed checkpoint. No expensive work repeats.
        empty = FrontierBuildResult(
            True, query.key.id, 0, 0, 0, 0, artifact.eligible_candidates, (),
            frontier_root / query.key.id / "manifest.json", "checkpoint", 0.0,
        )
        return artifact, 0, empty, empty
    seeded = (
        seed_frontier_shards(seed_frontier_root, frontier_root, query.key.id)
        if seed_frontier_root is not None else 0
    )
    keys = tuple(sorted(
        key for key in source.instruments()
        if not request.search_datasets or key.dataset_id in request.search_datasets
    ))
    build = build_exact_frontier(
        query, source, request, quality, frontier_root, stride=stride,
        batch_size=batch_size, instrument_keys=keys,
        rebuild_invalid=rebuild_invalid,
    )
    resume = build_exact_frontier(
        query, source, request, quality, frontier_root, stride=stride,
        batch_size=batch_size, instrument_keys=keys,
    )
    if not build.passed or not resume.passed:
        raise AuthorityError("frontier build/resume failed: " + "; ".join(
            (*build.failures, *resume.failures),
        ))
    first = exhaustive_frontier_search(
        query, source, request, frontier_root, quality=quality,
        tolerance=tolerance, frontier_batch_rows=frontier_batch_rows,
        representation_cache_shards=representation_cache_shards,
    )
    repeated = exhaustive_frontier_search(
        query, source, request, frontier_root, quality=quality,
        tolerance=tolerance, frontier_batch_rows=frontier_batch_rows,
        representation_cache_shards=representation_cache_shards,
    )
    artifact = write_authority_artifact(
        query, request, build, resume, first, repeated, artifact_path,
        source_fingerprint=source_fingerprint,
        benchmark_fingerprint=benchmark_fingerprint,
        registry_digest=registry_digest,
    )
    return artifact, seeded, build, resume


def write_authority_report(artifact_path: Path, output: Path) -> Path:
    artifact = load_authority_artifact(artifact_path)
    payload = json.loads(artifact_path.read_text())
    certificate = payload["certificate"]
    rows = "".join(
        "<tr>"
        f"<td>{rank}</td><td>{escape(str(match['dataset_id']))}:{escape(str(match['symbol']))}</td>"
        f"<td>{escape(str(match['cutoff']))}</td><td>{float(match['total_distance']):.10g}</td>"
        "</tr>"
        for rank, match in enumerate(payload["matches"], 1)
    )
    proof = "strict next-bound stop" if certificate["stopped_early"] else "full exhaustion"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Blind exact authority</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd}}</style></head><body><header><h1>Gate 12 blind exact authority: PASS</h1><p>This ranking was generated without reading production retrieval rankings or forward outcomes. Exactness proof: {escape(proof)}.</p></header><section><h2>Certificate</h2><pre>{escape(json.dumps(certificate, indent=2))}</pre></section><section><h2>Top matches</h2><table><thead><tr><th>Rank</th><th>Instrument</th><th>Cutoff</th><th>Distance</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Integrity</h2><p>Result digest: <code>{artifact.result_digest}</code></p><p>Authority digest: <code>{artifact.authority_digest}</code></p></section></body></html>""")
    return output
