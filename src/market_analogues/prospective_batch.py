"""Prospective monthly prediction sequencing, sealing, and outcome firewall."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


class ProspectiveBatchError(RuntimeError):
    pass


METHODS = ("candidate", "matched_causal_history", "locked_composite")
CLASSES = ("favorable_first", "adverse_first", "no_touch")
BATCH_FILES = ("SOURCE_LOCK.json", "QUERY_REGISTRY.json", "PREDICTIONS.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectiveBatchError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def encode(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def strict_decode(raw: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            require(key not in value, f"duplicate JSON key: {path}/{key}")
            value[key] = item
        return value
    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProspectiveBatchError(f"nonfinite JSON: {path}/{token}"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProspectiveBatchError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ProspectiveBatchError(f"unsafe file: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular file required: {path}")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode,
        )
        require(identity(before) == identity(after), f"file changed during read: {path}")
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def _complete_month_ends(
    sessions: Sequence[pd.Timestamp], freeze: pd.Timestamp, as_of: pd.Timestamp,
) -> list[pd.Timestamp]:
    values = pd.DatetimeIndex(sessions).sort_values().unique()
    require(not values.empty and not values.hasnans, "benchmark calendar differs")
    values = values[values > freeze.tz_localize(None)]
    if values.empty:
        return []
    frame = pd.Series(values, index=values.to_period("M"))
    ends = [pd.Timestamp(value) for value in frame.groupby(level=0).max()]
    now = as_of.tz_localize(None)
    return [value for value in ends if value.to_period("M").end_time < now]


@dataclass(frozen=True)
class ListenerDecision:
    action: str
    completed_batches: int
    required_batches: int
    batch_id: str | None
    cutoff: str | None
    eligible_stock_files: int | None
    reason: str


def decide_next_batch(
    *, freeze: pd.Timestamp, as_of: pd.Timestamp,
    benchmark_sessions: Sequence[pd.Timestamp], completed: Sequence[Mapping[str, str]],
    eligible_stock_files: Mapping[str, int], minimum_stocks: int,
    required_batches: int = 12,
) -> ListenerDecision:
    """Return one action; callers invoke this after a source-change event or schedule."""
    require(required_batches > 0 and minimum_stocks > 0, "positive limits required")
    cutoffs = _complete_month_ends(benchmark_sessions, freeze, as_of)
    require(len(completed) <= required_batches, "too many completed batches")
    for index, row in enumerate(completed):
        require(index < len(cutoffs), "completed batch has no calendar cutoff")
        expected_cutoff = cutoffs[index].date().isoformat()
        require(row == {
            "batch_id": cutoffs[index].strftime("%Y-%m"), "cutoff": expected_cutoff,
        }, "completed batches are not an exact chronological prefix")
    if len(completed) == required_batches:
        return ListenerDecision(
            "collection_complete", len(completed), required_batches, None, None, None,
            "all_preregistered_prediction_batches_are_sealed",
        )
    if len(cutoffs) <= len(completed):
        return ListenerDecision(
            "wait_for_completed_month", len(completed), required_batches, None, None,
            None, "no_unsealed_completed_post_freeze_month",
        )
    cutoff = cutoffs[len(completed)]
    cutoff_text = cutoff.date().isoformat()
    batch_id = cutoff.strftime("%Y-%m")
    count = eligible_stock_files.get(cutoff_text)
    if count is None or count < minimum_stocks:
        return ListenerDecision(
            "wait_for_stock_source", len(completed), required_batches, batch_id,
            cutoff_text, count, "minimum_cutoff_eligible_stock_inventory_not_met",
        )
    return ListenerDecision(
        "run_prediction_batch", len(completed), required_batches, batch_id,
        cutoff_text, count, "next_chronological_batch_is_source_ready",
    )


def _validate_source(value: Mapping[str, Any]) -> None:
    require(set(value) == {
        "schema_version", "batch_id", "cutoff", "maximum_source_timestamp",
        "stock_prefix_manifest_digest", "benchmark_prefix_digest",
        "source_values_after_cutoff_opened",
    }, "source lock field closure differs")
    require(value["schema_version"] == "prospective-source-lock-v1",
            "source lock schema differs")
    cutoff = pd.Timestamp(value["cutoff"])
    require(cutoff.strftime("%Y-%m") == value["batch_id"],
            "source batch month differs from cutoff")
    require(pd.Timestamp(value["maximum_source_timestamp"]) <= cutoff,
            "source lock crosses prediction cutoff")
    require(value["source_values_after_cutoff_opened"] is False,
            "post-cutoff source access differs")
    for field in ("stock_prefix_manifest_digest", "benchmark_prefix_digest"):
        require(type(value[field]) is str and len(value[field]) == 64,
                f"source digest differs: {field}")


def _validate_registry(value: Mapping[str, Any]) -> list[str]:
    require(set(value) == {
        "schema_version", "batch_id", "cutoff", "queries", "query_digest",
        "selection_used_outcomes",
    }, "registry field closure differs")
    require(value["schema_version"] == "prospective-query-registry-v1"
            and value["selection_used_outcomes"] is False,
            "registry boundary differs")
    queries = value["queries"]
    require(type(queries) is list and queries, "nonempty registry required")
    expected = {
        "query_id", "symbol", "quality_tier", "liquidity_stratum",
        "selection_hash", "stock_prefix_digest",
    }
    require(all(type(row) is dict and set(row) == expected for row in queries),
            "registry query closure differs")
    require(all(row["quality_tier"] in {"A", "B"}
                and row["liquidity_stratum"] in {"low", "middle", "high"}
                and type(row["symbol"]) is str and row["symbol"]
                and all(type(row[field]) is str and len(row[field]) == 64
                        for field in ("selection_hash", "stock_prefix_digest"))
                for row in queries), "registry query values differ")
    ids = [row["query_id"] for row in queries]
    require(all(type(value) is str and value for value in ids)
            and len(set(ids)) == len(ids), "registry query IDs differ")
    require(value["query_digest"] == stable(ids), "registry query digest differs")
    return ids


def _validate_predictions(value: Mapping[str, Any], query_ids: Sequence[str]) -> None:
    require(set(value) == {
        "schema_version", "batch_id", "cutoff", "rows", "query_digest",
        "query_outcomes_opened", "source_values_after_cutoff_opened",
    }, "prediction field closure differs")
    require(value["schema_version"] == "prospective-probability-predictions-v1"
            and value["query_outcomes_opened"] is False
            and value["source_values_after_cutoff_opened"] is False,
            "prediction access boundary differs")
    require(value["query_digest"] == stable(list(query_ids)),
            "prediction query digest differs")
    rows = value["rows"]
    require(type(rows) is list and [row.get("query_id") for row in rows] == list(query_ids),
            "prediction query order differs")
    for row in rows:
        require(set(row) == {"query_id", "probabilities", "provenance_digest"}
                and set(row["probabilities"]) == set(METHODS),
                "prediction row closure differs")
        require(type(row["provenance_digest"]) is str
                and len(row["provenance_digest"]) == 64,
                "prediction provenance digest differs")
        for method in METHODS:
            probabilities = row["probabilities"][method]
            require(type(probabilities) is list and len(probabilities) == len(CLASSES)
                    and all(type(item) in {int, float} and math.isfinite(item)
                            and 0 <= item <= 1 for item in probabilities)
                    and abs(sum(probabilities) - 1) <= 1e-12,
                    f"invalid probabilities: {row['query_id']}:{method}")


def _publish_restart_safe(path: Path, value: Mapping[str, Any]) -> bytes:
    content = encode(value)
    if path.exists() or path.is_symlink():
        require(not path.is_symlink() and snapshot(path) == content,
                f"restart prerequisite differs: {path.name}")
        return content
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            require(count > 0, f"short prerequisite write: {path.name}")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return content


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        raw = snapshot(root / name)
        rows.append({"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
    return rows


def _seal_state(
    *, contract_digest: str, batch_id: str, cutoff: str, query_ids: Sequence[str],
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "prospective-prediction-batch-seal-v1",
        "status": "predictions_sealed",
        "passed": True,
        "contract_digest": contract_digest,
        "batch_id": batch_id,
        "cutoff": cutoff,
        "query_count": len(query_ids),
        "query_digest": stable(list(query_ids)),
        "file_manifest": list(manifest),
        "source_values_after_cutoff_opened": False,
        "query_outcomes_opened_before_prediction_seal": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def validate_prediction_batch(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), "prediction batch directory differs")
    require({path.name for path in root.iterdir()} == {*BATCH_FILES, "PREDICTIONS_SEALED.json"}
            and not any(path.is_symlink() for path in root.iterdir()),
            "prediction batch layout differs")
    source = strict_decode(snapshot(root / BATCH_FILES[0]), root / BATCH_FILES[0])
    registry = strict_decode(snapshot(root / BATCH_FILES[1]), root / BATCH_FILES[1])
    predictions = strict_decode(snapshot(root / BATCH_FILES[2]), root / BATCH_FILES[2])
    seal = strict_decode(snapshot(root / "PREDICTIONS_SEALED.json"),
                         root / "PREDICTIONS_SEALED.json")
    _validate_source(source)
    query_ids = _validate_registry(registry)
    _validate_predictions(predictions, query_ids)
    batch_id = source["batch_id"]
    cutoff = source["cutoff"]
    require(all(value["batch_id"] == batch_id and value["cutoff"] == cutoff
                for value in (registry, predictions)), "batch binding differs")
    state = {key: value for key, value in seal.items()
             if key not in {"result_digest", "created_at"}}
    require(seal.get("result_digest") == stable(state), "prediction seal digest differs")
    expected = _seal_state(
        contract_digest=seal["contract_digest"], batch_id=batch_id, cutoff=cutoff,
        query_ids=query_ids, manifest=_manifest(root, BATCH_FILES),
    )
    require(state == expected, "prediction seal state differs")
    return seal


def seal_prediction_batch(
    output_root: Path, *, contract_digest: str, source_lock: Mapping[str, Any],
    registry: Mapping[str, Any], predictions: Mapping[str, Any], created_at: str,
    interrupt_after_prerequisites: bool = False,
    interrupt_after_seal: bool = False,
) -> dict[str, Any]:
    """Seal one batch atomically; identical interrupted attempts resume safely."""
    _validate_source(source_lock)
    query_ids = _validate_registry(registry)
    _validate_predictions(predictions, query_ids)
    batch_id = source_lock["batch_id"]
    cutoff = source_lock["cutoff"]
    require(all(value["batch_id"] == batch_id and value["cutoff"] == cutoff
                for value in (registry, predictions)), "batch binding differs")
    require(type(contract_digest) is str and len(contract_digest) == 64,
            "contract digest differs")
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / f"batch-{batch_id}"
    if final.exists() or final.is_symlink():
        seal = validate_prediction_batch(final)
        require(all(snapshot(final / name) == encode(value)
                    for name, value in zip(BATCH_FILES, (source_lock, registry, predictions)))
                and seal["contract_digest"] == contract_digest,
                "completed batch differs from requested inputs")
        return seal
    stage = output_root / f".batch-{batch_id}.staging"
    if stage.exists():
        require(stage.is_dir() and not stage.is_symlink(), "batch staging path differs")
    else:
        stage.mkdir()
    documents = (source_lock, registry, predictions)
    for name, value in zip(BATCH_FILES, documents):
        _publish_restart_safe(stage / name, value)
    if interrupt_after_prerequisites:
        raise ProspectiveBatchError("injected interruption after prerequisites")
    manifest = _manifest(stage, BATCH_FILES)
    state = _seal_state(
        contract_digest=contract_digest, batch_id=batch_id, cutoff=cutoff,
        query_ids=query_ids, manifest=manifest,
    )
    seal = {**state, "result_digest": stable(state), "created_at": created_at}
    seal_path = stage / "PREDICTIONS_SEALED.json"
    if seal_path.exists() or seal_path.is_symlink():
        existing = strict_decode(snapshot(seal_path), seal_path)
        existing_state = {key: value for key, value in existing.items()
                          if key not in {"result_digest", "created_at"}}
        require(existing_state == state and existing.get("result_digest") == stable(state),
                "restart prediction seal differs")
        seal = existing
    else:
        _publish_restart_safe(seal_path, seal)
    if interrupt_after_seal:
        raise ProspectiveBatchError("injected interruption after prediction seal")
    directory = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    try:
        os.rename(stage, final)
    except OSError:
        require(final.exists(), "atomic prediction publication failed")
        existing = validate_prediction_batch(final)
        require(all(snapshot(final / name) == encode(value)
                    for name, value in zip(BATCH_FILES, (source_lock, registry, predictions)))
                and existing["contract_digest"] == contract_digest,
                "concurrent completed batch differs from requested inputs")
        return existing
    parent = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return validate_prediction_batch(final)


def guarded_outcome_load(
    batch_root: Path, *, benchmark_sessions: Sequence[pd.Timestamp],
    source_as_of: pd.Timestamp, wall_clock: pd.Timestamp, horizon_sessions: int,
    loader: Callable[[Sequence[str]], Any],
) -> Any:
    """Call an outcome loader only after a valid seal and full horizon maturity."""
    require(horizon_sessions > 0, "positive outcome horizon required")
    seal = validate_prediction_batch(batch_root)
    cutoff = pd.Timestamp(seal["cutoff"])
    sessions = pd.DatetimeIndex(benchmark_sessions).sort_values().unique()
    future = sessions[sessions > cutoff]
    require(len(future) >= horizon_sessions, "outcome calendar has not matured")
    maturity = pd.Timestamp(future[horizon_sessions - 1])
    require(source_as_of.tz_localize(None) >= maturity
            and wall_clock.tz_localize(None) > maturity,
            "outcome access attempted before maturity")
    registry = strict_decode(snapshot(batch_root / "QUERY_REGISTRY.json"),
                             batch_root / "QUERY_REGISTRY.json")
    query_ids = _validate_registry(registry)
    result = loader(query_ids)
    require(result is not None, "outcome loader returned no result")
    return result


def decision_document(decision: ListenerDecision) -> dict[str, Any]:
    return {"schema_version": "prospective-listener-decision-v1", **asdict(decision)}
