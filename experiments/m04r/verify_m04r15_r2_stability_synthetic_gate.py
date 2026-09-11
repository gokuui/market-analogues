"""Independently reconstruct the R2-02 stability gate with exhaustive medoids."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from math import fsum
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "m04r15-r2-stability-synthetic-verification-v1"
RESULT = Path("config/data/analogues/m04r15/r2-stability-synthetic-gate-v1/RESULT.json")
OUTPUT = Path("config/data/analogues/m04r15/r2-stability-synthetic-gate-v1-verification")
PRODUCER_MODULES = {"market_analogues.future_modes", "experiments.m04r.m04r15_r2_stability_synthetic_gate"}


class StabilityVerificationError(RuntimeError): pass


def _require(value: bool, message: str) -> None:
    if not value: raise StabilityVerificationError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    raw = path.read_bytes(); value = json.loads(raw); _require(type(value) is dict, "JSON object required")
    return value, raw


def _assign(matrix, medoids, weights):
    labels = tuple(min(range(len(medoids)), key=lambda label: (matrix[row][medoids[label]], medoids[label]))
                   for row in range(len(matrix)))
    return labels, fsum(weights[row] * matrix[row][medoids[labels[row]]] for row in range(len(matrix)))


def _fit(matrix, k, weights=None):
    weights = [1.0] * len(matrix) if weights is None else weights
    positive = [index for index, weight in enumerate(weights) if weight > 0]
    candidates = []
    for medoids in combinations(positive, k):
        labels, objective = _assign(matrix, medoids, weights)
        candidates.append((objective, medoids, labels))
    return min(candidates)


def _ari(left, right):
    choose2 = lambda count: count * (count - 1) // 2
    rows, columns, cells = {}, {}, {}
    for a, b in zip(left, right, strict=True):
        rows[a] = rows.get(a, 0) + 1; columns[b] = columns.get(b, 0) + 1
        cells[(a, b)] = cells.get((a, b), 0) + 1
    cell = sum(choose2(value) for value in cells.values())
    row = sum(choose2(value) for value in rows.values())
    column = sum(choose2(value) for value in columns.values())
    total = choose2(len(left)); expected = row * column / total; maximum = (row + column) / 2
    return 1.0 if maximum == expected else (cell - expected) / (maximum - expected)


def _silhouette(matrix, labels):
    groups = {label: [index for index, value in enumerate(labels) if value == label]
              for label in sorted(set(labels))}
    if len(groups) < 2: return 0.0
    scores = []
    for index, label in enumerate(labels):
        own = groups[label]
        if len(own) == 1: scores.append(0.0); continue
        a = fsum(matrix[index][other] for other in own if other != index) / (len(own) - 1)
        b = min(fsum(matrix[index][other] for other in members) / len(members)
                for other, members in groups.items() if other != label)
        scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return fsum(scores) / len(scores)


def _integers(seed: bytes, stop: int, count: int):
    result, counter = [], 0; limit = ((1 << 64) // stop) * stop
    while len(result) < count:
        value = int.from_bytes(sha256(seed + counter.to_bytes(8, "big")).digest()[:8], "big")
        counter += 1
        if value < limit: result.append(value % stop)
    return result


def _stability(matrix, blocks, medoids, labels, *, name, k):
    unique = sorted(set(blocks))
    if len(unique) < max(4, k + 1): return None, 0, []
    scores = []
    prefix = "\0".join(("6bec009810afce7c58508fca28179579f5904382376dc9c5bce74aa74f41e08c",
                         name, "absolute_close_return")).encode()
    for replicate in range(256):
        picks = _integers(prefix + b"\0" + replicate.to_bytes(8, "big"), len(unique), len(unique))
        sampled = [unique[index] for index in picks]
        if len(set(sampled)) < k: continue
        counts = {block: sampled.count(block) for block in unique}
        weights = [float(counts[block]) for block in blocks]
        if sum(value > 0 for value in weights) < k: continue
        _, boot_medoids, boot_labels = _fit(matrix, k, weights)
        if len(set(boot_labels)) != k: continue
        scores.append(_ari(labels, boot_labels))
    ordered = sorted(scores); middle = len(ordered) // 2
    median = None if not scores else (ordered[middle] if len(ordered) % 2
             else (ordered[middle - 1] + ordered[middle]) / 2)
    return median, len(scores), scores


def _case(levels: Sequence[float], blocks: Sequence[str], name: str) -> dict[str, Any]:
    matrix = tuple(tuple(abs(a - b) for b in levels) for a in levels)
    if len(levels) < 3:
        return {"status": "abstain_insufficient_complete_primary_members", "selected_k": 0,
                "medoid_indices": [], "labels": [], "candidates": []}
    candidates = []
    for k in range(2, min(4, len(levels) // 3) + 1):
        _, medoids, labels = _fit(matrix, k)
        sizes = [labels.count(label) for label in range(k)]
        silhouette = _silhouette(matrix, labels); reasons = []; stability = None
        if min(sizes) < 3: reasons.append("mode_smaller_than_3")
        if silhouette < 0.25: reasons.append("mean_silhouette_below_0.25")
        if not reasons:
            median, valid, scores = _stability(matrix, blocks, medoids, labels, name=name, k=k)
            stability = {
                "valid_replicates": valid, "median_adjusted_rand_index": median,
                "adjusted_rand_indices_digest": _stable(scores),
                "minimum_adjusted_rand_index": min(scores) if scores else None,
                "maximum_adjusted_rand_index": max(scores) if scores else None,
            }
            if valid < 205: reasons.append("fewer_than_80_percent_valid_block_bootstraps")
            if median is None or median < 0.8: reasons.append("median_adjusted_rand_index_below_0.8")
        candidates.append({"k": k, "medoid_indices": list(medoids), "labels": list(labels),
                           "cluster_sizes": sizes, "mean_silhouette": silhouette,
                           "accepted": not reasons, "rejection_reasons": reasons,
                           "stability": stability})
    accepted = [value for value in candidates if value["accepted"]]
    if accepted:
        selected = min(accepted, key=lambda value: (-value["mean_silhouette"], value["k"]))
        return {"status": "stable_multiple_modes", "selected_k": selected["k"],
                "medoid_indices": selected["medoid_indices"], "labels": selected["labels"],
                "candidates": candidates}
    _, medoids, labels = _fit(matrix, 1)
    return {"status": "one_mode_fallback", "selected_k": 1,
            "medoid_indices": list(medoids), "labels": list(labels), "candidates": candidates}


def verify(repository: Path, result_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    result, raw = _json((result_path or repository / RESULT).resolve(strict=True))
    deterministic = {key: value for key, value in result.items() if key not in {"result_digest", "created_at"}}
    _require(result.get("result_digest") == _stable(deterministic), "result digest differs")
    expected = {
        "stable_three_modes": _case([-1.01, -1, -.99, -.01, 0, .01, .99, 1, 1.01],
            [f"20{20 + i // 4}-Q{i % 4 + 1}" for i in range(9)], "stable"),
        "date_confounded": _case([-1.01, -1, -.99, .99, 1, 1.01],
            ["2020-Q1"] * 3 + ["2020-Q2"] * 3, "date-confounded"),
        "tight_one_family": _case([0] * 6, [f"20{i:02d}-Q1" for i in range(6)], "tight"),
        "tiny_cohort": _case([0, 1], ["2020-Q1", "2021-Q1"], "tiny"),
    }
    _require(result.get("cases") == expected, "independent stability reconstruction differs")
    _require(all((result.get("passed") is True, result.get("repeat_identity") is True,
                  result.get("real_future_path_store_opened") is False,
                  result.get("bounded_consumed_data_poc_authorized") is True,
                  result.get("real_full_build_authorized") is False,
                  result.get("predictive_claim_authorized") is False,
                  result.get("production_promotion_authorized") is False,
                  result.get("trading_claim_authorized") is False)), "gate boundary differs")
    commit = result["implementation_commit"]
    for name, digest in result["runtime_sha256"].items():
        historical = subprocess.run(("git", "show", f"{commit}:{name}"), cwd=repository,
                                    capture_output=True, check=False)
        _require(historical.returncode == 0 and sha256(historical.stdout).hexdigest() == digest,
                 f"historical runtime differs: {name}")
    state = {"schema_version": SCHEMA, "status": "independently_verified", "passed": True,
             "producer_result_digest": result["result_digest"],
             "producer_result_sha256": sha256(raw).hexdigest(),
             "implementation_commit": commit,
             "independent_oracle": "exhaustive_weighted_medoid_and_sha256_block_bootstrap",
             "producer_modules_imported": False, "verified_case_count": 4,
             "real_future_path_store_opened": False,
             "bounded_consumed_data_poc_authorized": True, "real_full_build_authorized": False,
             "predictive_claim_authorized": False, "production_promotion_authorized": False,
             "trading_claim_authorized": False}
    return {**state, "result_digest": _stable(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".r2-stability-verification-", dir=path.parent))
    try:
        target = temporary / "VERIFIED.json"; descriptor = os.open(target, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()},
                                     indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        try:
            if (temporary / "VERIFIED.json").exists(): (temporary / "VERIFIED.json").unlink()
            temporary.rmdir()
        except OSError: pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--result", type=Path); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); value = verify(args.repository, args.result)
    if not args.dry_run: _publish(args.repository.resolve(strict=True) / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
