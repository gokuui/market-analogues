"""Locate non-finite values in the interrupted composite-batch worker window."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_t14_10_wf03_composite_topology_poc as kernel
from experiments.m04r import m04r14_t14_10_wf03_composite_batch as batch
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base


WINDOW_IDS = (
    "2a06972a7bb9e0832b8ce7da", "103d01f5fa4d8f28994a1020",
    "dd2d818ae29133bd4e7d7b73", "af773ef872f10e19682a18bc",
    "0184e6eafed9f3fce21c3f4f", "4eecc473804a0686988dd6fd",
    "10db5957c0e2d6dba55895ac", "9c0e2b094dcd9a2ab5ee3e7e",
    "83e8b46ad597cc5624ee7dbc", "e317ea5cc36eff26afb65988",
    "da3cdb481d6a722da8524601", "6485491813edae383b7b7dd7",
)


class CompositeNonfiniteDiagnosticError(RuntimeError):
    pass


def _nonfinite_paths(value: Any, prefix: str = "$") -> list[dict[str, str]]:
    output = []
    if type(value) is float and not math.isfinite(value):
        return [{"path": prefix, "value": value.hex()}]
    if type(value) is dict:
        for key, item in value.items():
            output.extend(_nonfinite_paths(item, f"{prefix}.{key}"))
    elif type(value) in (list, tuple):
        for index, item in enumerate(value):
            output.extend(_nonfinite_paths(item, f"{prefix}[{index}]"))
    return output


def _diagnose(
    repository_string: str, store_root_string: str, row: dict[str, Any],
) -> dict[str, Any]:
    original_dumps = json.dumps

    def diagnostic_dumps(value: Any, **kwargs: Any) -> str:
        return original_dumps(value, **{**kwargs, "allow_nan": True})

    kernel.json.dumps = diagnostic_dumps
    worker = kernel._run_group(
        repository_string, store_root_string, (row,), batch.THREADS_PER_PROCESS,
    )
    paths = _nonfinite_paths(worker)
    return {
        "query_id": row["episode_id"], "symbol": row["symbol"],
        "cutoff": row["cutoff"], "nonfinite_paths": paths,
        "matches_nonfinite": any(item["path"].startswith("$.cases[0].matches")
                                 for item in paths),
        "certificate_nonfinite": any(
            item["path"].startswith("$.cases[0].certificate") for item in paths
        ),
        "worker_nonfinite": any(not item["path"].startswith("$.cases[0]")
                                for item in paths),
    }


def run(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    registry, by_id = base._registry(repository)
    try:
        rows = [by_id[query_id] for query_id in WINDOW_IDS]
    except KeyError as exc:
        raise CompositeNonfiniteDiagnosticError("diagnostic query is absent") from exc
    resident = base._resident()
    context = multiprocessing.get_context("spawn")
    observations = []
    with ProcessPoolExecutor(max_workers=len(rows), mp_context=context) as executor:
        futures = {
            executor.submit(
                _diagnose, str(repository), resident["store_root"], dict(row),
            ): row for row in rows
        }
        for future in as_completed(futures):
            value = future.result()
            observations.append(value)
            print(
                f"[nonfinite-diagnostic] query={value['query_id']} "
                f"paths={len(value['nonfinite_paths'])}", flush=True,
            )
    observations.sort(key=lambda item: WINDOW_IDS.index(item["query_id"]))
    state = {
        "schema_version": "m04r14-wf03-composite-nonfinite-diagnostic-v1",
        "status": "complete", "registry_digest": registry["registry_digest"],
        "queries": len(observations),
        "queries_with_nonfinite": sum(bool(item["nonfinite_paths"])
                                      for item in observations),
        "matches_with_nonfinite": sum(item["matches_nonfinite"]
                                      for item in observations),
        "certificates_with_nonfinite": sum(item["certificate_nonfinite"]
                                           for item in observations),
        "worker_measurements_with_nonfinite": sum(item["worker_nonfinite"]
                                                  for item in observations),
        "observations": observations,
        "outcomes_or_labels_used": False,
    }
    return base._sealed(state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.repository)
    base._atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
