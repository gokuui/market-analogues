"""Prepare or seal an exact resident mirror for the frozen packed generation."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

from market_analogues.resident_store import prepare_resident_mirror_observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--expected-provenance-digest", required=True)
    parser.add_argument("--reserve-gib", type=float, default=1.0)
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    if not isfinite(args.reserve_gib) or args.reserve_gib < 0:
        raise ValueError("reserve GiB must be finite and nonnegative")
    result, observation = prepare_resident_mirror_observed(
        args.source_full_root / "store", args.mirror_root,
        args.generation_id,
        expected_provenance_digest=args.expected_provenance_digest,
        reserve_bytes=int(args.reserve_gib * 1024 ** 3),
        validate_existing=args.validate_existing,
    )
    print(json.dumps({
        "ready": True,
        "ready_path": str((args.mirror_root / "READY.json").resolve()),
        "schema_version": result["schema_version"],
        "mode": result["mode"],
        "content_digest": result["content_digest"],
        "seal_digest": result["seal_digest"],
        "ready_digest": result["ready_digest"],
        "startup_timings": result["startup_timings"],
        "fresh_validation_observation": observation,
        "latency_scope": result["seal"]["latency_scope"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
