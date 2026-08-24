from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adapters import OHLCVSource
from .authority import load_authority_artifact
from .distance import representation_distance
from .distance_v1_reference import (
    distance_v1_contract, reference_representation_distance,
)
from .episodes import build_episode
from .m04_candidate_recall import M04CandidateRecallSpec
from .representation import represent
from .types import InstrumentKey, stable_hash


M04R_DISTANCE_VERIFIER_SCHEMA = "m04r-distance-v1-verification-v1"
PRACTICAL_AUTHORITY_TOLERANCE = 1e-6
REFERENCE_PARITY_TOLERANCE = 1e-10


class M04RDistanceVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class M04RDistanceVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    cases: tuple[dict[str, Any], ...]
    contract: dict[str, Any]
    result_digest: str


def verify_m04r_distance_v1(
    spec: M04CandidateRecallSpec,
    source: OHLCVSource,
    artifact_dir: Path,
    *,
    dataset_id: str = "nasdaq",
) -> M04RDistanceVerificationResult:
    if source.spec.dataset_id != dataset_id:
        raise M04RDistanceVerificationError("source dataset differs")
    registry_path = artifact_dir / "gate12" / dataset_id / "query-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text()) or {}
    if registry.get("registry_digest") != spec.payload["registry_digests"][dataset_id]:
        raise M04RDistanceVerificationError("frozen registry digest differs")
    registry_rows = {
        str(row["episode_id"]): row for row in registry.get("cases_data", [])
    }
    authority_root = artifact_dir / "gate12" / "authorities" / dataset_id / "cases"
    episode_ids = [
        episode_id for episode_id in spec.payload["holdout_episode_ids"]
        if (authority_root / f"{episode_id}.json").exists()
    ]
    if len(episode_ids) != 12:
        raise M04RDistanceVerificationError(
            f"expected 12 {dataset_id} authorities, found {len(episode_ids)}"
        )
    failures: list[str] = []
    case_rows: list[dict[str, Any]] = []
    maximum_reference_total_delta = 0.0
    maximum_reference_component_delta = 0.0
    maximum_authority_total_delta = 0.0
    maximum_authority_component_delta = 0.0
    path_mismatches = symmetry_mismatches = nonfinite_outputs = 0
    identity_mismatches = triangle_violations = 0
    pairs = 0

    for episode_id in episode_ids:
        authority_path = authority_root / f"{episode_id}.json"
        authority = load_authority_artifact(authority_path)
        payload = json.loads(authority_path.read_text())
        if authority.authority_digest != spec.payload["authority_digests"][episode_id]:
            raise M04RDistanceVerificationError(f"authority digest differs for {episode_id}")
        query_meta = payload["query"]
        registry_row = registry_rows.get(episode_id)
        if registry_row is None:
            raise M04RDistanceVerificationError(f"registry omits {episode_id}")
        if source.fingerprint(InstrumentKey(dataset_id, str(query_meta["symbol"]))) != registry_row[
            "source_fingerprint"
        ]:
            raise M04RDistanceVerificationError(f"query source changed for {episode_id}")
        query = build_episode(
            source, InstrumentKey(dataset_id, str(query_meta["symbol"])),
            str(query_meta["cutoff"]), int(query_meta["lookback"]),
            str(query_meta["representation_version"]),
        )
        query_representation = represent(query)
        identity = reference_representation_distance(
            query_representation, query_representation,
        )
        if identity.total != 0.0:
            identity_mismatches += 1
        triangle_representations = [query_representation]
        pair_rows = []
        for match in payload["matches"]:
            candidate = build_episode(
                source, InstrumentKey(dataset_id, str(match["symbol"])),
                str(match["cutoff"]), int(match["lookback"]),
                str(match["representation_version"]),
            )
            candidate_representation = represent(candidate)
            production_total, production_components, production_path = representation_distance(
                query_representation, candidate_representation,
            )
            reference = reference_representation_distance(
                query_representation, candidate_representation,
            )
            reverse = reference_representation_distance(
                candidate_representation, query_representation,
            )
            total_delta = abs(production_total - reference.total)
            component_delta = max(
                abs(production_components[name] - reference.components[name])
                for name in reference.components
            )
            authority_total_delta = abs(production_total - float(match["total_distance"]))
            authority_component_delta = max(
                abs(production_components[name] - float(match["component_distances"][name]))
                for name in production_components
            )
            path_matches = tuple(production_path) == reference.alignment
            authority_path_matches = tuple(production_path) == tuple(
                (int(left), int(right)) for left, right in match["alignment"]
            )
            symmetry_delta = abs(reference.total - reverse.total)
            finite = (
                math.isfinite(production_total) and math.isfinite(reference.total)
                and all(math.isfinite(value) for value in production_components.values())
            )
            maximum_reference_total_delta = max(maximum_reference_total_delta, total_delta)
            maximum_reference_component_delta = max(
                maximum_reference_component_delta, component_delta,
            )
            maximum_authority_total_delta = max(
                maximum_authority_total_delta, authority_total_delta,
            )
            maximum_authority_component_delta = max(
                maximum_authority_component_delta, authority_component_delta,
            )
            path_mismatches += int(not path_matches or not authority_path_matches)
            symmetry_mismatches += int(symmetry_delta > REFERENCE_PARITY_TOLERANCE)
            nonfinite_outputs += int(not finite)
            pairs += 1
            if total_delta > REFERENCE_PARITY_TOLERANCE:
                failures.append(f"reference total parity failed {episode_id}/{match['episode_id']}")
            if component_delta > REFERENCE_PARITY_TOLERANCE:
                failures.append(f"reference component parity failed {episode_id}/{match['episode_id']}")
            if authority_total_delta > PRACTICAL_AUTHORITY_TOLERANCE:
                failures.append(f"authority total changed {episode_id}/{match['episode_id']}")
            if authority_component_delta > PRACTICAL_AUTHORITY_TOLERANCE:
                failures.append(f"authority component changed {episode_id}/{match['episode_id']}")
            if not path_matches or not authority_path_matches:
                failures.append(f"alignment path differs {episode_id}/{match['episode_id']}")
            if symmetry_delta > REFERENCE_PARITY_TOLERANCE:
                failures.append(f"symmetry failed {episode_id}/{match['episode_id']}")
            if not finite:
                failures.append(f"non-finite distance {episode_id}/{match['episode_id']}")
            if len(triangle_representations) < 3:
                triangle_representations.append(candidate_representation)
            pair_rows.append({
                "authority_rank": len(pair_rows) + 1,
                "candidate_episode_id": str(match["episode_id"]),
                "candidate_symbol": str(match["symbol"]),
                "candidate_cutoff": str(match["cutoff"]),
                "production_total": production_total,
                "reference_total": reference.total,
                "stored_authority_total": float(match["total_distance"]),
                "reference_total_delta": total_delta,
                "reference_component_max_delta": component_delta,
                "authority_total_delta": authority_total_delta,
                "authority_component_max_delta": authority_component_delta,
                "alignment_path_matches": path_matches and authority_path_matches,
                "symmetry_delta": symmetry_delta,
            })
        if len(triangle_representations) == 3:
            a, b, c = triangle_representations
            ab = reference_representation_distance(a, b).total
            bc = reference_representation_distance(b, c).total
            ac = reference_representation_distance(a, c).total
            triangle_violations += int(ac > ab + bc + REFERENCE_PARITY_TOLERANCE)
        case_rows.append({
            "query_episode_id": episode_id,
            "symbol": str(query_meta["symbol"]),
            "cutoff": str(query_meta["cutoff"]),
            "authority_digest": authority.authority_digest,
            "pairs": pair_rows,
        })
    if pairs != 240:
        failures.append(f"verified {pairs} authority pairs; require 240")
    if identity_mismatches:
        failures.append(f"identity failed for {identity_mismatches} queries")
    contract = distance_v1_contract()
    metrics = {
        "schema_version": M04R_DISTANCE_VERIFIER_SCHEMA,
        "dataset_id": dataset_id,
        "distance_contract_digest": contract["digest"],
        "m04_contract_digest": spec.digest,
        "authorities_verified": len(case_rows),
        "authority_pairs_verified": pairs,
        "maximum_reference_total_delta": maximum_reference_total_delta,
        "maximum_reference_component_delta": maximum_reference_component_delta,
        "maximum_authority_total_delta": maximum_authority_total_delta,
        "maximum_authority_component_delta": maximum_authority_component_delta,
        "reference_parity_tolerance": REFERENCE_PARITY_TOLERANCE,
        "authority_practical_tolerance": PRACTICAL_AUTHORITY_TOLERANCE,
        "alignment_path_mismatches": path_mismatches,
        "symmetry_mismatches": symmetry_mismatches,
        "identity_mismatches": identity_mismatches,
        "nonfinite_outputs": nonfinite_outputs,
        "triangle_violations_diagnostic_only": triangle_violations,
        "real_forward_outcomes_accessed": False,
    }
    deterministic = {
        "schema_version": M04R_DISTANCE_VERIFIER_SCHEMA,
        "metrics": metrics,
        "failures": sorted(failures),
        "cases": case_rows,
        "contract": contract,
    }
    return M04RDistanceVerificationResult(
        not failures, metrics, tuple(sorted(failures)), tuple(case_rows), contract,
        stable_hash(deterministic),
    )


def write_m04r_distance_verification(
    result: M04RDistanceVerificationResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_path = output_dir / "m04r-distance-v1.json"
    contract_path = output_dir / "distance-v1-contract.json"
    html_path = output_dir / "m04r-distance-v1.html"
    payload = {
        "schema_version": M04R_DISTANCE_VERIFIER_SCHEMA,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "cases": list(result.cases),
        "contract": result.contract,
        "result_digest": result.result_digest,
    }
    for path, value in ((machine_path, payload), (contract_path, result.contract)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    rows = "".join(
        f"<tr><td>{escape(str(row['symbol']))}</td><td>{escape(str(row['cutoff']))}</td>"
        f"<td>{len(row['pairs'])}</td><td><code>{escape(str(row['authority_digest']))}</code></td></tr>"
        for row in result.cases
    )
    failure_items = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    temporary = html_path.with_suffix(html_path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R distance-v1 audit</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}code{{font-size:.8em;overflow-wrap:anywhere}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M04R exact distance-v1 audit: <span class="{status.lower()}">{status}</span></h1><p>A separately implemented slow scorer is compared with production and every stored NASDAQ exact-authority pair. Outcomes and setup names are excluded.</p><p>Contract: <code>{result.contract['digest']}</code> · Result: <code>{result.result_digest}</code></p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Authorities</h2><table><thead><tr><th>Query</th><th>Cutoff</th><th>Pairs</th><th>Authority digest</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Contract summary</h2><pre>{escape(json.dumps(result.contract, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>""")
    temporary.replace(html_path)
    return machine_path, html_path, contract_path
