from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .distance import representation_distance, representation_distance_lower_bound
from .episodes import build_episode
from .representation import Representation, represent
from .types import InstrumentKey


LAYOUT_VERSION = "exact-representation-layout-v1"
STORAGE_LAYOUTS = ("native", "float32", "float16")


@dataclass(frozen=True)
class ExactStorageFeasibility:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    layouts: pd.DataFrame


def _canonical_names(representations: list[Representation]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names_48 = tuple(sorted(representations[0].samples_48))
    names_64 = tuple(sorted(representations[0].samples_64))
    for representation in representations[1:]:
        if tuple(sorted(representation.samples_48)) != names_48:
            raise ValueError("48-sample representation keys are inconsistent")
        if tuple(sorted(representation.samples_64)) != names_64:
            raise ValueError("64-sample representation keys are inconsistent")
    return names_48, names_64


def _pack(
    representations: list[Representation],
    layout: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if layout not in STORAGE_LAYOUTS:
        raise ValueError(f"unknown storage layout {layout}")
    names_48, names_64 = _canonical_names(representations)
    coarse_dtype = np.float32 if layout in {"native", "float32"} else np.float16
    value_dtype = {
        "native": np.float64, "float32": np.float32, "float16": np.float16,
    }[layout]
    coarse = np.stack([item.coarse for item in representations]).astype(coarse_dtype)
    masks: list[list[bool]] = []
    values: list[np.ndarray] = []
    for item in representations:
        present: list[bool] = []
        parts: list[np.ndarray] = []
        for name in names_48:
            value = item.samples_48[name]
            present.append(value is not None)
            parts.append(np.zeros(48) if value is None else np.asarray(value))
        for name in names_64:
            value = item.samples_64[name]
            present.append(value is not None)
            parts.append(np.zeros(64) if value is None else np.asarray(value))
        parts.extend([np.asarray(item.stage), np.asarray(item.structural)])
        masks.append(present)
        values.append(np.concatenate(parts))
    return (
        coarse,
        np.stack(values).astype(value_dtype),
        np.asarray(masks, dtype=np.bool_),
        names_48,
        names_64,
    )


def _unpack(
    coarse: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    names_48: tuple[str, ...],
    names_64: tuple[str, ...],
) -> list[Representation]:
    restored: list[Representation] = []
    for row in range(len(coarse)):
        offset = 0
        samples_48: dict[str, np.ndarray | None] = {}
        samples_64: dict[str, np.ndarray | None] = {}
        mask_position = 0
        for name in names_48:
            sample = values[row, offset:offset + 48].astype(np.float64)
            samples_48[name] = sample if masks[row, mask_position] else None
            offset += 48
            mask_position += 1
        for name in names_64:
            sample = values[row, offset:offset + 64].astype(np.float64)
            samples_64[name] = sample if masks[row, mask_position] else None
            offset += 64
            mask_position += 1
        stage = values[row, offset:offset + 48].astype(np.float64)
        offset += 48
        structural = values[row, offset:offset + 9].astype(np.float64)
        restored.append(Representation(
            pd.DataFrame(), coarse[row].astype(np.float32), samples_48,
            samples_64, stage, structural,
        ))
    return restored


def _payload_digest(
    coarse: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    names_48: tuple[str, ...],
    names_64: tuple[str, ...],
) -> str:
    digest = sha256()
    for array in (coarse, values, masks):
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    digest.update("\0".join(names_48).encode())
    digest.update(b"\0\0")
    digest.update("\0".join(names_64).encode())
    return digest.hexdigest()


def _serialize_layout(
    coarse: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    names_48: tuple[str, ...],
    names_64: tuple[str, ...],
    layout: str,
) -> bytes:
    if layout not in STORAGE_LAYOUTS:
        raise ValueError(f"unknown exact layout {layout!r}")
    metadata = {
        "layout_version": LAYOUT_VERSION,
        "layout": layout,
        "names_48": names_48,
        "names_64": names_64,
        "rows": len(coarse),
        "digest": _payload_digest(coarse, values, masks, names_48, names_64),
    }
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        coarse=coarse,
        values=values,
        masks=masks,
    )
    return buffer.getvalue()


def _deserialize_layout(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], str]:
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            required = {"metadata_json", "coarse", "values", "masks"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"exact layout is missing arrays: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            coarse = np.asarray(archive["coarse"])
            values = np.asarray(archive["values"])
            masks = np.asarray(archive["masks"])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot load exact layout: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("exact layout metadata must be an object")
    if metadata.get("layout_version") != LAYOUT_VERSION:
        raise ValueError(
            f"unsupported exact layout version {metadata.get('layout_version')!r}; "
            f"expected {LAYOUT_VERSION!r}"
        )
    layout = str(metadata.get("layout"))
    if layout not in STORAGE_LAYOUTS:
        raise ValueError(f"unknown exact layout {layout!r}")
    names_48 = tuple(str(value) for value in metadata.get("names_48", ()))
    names_64 = tuple(str(value) for value in metadata.get("names_64", ()))
    if coarse.ndim != 2 or values.ndim != 2 or masks.ndim != 2:
        raise ValueError("exact layout arrays must be two-dimensional")
    if not (len(coarse) == len(values) == len(masks) == int(metadata.get("rows", -1))):
        raise ValueError("exact layout row counts disagree")
    expected_value_columns = len(names_48) * 48 + len(names_64) * 64 + 48 + 9
    if (
        coarse.shape[1] != 128
        or values.shape[1] != expected_value_columns
        or masks.shape[1] != len(names_48) + len(names_64)
    ):
        raise ValueError("exact layout dimensions disagree with channel metadata")
    expected_coarse_dtype = np.dtype(
        np.float32 if layout in {"native", "float32"} else np.float16
    )
    expected_value_dtype = np.dtype({
        "native": np.float64, "float32": np.float32, "float16": np.float16,
    }[layout])
    if coarse.dtype != expected_coarse_dtype or values.dtype != expected_value_dtype:
        raise ValueError("exact layout dtypes disagree with layout metadata")
    if masks.dtype != np.dtype(np.bool_):
        raise ValueError("exact layout presence masks must be boolean")
    if not np.isfinite(coarse).all() or not np.isfinite(values).all():
        raise ValueError("exact layout contains non-finite values")
    actual_digest = _payload_digest(coarse, values, masks, names_48, names_64)
    if actual_digest != metadata.get("digest"):
        raise ValueError("exact layout content digest mismatch")
    return coarse, values, masks, names_48, names_64, layout


def _sample_representations(
    sources: dict[str, OHLCVSource],
    registries: dict[str, pd.DataFrame],
    *,
    sample_windows_per_symbol: int,
) -> tuple[list[Representation], list[str]]:
    representations: list[Representation] = []
    labels: list[str] = []
    for dataset_id in sorted(registries):
        source = sources[dataset_id]
        registry = registries[dataset_id]
        for symbol in sorted(registry.symbol.astype(str).unique()):
            row = registry[registry.symbol.astype(str) == symbol].iloc[0]
            lookback = int(row.lookback)
            bars = source.load(InstrumentKey(dataset_id, symbol))
            positions = np.unique(np.linspace(
                lookback - 1, len(bars) - 1,
                min(sample_windows_per_symbol, len(bars) - lookback + 1),
                dtype=int,
            ))
            for position in positions:
                episode = build_episode(
                    source, InstrumentKey(dataset_id, symbol),
                    bars.timestamp.iloc[position], lookback,
                    str(row.representation_version), str(row.quality_tier),
                )
                representations.append(represent(episode))
                labels.append(f"{dataset_id}:{symbol}:{pd.Timestamp(bars.timestamp.iloc[position]).isoformat()}")
    return representations, labels


def verify_exact_storage_feasibility(
    sources: dict[str, OHLCVSource],
    registries: dict[str, pd.DataFrame],
    *,
    total_universe_rows: int,
    sample_windows_per_symbol: int = 4,
    comparison_queries: int = 4,
    comparison_candidates: int = 8,
    tolerance: float = 1e-12,
    disk_path: Path = Path("."),
    disk_reserve_bytes: int = 5 * 1024 ** 3,
    frontier_bytes_per_row: int = 64,
    compression_safety_factor: float = 1.25,
    available_bytes: int | None = None,
) -> ExactStorageFeasibility:
    if sample_windows_per_symbol < 1:
        raise ValueError("sample_windows_per_symbol must be positive")
    if comparison_queries < 1 or comparison_candidates < 1:
        raise ValueError("comparison query and candidate counts must be positive")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    if disk_reserve_bytes < 0 or frontier_bytes_per_row < 0:
        raise ValueError("disk reserve and frontier bytes cannot be negative")
    if compression_safety_factor < 1:
        raise ValueError("compression_safety_factor must be at least one")
    if available_bytes is not None and available_bytes < 0:
        raise ValueError("available bytes cannot be negative")
    representations, labels = _sample_representations(
        sources, registries, sample_windows_per_symbol=sample_windows_per_symbol,
    )
    failures: list[str] = []
    if len(representations) < 2:
        failures.append("fewer than two exact representations were sampled")
    if total_universe_rows < 1:
        failures.append("total universe rows must be positive")
    if failures:
        return ExactStorageFeasibility(False, {}, tuple(failures), pd.DataFrame())

    query_count = min(comparison_queries, len(representations) - 1)
    query_positions = np.unique(np.linspace(
        0, len(representations) - 1, query_count, dtype=int,
    )).tolist()
    available_positions = [
        position for position in range(len(representations))
        if position not in set(query_positions)
    ]
    candidate_count = min(comparison_candidates, len(available_positions))
    candidate_positions = [
        available_positions[position] for position in np.unique(np.linspace(
            0, len(available_positions) - 1, candidate_count, dtype=int,
        ))
    ]
    baseline: dict[int, list[tuple[int, float, dict[str, float], float, dict[str, float], float]]] = {}
    for query_position in query_positions:
        measured = []
        for candidate_position in candidate_positions:
            lower, lower_components, rigid = representation_distance_lower_bound(
                representations[query_position], representations[candidate_position],
            )
            total, components, _ = representation_distance(
                representations[query_position], representations[candidate_position],
            )
            measured.append((
                candidate_position, total, components, lower, lower_components, rigid,
            ))
        baseline[query_position] = measured

    layout_rows: list[dict[str, object]] = []
    for layout in STORAGE_LAYOUTS:
        coarse, values, masks, names_48, names_64 = _pack(representations, layout)
        payload = _serialize_layout(
            coarse, values, masks, names_48, names_64, layout,
        )
        loaded = _deserialize_layout(payload)
        loaded_coarse, loaded_values, loaded_masks, loaded_names_48, loaded_names_64, loaded_layout = loaded
        if loaded_layout != layout:
            raise AssertionError("serialized layout identity changed during round trip")
        restored = _unpack(
            loaded_coarse, loaded_values, loaded_masks,
            loaded_names_48, loaded_names_64,
        )
        maximum_field_delta = 0.0
        for original, rebuilt in zip(representations, restored):
            maximum_field_delta = max(
                maximum_field_delta,
                float(np.max(np.abs(original.coarse.astype(float) - rebuilt.coarse.astype(float)))),
                float(np.max(np.abs(original.stage - rebuilt.stage))),
                float(np.max(np.abs(original.structural - rebuilt.structural))),
            )
            for name in names_48:
                left, right = original.samples_48[name], rebuilt.samples_48[name]
                if (left is None) != (right is None):
                    maximum_field_delta = float("inf")
                if left is not None and right is not None:
                    maximum_field_delta = max(
                        maximum_field_delta, float(np.max(np.abs(left - right))),
                    )
            for name in names_64:
                left, right = original.samples_64[name], rebuilt.samples_64[name]
                if (left is None) != (right is None):
                    maximum_field_delta = float("inf")
                if left is not None and right is not None:
                    maximum_field_delta = max(
                        maximum_field_delta, float(np.max(np.abs(left - right))),
                    )
        maximum_total_delta = 0.0
        maximum_component_delta = 0.0
        maximum_lower_bound_delta = 0.0
        maximum_lower_component_delta = 0.0
        maximum_rigid_price_delta = 0.0
        rankings_equal = True
        for query_position, expected in baseline.items():
            actual: list[tuple[int, float]] = []
            for (
                candidate_position, expected_total, expected_components,
                expected_lower, expected_lower_components, expected_rigid,
            ) in expected:
                lower, lower_components, rigid = representation_distance_lower_bound(
                    restored[query_position], restored[candidate_position],
                )
                total, components, _ = representation_distance(
                    restored[query_position], restored[candidate_position],
                )
                actual.append((candidate_position, total))
                maximum_total_delta = max(maximum_total_delta, abs(total - expected_total))
                maximum_lower_bound_delta = max(
                    maximum_lower_bound_delta, abs(lower - expected_lower),
                )
                maximum_rigid_price_delta = max(
                    maximum_rigid_price_delta, abs(rigid - expected_rigid),
                )
                for name, value in expected_components.items():
                    maximum_component_delta = max(
                        maximum_component_delta, abs(components[name] - value),
                    )
                for name, value in expected_lower_components.items():
                    maximum_lower_component_delta = max(
                        maximum_lower_component_delta,
                        abs(lower_components[name] - value),
                    )
            expected_order = [
                item[0] for item in sorted(expected, key=lambda item: (item[1], item[0]))
            ]
            actual_order = [
                position for position, _ in sorted(actual, key=lambda item: (item[1], item[0]))
            ]
            rankings_equal &= expected_order == actual_order
        compressed_bytes = len(payload)
        raw_bytes = coarse.nbytes + values.nbytes + masks.nbytes
        compressed_bytes_per_row = compressed_bytes / len(representations)
        projected_store_bytes = int(
            compressed_bytes_per_row * total_universe_rows * compression_safety_factor
        )
        fidelity_passed = (
            maximum_total_delta <= tolerance
            and maximum_component_delta <= tolerance
            and maximum_lower_bound_delta <= tolerance
            and maximum_lower_component_delta <= tolerance
            and maximum_rigid_price_delta <= tolerance
            and maximum_field_delta <= tolerance
            and rankings_equal
        )
        layout_rows.append({
            "layout": layout,
            "coarse_dtype": str(coarse.dtype),
            "value_dtype": str(values.dtype),
            "dimensions": coarse.shape[1] + values.shape[1],
            "presence_mask_bits": masks.shape[1],
            "raw_bytes_per_row": raw_bytes / len(representations),
            "compressed_bytes_per_row_sample": compressed_bytes_per_row,
            "sample_compression_ratio": compressed_bytes / raw_bytes,
            "projected_store_bytes_with_safety": projected_store_bytes,
            "maximum_field_delta": maximum_field_delta,
            "maximum_total_delta": maximum_total_delta,
            "maximum_component_delta": maximum_component_delta,
            "maximum_lower_bound_delta": maximum_lower_bound_delta,
            "maximum_lower_component_delta": maximum_lower_component_delta,
            "maximum_rigid_price_delta": maximum_rigid_price_delta,
            "rankings_equal": rankings_equal,
            "fidelity_passed": fidelity_passed,
        })

    layouts = pd.DataFrame(layout_rows)
    native = layouts.set_index("layout").loc["native"]
    if not bool(native.fidelity_passed):
        failures.append("native lossless layout failed exact fidelity")
    available = (
        int(available_bytes) if available_bytes is not None
        else int(shutil.disk_usage(disk_path).free)
    )
    frontier_bytes = total_universe_rows * frontier_bytes_per_row
    required_native = int(native.projected_store_bytes_with_safety) + frontier_bytes
    full_store_feasible = (
        bool(native.fidelity_passed)
        and required_native + disk_reserve_bytes <= available
    )
    selected_design = "lossless_exact_store" if full_store_feasible else "two_pass_streaming"
    metrics: dict[str, object] = {
        "layout_version": LAYOUT_VERSION,
        "sample_representations": len(representations),
        "sample_datasets": sorted(registries),
        "sample_labels_digest": sha256(
            "\0".join(labels).encode()
        ).hexdigest(),
        "distance_query_positions": query_positions,
        "distance_candidate_positions": candidate_positions,
        "distance_comparisons": len(query_positions) * len(candidate_positions),
        "samples_48_channels": len(_canonical_names(representations)[0]),
        "samples_64_channels": len(_canonical_names(representations)[1]),
        "exact_numeric_dimensions": int(native.dimensions),
        "total_universe_rows": total_universe_rows,
        "available_disk_bytes": available,
        "required_disk_reserve_bytes": disk_reserve_bytes,
        "projected_frontier_bytes": frontier_bytes,
        "projected_native_store_bytes_with_safety": int(
            native.projected_store_bytes_with_safety
        ),
        "projected_native_total_bytes": required_native,
        "full_exact_store_feasible": full_store_feasible,
        "selected_design": selected_design,
        "tolerance": tolerance,
        "compression_safety_factor": compression_safety_factor,
    }
    return ExactStorageFeasibility(not failures, metrics, tuple(failures), layouts)


def write_exact_storage_report(result: ExactStorageFeasibility, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{escape(str(row.layout))}</td><td>{escape(str(row.coarse_dtype))} / "
        f"{escape(str(row.value_dtype))}</td><td>{int(row.dimensions)}</td>"
        f"<td>{float(row.raw_bytes_per_row):,.0f}</td>"
        f"<td>{float(row.compressed_bytes_per_row_sample):,.0f}</td>"
        f"<td>{float(row.projected_store_bytes_with_safety) / 1024 ** 3:.2f} GiB</td>"
        f"<td>{float(row.maximum_total_delta):.2e}</td>"
        f"<td>{float(row.maximum_component_delta):.2e}</td>"
        f"<td>{'YES' if row.rankings_equal else 'NO'}</td>"
        f"<td>{'PASS' if row.fidelity_passed else 'FAIL'}</td></tr>"
        for row in result.layouts.itertuples(index=False)
    )
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact representation storage feasibility</title><style>body{{font-family:system-ui,sans-serif;max-width:1450px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}section,header{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{overflow:auto}}</style></head><body><header><h1>Exact-storage feasibility: <span class="{status.lower()}">{status}</span></h1><p>This gate measures the actual native representation precision, lossless compression and reduced-precision error. It selects a full store only when exact fidelity and the disk reserve both pass; otherwise the successful decision is the memory-bounded two-pass streaming authority.</p></header><section><h2>Decision</h2><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section><h2>Measured layouts</h2><table><thead><tr><th>Layout</th><th>Dtypes</th><th>Values</th><th>Raw B/row</th><th>Compressed B/row</th><th>Projected store</th><th>Max total Δ</th><th>Max component Δ</th><th>Ranks</th><th>Fidelity</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>"""
    path.write_text(html)
    return path
