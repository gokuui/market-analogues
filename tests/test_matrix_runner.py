from pathlib import Path

import yaml

from market_analogues.matrix_runner import (
    authority_matrix_cases, run_authority_matrix,
)


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "datasets.yaml"
    artifacts = tmp_path / "artifacts"
    config.write_text(yaml.safe_dump({
        "artifact_dir": str(artifacts),
        "datasets": {
            dataset: {
                "adapter": "directory", "path": str(tmp_path / dataset),
                "format": "parquet",
            }
            for dataset in ("nse", "nasdaq")
        },
    }))
    for dataset in ("nse", "nasdaq"):
        directory = artifacts / "gate12" / dataset
        directory.mkdir(parents=True)
        (directory / "query-registry.yaml").write_text(yaml.safe_dump({
            "cases_data": [
                {
                    "case_id": f"{dataset}-case-{index:02d}",
                    "episode_id": f"{dataset}-episode-{index:02d}",
                }
                for index in range(12)
            ],
        }))
    return config


def test_matrix_runner_orders_all_cases_and_aggregates_each_dataset(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    commands: list[tuple[str, ...]] = []

    def runner(command) -> int:
        commands.append(tuple(command))
        return 0

    cases = authority_matrix_cases(config, ("nse", "nasdaq"))
    assert len(cases) == 24
    assert cases[0].case_id == "nse-case-00"
    assert cases[-1].case_id == "nasdaq-case-11"
    result = run_authority_matrix(
        config, command_runner=runner, disk_reserve_bytes=0,
    )
    assert result.passed and result.completed_cases == result.required_cases == 24
    build_commands = [command for command in commands if "build-gate12-authority" in command]
    aggregate_commands = [
        command for command in commands if "aggregate-gate12-authorities" in command
    ]
    assert len(build_commands) == 24 and len(aggregate_commands) == 2
    assert build_commands[0][build_commands[0].index("--case-id") + 1] == "nse-case-00"
    assert build_commands[-1][build_commands[-1].index("--case-id") + 1] == "nasdaq-case-11"
    payload = yaml.safe_load(result.progress_path.read_text())
    assert payload["status"] == "passed"
    assert payload["completed_cases"] == 24
    assert set(payload["aggregates"].values()) == {"passed"}


def test_matrix_runner_stops_on_first_failed_case(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls = 0

    def runner(_command) -> int:
        nonlocal calls
        calls += 1
        return 2 if calls == 4 else 0

    result = run_authority_matrix(
        config, datasets=("nse",), command_runner=runner,
        disk_reserve_bytes=0,
    )
    assert not result.passed
    assert result.completed_cases == 3
    assert result.failed_case == "nse-case-03"
    payload = yaml.safe_load(result.progress_path.read_text())
    assert payload["status"] == "failed"
    assert payload["cases"]["nse-case-03"]["status"] == "failed"
    assert payload["cases"]["nse-case-04"]["status"] == "pending"
