from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_e2e_nse_authority_matrix as matrix


def test_nse_authority_preregistration_covers_complete_runtime_manifest() -> None:
    expected = {
        matrix.THIS_RUNTIME,
        *matrix._git(
            ROOT, "ls-files", "--", "src/market_analogues/*.py",
        ).splitlines(),
    }
    assert set(matrix._runtime_files(ROOT)) == expected
