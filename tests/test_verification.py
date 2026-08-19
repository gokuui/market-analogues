from market_analogues.verification import run_synthetic_verifier


def test_synthetic_verifier_executes_all_gates() -> None:
    report = run_synthetic_verifier(seeds_per_family=1)
    assert report.passed
    assert report.metrics["exact_clone_rank1"] == 1
    assert report.metrics["future_mutation_delta"] == 0
    assert report.metrics["coarse_index_roundtrip_recall_at_50"] == 1
