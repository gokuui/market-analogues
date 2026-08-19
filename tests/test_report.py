from __future__ import annotations

from market_analogues.report import write_search_report
from market_analogues.search import SearchCandidate, exact_search
from market_analogues.synthetic import generate_case
from market_analogues.types import SearchQuery


def test_report_is_self_contained_and_has_warning(tmp_path) -> None:
    original_query = generate_case("steady_trend", 1)
    from market_analogues.synthetic import transform_case
    query = transform_case(original_query, name="later", time_shift_days=1000).episode
    candidate = generate_case("steady_trend", 2).episode
    matches = exact_search(
        query, [SearchCandidate.from_episode(candidate)],
        SearchQuery(query.key, top_k=1),
    )
    path = write_search_report(query, matches, tmp_path / "report.html", provenance={"version": "test"})
    text = path.read_text()
    assert "<!doctype html>" in text
    assert "not a forecast" in text
    assert "Distance breakdown" in text
    assert "version" in text
