from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from .types import AnalogueMatch, Episode


def _fmt(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.4f}"


def write_search_report(
    query: Episode,
    matches: list[AnalogueMatch],
    path: Path,
    outcome_summary: pd.DataFrame | None = None,
    provenance: dict[str, str] | None = None,
) -> Path:
    rows = []
    for rank, match in enumerate(matches, 1):
        components = " ".join(
            f"<span><b>{escape(name)}</b> {_fmt(value)}</span>"
            for name, value in sorted(match.component_distances.items())
        )
        rows.append(
            f"<tr><td>{rank}</td><td>{escape(str(match.episode_key.instrument))}</td>"
            f"<td>{escape(match.episode_key.cutoff.date().isoformat())}</td>"
            f"<td>{_fmt(match.total_distance)}</td><td class='components'>{components}</td>"
            f"<td>{escape(match.quality_tier)}</td></tr>"
        )
    outcome_html = "<p>No complete historical outcomes were supplied.</p>"
    if outcome_summary is not None and len(outcome_summary):
        outcome_html = outcome_summary.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
    provenance_rows = "".join(
        f"<tr><th>{escape(k)}</th><td><code>{escape(v)}</code></td></tr>"
        for k, v in sorted((provenance or {}).items())
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Historical analogues — {escape(str(query.key.instrument))}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#17202a;background:#f7f8fa}}
h1,h2{{line-height:1.15}} .notice{{padding:1rem;border-left:5px solid #b9770e;background:#fff4dc}}
table{{width:100%;border-collapse:collapse;background:white;margin:1rem 0}} th,td{{text-align:left;padding:.65rem;border-bottom:1px solid #ddd;vertical-align:top}}
.components{{display:flex;gap:.7rem;flex-wrap:wrap;font-size:.85rem}} code{{overflow-wrap:anywhere}} .meta{{color:#566573}}
</style></head><body>
<h1>Historical chart analogues</h1>
<p class="meta">Query: <b>{escape(str(query.key.instrument))}</b> through {escape(query.key.cutoff.date().isoformat())}; lookback {query.key.lookback} bars. Generated {datetime.now(timezone.utc).isoformat()}.</p>
<p class="notice"><b>Descriptive evidence, not a forecast.</b> Ranking uses information available at each cutoff. Subsequent returns are displayed only after retrieval and never enter similarity.</p>
<h2>Nearest episodes</h2><table><thead><tr><th>#</th><th>Instrument</th><th>Historical cutoff</th><th>Distance ↓</th><th>Distance breakdown</th><th>Quality</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>What happened next across these matches</h2>{outcome_html}
<h2>Provenance</h2><table>{provenance_rows}</table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
