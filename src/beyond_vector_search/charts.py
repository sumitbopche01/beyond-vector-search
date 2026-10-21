"""charts.py — SVG comparison of one eval run.

No plotting dependency. The file sits next to the markdown report so a
GitHub diff shows when a system gains or loses.
"""

from __future__ import annotations

from beyond_vector_search.evaluate import Row

_COLORS = ("#1d4e89", "#b86e00", "#2f6f4e", "#6b4c7a")


def render_comparison_chart(rows: list[Row]) -> str:
    """Grouped bars for keyword accuracy and citation support, then by question type."""
    if not rows:
        return _svg(520, 80, [
            "<title>Eval comparison</title>",
            '<text x="16" y="36" font-size="14" fill="#222">No eval rows.</text>',
        ])

    systems = list(dict.fromkeys(row.system for row in rows))
    types = list(dict.fromkeys(row.question.type for row in rows))
    grouped = {name: [row for row in rows if row.system == name] for name in systems}
    has_citations = any(row.citation is not None for row in rows)

    keyword = [_percent(row.keyword_ok for row in grouped[name]) for name in systems]
    series: list[tuple[str, list[float], str]] = [("keyword accuracy", keyword, _COLORS[0])]
    if has_citations:
        cited = [
            _percent(
                row.citation.grounded
                for row in grouped[name]
                if row.citation is not None
            )
            for name in systems
        ]
        series.append(("citation support", cited, _COLORS[1]))

    type_series = [
        (
            name,
            [
                _percent(row.keyword_ok for row in grouped[name] if row.question.type == kind)
                for kind in types
            ],
            _COLORS[index % len(_COLORS)],
        )
        for index, name in enumerate(systems)
    ]
    n = len(rows) // len(systems) if systems else 0
    width = max(640, 80 + max(len(systems), len(types)) * 110)
    plot_h = 180
    top = 56
    gap = 72
    second = top + plot_h + gap
    height = second + plot_h + 56
    parts = [
        "<title>Keyword accuracy and citation support by system</title>",
        "<desc>One eval run. Percents are the share of questions. Not a leaderboard.</desc>",
        _text(16, 28, "Keyword accuracy and citation support by system", 16, "#222"),
        _text(16, 46, f"Percent of questions · n={n} per system · one run, not a leaderboard", 12, "#555"),
    ]
    parts.extend(_legend(16, top - 4, series))
    parts.extend(_bars(systems, series, 56, top + 8, width - 80, plot_h))
    parts.append(_text(16, second - 28, "Keyword accuracy by question type", 16, "#222"))
    parts.append(_text(16, second - 10, "Percent of questions of that type", 12, "#555"))
    parts.extend(_legend(16, second + 8, type_series))
    parts.extend(_bars(types, type_series, 56, second + 20, width - 80, plot_h))
    return _svg(width, height, parts)


def _percent(flags) -> float:
    values = list(flags)
    if not values:
        return 0.0
    return 100.0 * sum(bool(item) for item in values) / len(values)


def _bars(
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    origin_x: float,
    origin_y: float,
    plot_w: float,
    plot_h: float,
) -> list[str]:
    if not categories or not series:
        return []
    slot = plot_w / len(categories)
    count = len(series)
    bar_w = min(26.0, max(8.0, (slot * 0.7) / count))
    parts: list[str] = []
    for pct in (0, 25, 50, 75, 100):
        y = origin_y + plot_h - (pct / 100) * plot_h
        parts.append(
            f'<line x1="{origin_x:.1f}" y1="{y:.1f}" x2="{origin_x + plot_w:.1f}" '
            f'y2="{y:.1f}" stroke="#e6e6e6"/>'
        )
        parts.append(_text(origin_x - 8, y + 4, str(pct), 11, "#444", anchor="end"))
    parts.append(
        f'<line x1="{origin_x:.1f}" y1="{origin_y + plot_h:.1f}" '
        f'x2="{origin_x + plot_w:.1f}" y2="{origin_y + plot_h:.1f}" stroke="#222"/>'
    )
    for index, category in enumerate(categories):
        center = origin_x + index * slot + slot / 2
        group_w = count * bar_w
        for series_index, (_name, values, color) in enumerate(series):
            value = 0.0 if index >= len(values) else max(0.0, min(100.0, values[index]))
            height = (value / 100) * plot_h
            x = center - group_w / 2 + series_index * bar_w
            y = origin_y + plot_h - height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" height="{max(height, 0):.1f}" '
                f'fill="{color}"><title>{_xml(category)} {value:.0f}%</title></rect>'
            )
            if height > 14:
                parts.append(_text(x + (bar_w - 3) / 2, y + 12, f"{value:.0f}", 10, "#fff", anchor="middle"))
        parts.append(_text(center, origin_y + plot_h + 16, category, 11, "#222", anchor="middle"))
    return parts


def _legend(x: float, y: float, series: list[tuple[str, list[float], str]]) -> list[str]:
    parts: list[str] = []
    cursor = x
    for name, _values, color in series:
        parts.append(f'<rect x="{cursor:.1f}" y="{y - 10:.1f}" width="10" height="10" fill="{color}"/>')
        parts.append(_text(cursor + 14, y, name, 11, "#222"))
        cursor += 14 + len(name) * 6.4 + 18
    return parts


def _text(x: float, y: float, text: str, size: int, fill: str, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="{fill}" font-family="ui-sans-serif, sans-serif">{_xml(text)}</text>'
    )


def _xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg(width: int, height: int, parts: list[str]) -> str:
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n{body}\n</svg>\n'
    )
