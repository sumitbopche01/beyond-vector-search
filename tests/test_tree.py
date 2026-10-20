"""Heading trees: line ranges nest, and summaries merge from model JSON."""

from beyond_vector_search.tree import apply_summaries, build_tree, render_nodes


def test_sections_run_until_the_next_sibling() -> None:
    lines = ["# A", "intro", "## B", "body", "# C", "end"]
    nodes = build_tree(lines)

    assert [node.title for node in nodes] == ["A", "C"]
    assert nodes[0].start_line == 1
    assert nodes[0].end_line == 4
    assert nodes[0].children[0].title == "B"
    assert nodes[0].children[0].end_line == 4
    assert nodes[1].start_line == 5
    assert nodes[1].end_line == 6


def test_summaries_accept_a_fenced_json_payload() -> None:
    nodes = build_tree(["# A", "body"])
    apply_summaries(nodes, '```json\n[{"start_line": 1, "summary": "The top of the doc"}]\n```')
    rendered = render_nodes(nodes)
    assert "The top of the doc" in rendered
    assert "1-2" in rendered
