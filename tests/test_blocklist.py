from kagent.search.blocklist import ENTRY_CHARS, build_blocklist
from kagent.search.tree import Node, SolutionTree


def _tree(tmp_path, higher=True):
    return SolutionTree(tmp_path / "nodes.jsonl", higher_is_better=higher)


def _add(tree, tmp_path, *, parent=None, status="ok", cv=None, code="import os\n", **kw):
    node_id = tree.next_id()
    path = tmp_path / f"{node_id}.py"
    path.write_text(code)
    return tree.add(
        Node(
            id=node_id,
            parent=parent,
            operator=kw.pop("operator", "draft" if parent is None else "improve"),
            code_path=path,
            status=status,
            cv_score=cv,
            cv_std=0.01 if cv is not None else None,
            **kw,
        )
    )


def test_nothing_has_failed_means_an_empty_blocklist(tmp_path):
    tree = _tree(tmp_path)
    assert build_blocklist(tree) == []
    root = _add(tree, tmp_path, cv=0.80)
    _add(tree, tmp_path, parent=root.id, cv=0.81)  # a gain is not a failure
    assert build_blocklist(tree) == []


def test_dead_node_is_listed_by_its_plan_header_and_last_error_line(tmp_path):
    tree = _tree(tmp_path)
    root = _add(tree, tmp_path, cv=0.80)
    _add(
        tree,
        tmp_path,
        parent=root.id,
        status="dead",
        code="# PLAN: target-encode the zip code\n# - then LightGBM\nimport pandas\n# later\n",
        error="Traceback (most recent call last):\n  File 'x.py', line 3\nKeyError: 'zip'\n",
    )
    assert build_blocklist(tree) == [
        "target-encode the zip code - then LightGBM -> failed with KeyError: 'zip'"
    ]


def test_dead_node_without_a_plan_header_still_gets_an_entry(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, tmp_path, status="dead", error="MemoryError")
    assert build_blocklist(tree) == ["an earlier attempt -> failed with MemoryError"]


def test_improvement_that_lost_ground_is_listed_with_both_scores(tmp_path):
    tree = _tree(tmp_path)
    root = _add(tree, tmp_path, cv=0.80)
    _add(tree, tmp_path, parent=root.id, cv=0.79, summary="Added TF-IDF to the name column")
    assert build_blocklist(tree) == [
        "Added TF-IDF to the name column -> cv 0.79000 against 0.80000 before: no gain"
    ]


def test_an_equal_score_counts_as_no_gain(tmp_path):
    tree = _tree(tmp_path)
    root = _add(tree, tmp_path, cv=0.80)
    _add(tree, tmp_path, parent=root.id, cv=0.80, summary="Swapped the seed")
    assert len(build_blocklist(tree)) == 1


def test_no_gain_follows_the_metric_direction(tmp_path):
    tree = _tree(tmp_path, higher=False)
    root = _add(tree, tmp_path, cv=0.50)
    _add(tree, tmp_path, parent=root.id, cv=0.48, summary="better")
    _add(tree, tmp_path, parent=root.id, cv=0.52, summary="worse")
    assert build_blocklist(tree) == ["worse -> cv 0.52000 against 0.50000 before: no gain"]


def test_nodes_still_being_debugged_or_waiting_are_not_listed(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, tmp_path, status="buggy", error="KeyError: 'a'")
    _add(tree, tmp_path, status="blocked", error="No module named 'catboost'")
    assert build_blocklist(tree) == []


def test_newest_first_without_duplicates_and_capped(tmp_path):
    tree = _tree(tmp_path)
    _add(tree, tmp_path, status="dead", code="# PLAN: a\n", error="E1")
    _add(tree, tmp_path, status="dead", code="# PLAN: b\n", error="E2")
    _add(tree, tmp_path, status="dead", code="# PLAN: b\n", error="E2")

    assert build_blocklist(tree) == ["b -> failed with E2", "a -> failed with E1"]
    assert build_blocklist(tree, limit=1) == ["b -> failed with E2"]


def test_missing_package_failures_go_stale_once_the_package_is_installed(tmp_path):
    tree = _tree(tmp_path)
    _add(
        tree,
        tmp_path,
        status="dead",
        code="# PLAN: catboost\n",
        error="Traceback\nModuleNotFoundError: No module named 'catboost'\n",
    )
    _add(
        tree,
        tmp_path,
        status="dead",
        code="# PLAN: sklearn pipeline\n",
        error="Traceback\nModuleNotFoundError: No module named 'sklearn'\n",
    )

    assert len(build_blocklist(tree)) == 2
    assert len(build_blocklist(tree, {"catboost"})) == 1  # only the sklearn one is left
    assert build_blocklist(tree, {"catboost", "scikit-learn"}) == []  # import name -> dist name


def test_children_of_the_improved_node_are_skipped_because_the_prompt_lists_them(tmp_path):
    tree = _tree(tmp_path)
    first = _add(tree, tmp_path, cv=0.80)
    _add(tree, tmp_path, parent=first.id, cv=0.79, summary="shown elsewhere")
    second = _add(tree, tmp_path, cv=0.70)
    _add(tree, tmp_path, parent=second.id, cv=0.60, summary="other branch")

    entries = build_blocklist(tree, skip_children_of=first.id)

    assert len(entries) == 1 and entries[0].startswith("other branch")


def test_long_descriptions_are_clipped(tmp_path):
    tree = _tree(tmp_path)
    root = _add(tree, tmp_path, cv=0.80)
    _add(tree, tmp_path, parent=root.id, cv=0.70, summary="word " * 100)

    approach = build_blocklist(tree)[0].split(" -> ")[0]

    assert len(approach) <= ENTRY_CHARS and approach.endswith("...")
