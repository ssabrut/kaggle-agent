import os

from kagent.integrity.hashing import find_changes, hash_tree, is_regular_file, snapshot


def test_detects_modified_added_removed(tmp_path):
    (tmp_path / "a.csv").write_text("1")
    (tmp_path / "b.csv").write_text("2")
    before = snapshot({"data": tmp_path})

    (tmp_path / "a.csv").write_text("changed")
    (tmp_path / "b.csv").unlink()
    (tmp_path / "c.csv").write_text("3")

    assert find_changes(before, snapshot({"data": tmp_path})) == [
        "data/a.csv modified",
        "data/b.csv removed",
        "data/c.csv added",
    ]


def test_untouched_tree_has_no_changes(tmp_path):
    (tmp_path / "a.csv").write_text("1")
    before = snapshot({"data": tmp_path})
    assert find_changes(before, snapshot({"data": tmp_path})) == []


def test_pycache_is_ignored(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1")
    before = hash_tree(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
    assert hash_tree(tmp_path) == before


def test_new_symlink_counts_as_change(tmp_path):
    (tmp_path / "a.csv").write_text("1")
    before = snapshot({"data": tmp_path})
    os.symlink("/etc/passwd", tmp_path / "link")
    assert find_changes(before, snapshot({"data": tmp_path})) == ["data/link added"]


def test_is_regular_file_rejects_symlink_and_missing(tmp_path):
    real = tmp_path / "real"
    real.write_text("x")
    os.symlink(real, tmp_path / "link")
    assert is_regular_file(real)
    assert not is_regular_file(tmp_path / "link")
    assert not is_regular_file(tmp_path / "missing")
    assert not is_regular_file(tmp_path)
