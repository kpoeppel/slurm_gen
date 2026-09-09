"""Tests for node-exclusion list parsing."""

import logging
from pathlib import Path

from slurm_gen.exclude import read_exclude_nodes


class TestReadExcludeNodes:
    """Tests for read_exclude_nodes."""

    def test_one_node_per_line(self, tmp_path: Path):
        target = tmp_path / "excluded.txt"
        target.write_text("node01\nnode02\nnode03\n")

        assert read_exclude_nodes(target) == "node01,node02,node03"

    def test_ignores_comments_and_blank_lines(self, tmp_path: Path):
        target = tmp_path / "excluded.txt"
        target.write_text("# ruled out 2026-01-01\nnode01\n\n   \n# note\nnode02\n")

        assert read_exclude_nodes(target) == "node01,node02"

    def test_accepts_comma_and_whitespace_separated_entries(self, tmp_path: Path):
        target = tmp_path / "excluded.txt"
        target.write_text("node01,node02 node03\n\tnode04\n")

        assert read_exclude_nodes(target) == "node01,node02,node03,node04"

    def test_deduplicates_preserving_first_seen_order(self, tmp_path: Path):
        target = tmp_path / "excluded.txt"
        target.write_text("node03\nnode01\nnode03\nnode02\nnode01\n")

        assert read_exclude_nodes(target) == "node03,node01,node02"

    def test_custom_separator(self, tmp_path: Path):
        target = tmp_path / "excluded.txt"
        target.write_text("node01\nnode02\n")

        assert read_exclude_nodes(target, sep=" ") == "node01 node02"

    def test_empty_file_returns_none(self, tmp_path: Path):
        target = tmp_path / "excluded.txt"
        target.write_text("# nothing excluded yet\n\n")

        assert read_exclude_nodes(target) is None

    def test_missing_file_returns_none_and_warns(self, tmp_path: Path, caplog):
        """Silence here would turn a typo'd path into a job with no exclusions."""
        missing = tmp_path / "nope.txt"

        with caplog.at_level(logging.WARNING, logger="slurm_gen.exclude"):
            assert read_exclude_nodes(missing) is None

        assert "does not exist" in caplog.text
        assert str(missing) in caplog.text

    def test_expands_user_path(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "excluded.txt").write_text("node01\n")

        assert read_exclude_nodes("~/excluded.txt") == "node01"
