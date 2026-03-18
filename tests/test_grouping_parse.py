"""Tests for --group-by spec parsing."""

import pytest

from claude_logs.grouping import parse_group_by_spec
from claude_logs.models import GroupByConfig


class TestParseGroupBySpec:
    def test_project_only(self):
        result = parse_group_by_spec("project")
        assert result.by_project is True
        assert result.time_format is None
        assert result.project_first is True

    def test_time_only(self):
        result = parse_group_by_spec("time:%Y%m%d%H")
        assert result.by_project is False
        assert result.time_format == "%Y%m%d%H"

    def test_project_then_time(self):
        result = parse_group_by_spec("project,time:%Y%m%d")
        assert result.by_project is True
        assert result.time_format == "%Y%m%d"
        assert result.project_first is True

    def test_time_then_project_raises(self):
        """time-then-project ordering is not yet supported in v0.4.0."""
        with pytest.raises(ValueError, match="not yet supported"):
            parse_group_by_spec("time:%H,project")

    def test_duplicate_project_deduplicated(self):
        result = parse_group_by_spec("project,project")
        assert result.by_project is True
        assert result.time_format is None

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="invalid group-by key"):
            parse_group_by_spec("foo")

    def test_empty_time_pattern_raises(self):
        with pytest.raises(ValueError, match="invalid group-by key"):
            parse_group_by_spec("time:")

    def test_multiple_time_keys_raises(self):
        with pytest.raises(ValueError, match="only one time:"):
            parse_group_by_spec("time:%H,time:%Y")

    def test_pure_literal_strftime_raises(self):
        """A pattern that produces no formatting (e.g., just 'hello') is invalid."""
        with pytest.raises(ValueError, match="invalid strftime"):
            parse_group_by_spec("time:hello")
