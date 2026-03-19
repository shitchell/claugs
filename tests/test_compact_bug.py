"""Tests for --compact flag behavior."""

import sys
from unittest.mock import patch

from claude_logs.cli import parse_args, _build_filters


class TestCompactFlag:
    def test_compact_hides_thinking(self):
        """--compact alone should hide thinking."""
        with patch.object(sys, "argv", ["claugs", "show", "--compact", "--latest"]):
            _, args = parse_args()
        assert args.compact is True
        filters = _build_filters(args)
        assert filters.is_visible("thinking") is False

    def test_compact_with_explicit_override(self):
        """--compact --show thinking should show thinking (explicit wins)."""
        with patch.object(
            sys,
            "argv",
            ["claugs", "show", "--compact", "--show", "thinking", "--latest"],
        ):
            _, args = parse_args()
        assert args.compact is True
        filters = _build_filters(args)
        assert filters.is_visible("thinking") is True

    def test_no_compact_defaults(self):
        """Without --compact, thinking/tools/metadata have default visibility."""
        with patch.object(sys, "argv", ["claugs", "show", "--latest"]):
            _, args = parse_args()
        assert args.compact is False
        filters = _build_filters(args)
        assert filters.is_visible("thinking") is True
        assert filters.is_visible("tools") is True
        assert filters.is_visible("metadata") is False  # default hidden
