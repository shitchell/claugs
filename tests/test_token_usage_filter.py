"""Tests for the token-usage content filter."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_logs.cli import main
from claude_logs.models import (
    FilterConfig,
    RenderConfig,
    get_filter_registry,
    parse_message,
)
from conftest import create_session_file


FIXTURE = Path(__file__).parent / "fixtures" / "v2.1.77" / "complete_session.jsonl"


def _run(argv: list[str], capsys) -> tuple[int, str]:
    with patch.object(sys, "argv", argv):
        code = main()
    return code, capsys.readouterr().out


# =============================================================================
# 1. Registry
# =============================================================================


class TestTokenUsageRegistry:
    def test_registered(self):
        reg = get_filter_registry()
        assert "token-usage" in reg

    def test_hidden_by_default(self):
        reg = get_filter_registry()
        assert reg["token-usage"]["default_visible"] is False

    def test_category_is_content(self):
        reg = get_filter_registry()
        assert reg["token-usage"]["category"] == "content"

    def test_description(self):
        reg = get_filter_registry()
        assert "token" in reg["token-usage"]["description"].lower()

    def test_metadata_description_no_token(self):
        """metadata description should not mention token usage (separate filter)."""
        reg = get_filter_registry()
        assert "token" not in reg["metadata"]["description"].lower()


# =============================================================================
# 2. FilterConfig visibility
# =============================================================================


class TestTokenUsageFilterConfig:
    def test_hidden_by_default(self):
        fc = FilterConfig()
        assert fc.is_visible("token-usage") is False

    def test_show_makes_visible(self):
        fc = FilterConfig(shown={"token-usage"})
        assert fc.is_visible("token-usage") is True

    def test_hide_keeps_hidden(self):
        fc = FilterConfig(hidden={"token-usage"})
        assert fc.is_visible("token-usage") is False

    def test_show_overrides_hide(self):
        fc = FilterConfig(shown={"token-usage"}, hidden={"token-usage"})
        assert fc.is_visible("token-usage") is True

    def test_independent_of_metadata(self):
        """token-usage visibility should not depend on metadata visibility."""
        fc = FilterConfig(shown={"token-usage"})
        assert fc.is_visible("token-usage") is True
        assert fc.is_visible("metadata") is False


# =============================================================================
# 3. Render-level tests (AssistantMessage)
# =============================================================================


class TestAssistantTokenUsageRender:
    def _make_assistant(self):
        return {
            "type": "assistant",
            "uuid": "tok-001",
            "sessionId": "tok-session",
            "timestamp": "2026-03-17T10:00:00.000Z",
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 500, "output_tokens": 200},
            },
        }

    def test_hidden_by_default(self):
        msg = parse_message(self._make_assistant())
        config = RenderConfig(filters=FilterConfig())
        blocks = msg.render(config)
        texts = [b.text for b in blocks if hasattr(b, "text")]
        assert not any("Tokens:" in t for t in texts)

    def test_visible_when_shown(self):
        msg = parse_message(self._make_assistant())
        config = RenderConfig(filters=FilterConfig(shown={"token-usage"}))
        blocks = msg.render(config)
        texts = [b.text for b in blocks if hasattr(b, "text")]
        assert any("Tokens:" in t for t in texts)

    def test_show_token_usage_alone_works(self):
        """--show token-usage without --show metadata should still show tokens."""
        msg = parse_message(self._make_assistant())
        config = RenderConfig(filters=FilterConfig(shown={"token-usage"}))
        blocks = msg.render(config)
        texts = [b.text for b in blocks if hasattr(b, "text")]
        assert any("Tokens:" in t for t in texts)
        # metadata should still be hidden
        assert not any("-- Metadata" in t for t in texts)

    def test_hide_token_usage_hides_tokens(self):
        msg = parse_message(self._make_assistant())
        config = RenderConfig(
            filters=FilterConfig(shown={"token-usage"}, hidden={"token-usage"})
        )
        # shown overrides hidden, so tokens should appear
        blocks = msg.render(config)
        texts = [b.text for b in blocks if hasattr(b, "text")]
        assert any("Tokens:" in t for t in texts)


# =============================================================================
# 4. Render-level tests (ResultMessage)
# =============================================================================


class TestResultTokenUsageRender:
    def _make_result(self):
        return {
            "type": "result",
            "uuid": "res-001",
            "sessionId": "tok-session",
            "timestamp": "2026-03-17T10:05:00.000Z",
            "subtype": "success",
            "total_cost_usd": 0.05,
            "duration_ms": 300000,
            "num_turns": 5,
            "usage": {
                "input_tokens": 3000,
                "output_tokens": 1000,
                "cache_read_input_tokens": 500,
            },
        }

    def test_hidden_by_default(self):
        msg = parse_message(self._make_result())
        config = RenderConfig(filters=FilterConfig())
        blocks = msg.render(config)
        texts = [getattr(b, "key", "") for b in blocks]
        assert "Tokens" not in texts

    def test_visible_when_shown(self):
        msg = parse_message(self._make_result())
        config = RenderConfig(filters=FilterConfig(shown={"token-usage"}))
        blocks = msg.render(config)
        texts = [getattr(b, "key", "") for b in blocks]
        assert "Tokens" in texts


# =============================================================================
# 5. CLI integration
# =============================================================================


class TestTokenUsageCLI:
    def test_default_no_tokens(self, capsys):
        """Tokens line absent by default."""
        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "Tokens:" not in out

    def test_show_token_usage(self, capsys):
        """--show token-usage renders token lines."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show",
                "token-usage",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "Tokens:" in out

    def test_show_metadata_no_tokens(self, capsys):
        """--show metadata does not leak token lines."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show",
                "metadata",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "-- Metadata" in out
        assert "Tokens:" not in out

    def test_list_filters_includes_token_usage(self, capsys):
        """--list-filters mentions token-usage."""
        code, out = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        assert "token-usage" in out
