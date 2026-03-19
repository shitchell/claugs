"""Comprehensive integration tests for the unified --show/--hide/--show-only filter system.

Tests exercise the CLI filter flags through main() using JSONL fixtures and
custom messages, verifying that the FilterConfig resolution priority is correctly
applied end-to-end through the CLI.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_logs.cli import main, parse_args, _build_filters
from conftest import create_session_file


# =============================================================================
# Helper
# =============================================================================

FIXTURE = Path(__file__).parent / "fixtures" / "v2.1.77" / "complete_session.jsonl"


def _run(argv: list[str], capsys) -> tuple[int, str]:
    """Invoke main() with a given argv, return (exit_code, stdout)."""
    with patch.object(sys, "argv", argv):
        code = main()
    return code, capsys.readouterr().out


# =============================================================================
# 1. TestShowHideFlags
# =============================================================================


class TestShowHideFlags:
    """Test --hide and --show flags for individual filter names."""

    def test_hide_thinking(self, capsys):
        """--hide thinking removes thinking blocks from assistant output."""
        code, out = _run(["claugs", "show", "--hide", "thinking", str(FIXTURE)], capsys)
        assert code == 0
        # The fixture's first assistant has a thinking block
        assert "Thinking:" not in out
        # But the assistant's text content should still appear
        assert "I'll check the directory contents" in out

    def test_thinking_visible_by_default(self, capsys):
        """Thinking blocks appear without --hide thinking."""
        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "Thinking:" in out

    def test_hide_metadata(self, capsys):
        """--hide metadata keeps metadata absent (it is hidden by default)."""
        code, out = _run(["claugs", "show", "--hide", "metadata", str(FIXTURE)], capsys)
        assert code == 0
        assert "-- Metadata" not in out

    def test_show_metadata(self, capsys):
        """--show metadata reveals the metadata block."""
        code, out = _run(["claugs", "show", "--show", "metadata", str(FIXTURE)], capsys)
        assert code == 0
        assert "-- Metadata" in out

    def test_hide_timestamps(self, capsys):
        """--hide timestamps removes the · separator from headers."""
        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "\u00b7" not in out

    def test_show_timestamps(self, capsys):
        """--show timestamps adds the · separator to headers."""
        code, out = _run(
            ["claugs", "show", "--show", "timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "\u00b7" in out

    def test_hide_tools(self, capsys):
        """--hide tools removes tool-use blocks and tool-result messages."""
        code, out = _run(
            ["claugs", "show", "--hide", "tools", "--hide", "timestamps", str(FIXTURE)],
            capsys,
        )
        assert code == 0
        # The fixture has a Bash tool call; its header should not appear
        assert "Tool: Bash" not in out
        # The assistant text content should still appear
        assert "I'll check the directory contents" in out

    def test_tools_visible_by_default(self, capsys):
        """Tool blocks appear by default."""
        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "Tool: Bash" in out

    def test_hide_message_type_system(self, capsys):
        """--hide system suppresses system init messages."""
        code, out = _run(
            ["claugs", "show", "--hide", "system", "--hide", "timestamps", str(FIXTURE)],
            capsys,
        )
        assert code == 0
        assert "SYSTEM (init)" not in out

    def test_hide_message_type_result(self, capsys):
        """--hide result suppresses the SESSION COMPLETE block."""
        code, out = _run(
            ["claugs", "show", "--hide", "result", "--hide", "timestamps", str(FIXTURE)],
            capsys,
        )
        assert code == 0
        assert "SESSION COMPLETE" not in out

    def test_hide_message_type_user(self, capsys):
        """--hide user suppresses all user messages."""
        code, out = _run(
            ["claugs", "show", "--hide", "user", "--hide", "timestamps", str(FIXTURE)],
            capsys,
        )
        assert code == 0
        assert "Hello, what files are in this directory?" not in out
        # Assistant messages should still appear
        assert "I'll check the directory contents" in out

    def test_hide_message_type_assistant(self, capsys):
        """--hide assistant suppresses all assistant messages."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--hide",
                "assistant",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "ASSISTANT" not in out
        # User messages should still appear
        assert "Hello, what files are in this directory?" in out

    def test_comma_separated_hide(self, capsys):
        """--hide thinking,tools hides both in one flag invocation."""
        code, out = _run(
            ["claugs", "show", "--hide", "thinking,tools", "--hide", "timestamps", str(FIXTURE)],
            capsys,
        )
        assert code == 0
        assert "Thinking:" not in out
        assert "Tool: Bash" not in out
        # Regular text still present
        assert "I'll check the directory contents" in out

    def test_comma_separated_show(self, capsys):
        """--show metadata,timestamps shows both in one flag invocation."""
        code, out = _run(
            ["claugs", "show", "--show", "metadata,timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "-- Metadata" in out
        assert "\u00b7" in out

    def test_repeated_hide_flags(self, capsys):
        """Multiple --hide flags are cumulative."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--hide",
                "thinking",
                "--hide",
                "tools",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "Thinking:" not in out
        assert "Tool: Bash" not in out


# =============================================================================
# 2. TestShowOnlyFlag
# =============================================================================


class TestShowOnlyFlag:
    """Test --show-only whitelist behaviour."""

    def test_show_only_assistant(self, capsys):
        """--show-only assistant hides user, system, and result messages."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "assistant",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "ASSISTANT" in out
        assert "Hello, what files are in this directory?" not in out
        assert "SYSTEM (init)" not in out
        assert "SESSION COMPLETE" not in out

    def test_show_only_user(self, capsys):
        """--show-only user hides assistant and system messages."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "user",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "Hello, what files are in this directory?" in out
        # Assistant text content should not appear
        assert "I'll check the directory contents" not in out
        assert "SYSTEM (init)" not in out

    def test_show_only_with_show_override(self, capsys):
        """--show-only assistant --show system allows system through as well."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "assistant",
                "--show",
                "system",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "ASSISTANT" in out
        assert "SYSTEM (init)" in out
        # User messages should still be hidden
        assert "Hello, what files are in this directory?" not in out

    def test_show_only_with_hide(self, capsys):
        """--show-only assistant,user with --hide thinking drops thinking even though assistant is shown."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "assistant,user",
                "--hide",
                "thinking",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "ASSISTANT" in out
        assert "Hello, what files are in this directory?" in out
        assert "Thinking:" not in out

    def test_show_only_comma_separated(self, capsys):
        """--show-only user,result shows exactly those two types."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "user,result",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "Hello, what files are in this directory?" in out
        assert "SESSION COMPLETE" in out
        assert "ASSISTANT" not in out
        assert "SYSTEM (init)" not in out

    def test_show_only_repeated(self, capsys):
        """--show-only assistant --show-only result accumulates both."""
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "assistant",
                "--show-only",
                "result",
                "--hide",
                "timestamps",
                str(FIXTURE),
            ],
            capsys,
        )
        assert code == 0
        assert "ASSISTANT" in out
        assert "SESSION COMPLETE" in out
        assert "SYSTEM (init)" not in out


# =============================================================================
# 3. TestCompactFlag
# =============================================================================


class TestCompactFlag:
    """Test --compact preset and interactions with other flags."""

    def test_compact_hides_thinking(self, capsys):
        """--compact removes thinking blocks."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "Thinking:" not in out

    def test_compact_hides_tools(self, capsys):
        """--compact removes tool-use output."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "Tool: Bash" not in out

    def test_compact_hides_timestamps(self, capsys):
        """--compact removes timestamp suffix from headers."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "\u00b7" not in out

    def test_compact_hides_metadata(self, capsys):
        """--compact keeps metadata hidden (already default, stays hidden)."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "-- Metadata" not in out

    def test_compact_hides_system(self, capsys):
        """--compact suppresses system init messages."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "SYSTEM (init)" not in out

    def test_compact_hides_result(self, capsys):
        """--compact suppresses the session result block."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "SESSION COMPLETE" not in out

    def test_compact_preserves_user_text(self, capsys):
        """--compact still shows user message text content."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "Hello, what files are in this directory?" in out

    def test_compact_preserves_assistant_text(self, capsys):
        """--compact still shows assistant text content."""
        code, out = _run(["claugs", "show", "--compact", str(FIXTURE)], capsys)
        assert code == 0
        assert "I'll check the directory contents" in out

    def test_compact_with_show_override_thinking(self, capsys):
        """--compact --show thinking brings thinking back."""
        code, out = _run(
            ["claugs", "show", "--compact", "--show", "thinking", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "Thinking:" in out

    def test_compact_with_show_override_timestamps(self, capsys):
        """--compact --show timestamps restores timestamp display."""
        code, out = _run(
            ["claugs", "show", "--compact", "--show", "timestamps", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "\u00b7" in out

    def test_compact_with_show_override_tools(self, capsys):
        """--compact --show tools restores tool-use output."""
        code, out = _run(
            ["claugs", "show", "--compact", "--show", "tools", str(FIXTURE)], capsys
        )
        assert code == 0
        assert "Tool: Bash" in out

    def test_compact_args_parsed_correctly(self):
        """parse_args() sets compact=True and _build_filters reflects it."""
        with patch.object(sys, "argv", ["claugs", "show", "--compact", "--latest"]):
            _, args = parse_args()
        assert args.compact is True
        filters = _build_filters(args)
        assert filters.is_visible("thinking") is False
        assert filters.is_visible("tools") is False
        assert filters.is_visible("timestamps") is False
        assert filters.is_visible("metadata") is False


# =============================================================================
# 4. TestListFilters
# =============================================================================


class TestListFilters:
    """Test --list-filters output."""

    def test_list_filters_exits_zero(self, capsys):
        """--list-filters returns exit code 0."""
        code, _ = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0

    def test_list_filters_contains_message_types(self, capsys):
        """--list-filters output mentions core message type filter names."""
        code, out = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        for name in ("assistant", "user", "system", "result", "summary"):
            assert name in out, f"expected {name!r} in --list-filters output"

    def test_list_filters_contains_content_filters(self, capsys):
        """--list-filters output mentions content-level filter names."""
        code, out = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        for name in ("thinking", "tools", "metadata", "timestamps", "line-numbers"):
            assert name in out, f"expected {name!r} in --list-filters output"

    def test_list_filters_contains_user_subtypes(self, capsys):
        """--list-filters output mentions user message subtype names."""
        code, out = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        for name in ("user-input", "tool-result", "system-meta"):
            assert name in out, f"expected {name!r} in --list-filters output"

    def test_list_filters_contains_system_subtypes(self, capsys):
        """--list-filters output mentions system message subtype names."""
        code, out = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        for name in ("init", "compact-boundary"):
            assert name in out, f"expected {name!r} in --list-filters output"

    def test_list_filters_mentions_defaults(self, capsys):
        """--list-filters output mentions what is hidden by default."""
        code, out = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        # The help text calls out the defaults
        assert "metadata" in out
        assert "line-numbers" in out


# =============================================================================
# 5. TestSubtypeNormalization
# =============================================================================


class TestSubtypeNormalization:
    """Test that underscore subtype names are normalised to kebab-case."""

    def test_compact_boundary_hide_kebab(self, tmp_path, capsys):
        """--hide compact-boundary hides compact_boundary system messages (kebab input)."""
        compact_boundary_msg = {
            "type": "system",
            "subtype": "compact_boundary",
            "sessionId": "norm-test-001",
            "timestamp": "2026-03-17T10:00:00.000Z",
            "content": "Compacted",
            "compactMetadata": {"preTokens": 5000, "postTokens": 1200},
        }
        regular_system_msg = {
            "type": "system",
            "subtype": "init",
            "sessionId": "norm-test-001",
            "timestamp": "2026-03-17T10:00:01.000Z",
            "model": "claude-sonnet-4-5",
            "claude_code_version": "2.1.77",
            "cwd": "/tmp",
        }
        session_file = create_session_file(
            tmp_path, "norm-session", [compact_boundary_msg, regular_system_msg]
        )

        code, out = _run(
            [
                "claugs",
                "show",
                "--hide",
                "compact-boundary",
                "--hide",
                "timestamps",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        # The compact-boundary message should be gone
        assert "Compacted" not in out
        # The init message should still be present
        assert "SYSTEM (init)" in out

    def test_compact_boundary_visible_by_default(self, tmp_path, capsys):
        """compact_boundary system messages appear without any --hide flag."""
        compact_boundary_msg = {
            "type": "system",
            "subtype": "compact_boundary",
            "sessionId": "norm-test-002",
            "timestamp": "2026-03-17T10:00:00.000Z",
            "content": "Context compacted",
            "compactMetadata": {"preTokens": 3000, "postTokens": 800},
        }
        session_file = create_session_file(
            tmp_path, "norm-session-2", [compact_boundary_msg]
        )

        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(session_file)], capsys
        )
        assert code == 0
        assert "SYSTEM (compact_boundary)" in out

    def test_show_only_compact_boundary_shows_only_that(self, tmp_path, capsys):
        """--show-only system,compact-boundary shows only compact_boundary system messages.

        The type name ('system') must be included in show_only so the message passes
        the type-level visibility check, then the subtype name ('compact-boundary')
        restricts which system messages are allowed through.
        """
        compact_boundary_msg = {
            "type": "system",
            "subtype": "compact_boundary",
            "sessionId": "norm-test-003",
            "timestamp": "2026-03-17T10:00:00.000Z",
            "content": "Compacted here",
            "compactMetadata": {"preTokens": 2000, "postTokens": 500},
        }
        init_msg = {
            "type": "system",
            "subtype": "init",
            "sessionId": "norm-test-003",
            "timestamp": "2026-03-17T10:00:01.000Z",
            "model": "claude-sonnet-4-5",
            "claude_code_version": "2.1.77",
            "cwd": "/tmp",
        }
        user_msg = {
            "type": "user",
            "uuid": "u-001",
            "sessionId": "norm-test-003",
            "timestamp": "2026-03-17T10:00:02.000Z",
            "userType": "external",
            "message": {"role": "user", "content": "hello"},
        }
        session_file = create_session_file(
            tmp_path,
            "norm-session-3",
            [compact_boundary_msg, init_msg, user_msg],
        )

        # Must include both the type ('system') and the subtype ('compact-boundary').
        # The type passes the FilterConfig.is_visible() gate; the subtype name in
        # show_only then activates subtype-level whitelisting in should_show_message.
        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "system,compact-boundary",
                "--hide",
                "timestamps",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        # compact_boundary message should show
        assert "Compacted here" in out
        # init subtype is blocked by the subtype whitelist
        assert "SYSTEM (init)" not in out
        # user type is blocked by the type-level whitelist
        assert "hello" not in out


# =============================================================================
# 6. TestSystemMetaRename
# =============================================================================


class TestSystemMetaRename:
    """Test --hide system-meta / --show-only system-meta behaviour.

    system-meta refers to user messages that have isMeta=True, i.e., messages
    injected by the Claude system (skill loading, etc.), not regular user input.
    """

    def _make_meta_message(self, session_id: str, uuid: str, content: str) -> dict:
        return {
            "type": "user",
            "uuid": uuid,
            "sessionId": session_id,
            "timestamp": "2026-03-17T10:00:05.000Z",
            "userType": "external",
            "isMeta": True,
            "message": {"role": "user", "content": content},
        }

    def _make_regular_user_message(
        self, session_id: str, uuid: str, content: str
    ) -> dict:
        return {
            "type": "user",
            "uuid": uuid,
            "sessionId": session_id,
            "timestamp": "2026-03-17T10:00:10.000Z",
            "userType": "external",
            "message": {"role": "user", "content": content},
        }

    def test_hide_system_meta_hides_meta_user_messages(self, tmp_path, capsys):
        """--hide system-meta removes user messages with isMeta=True."""
        session_id = "sysmeta-test-001"
        meta_msg = self._make_meta_message(session_id, "m-001", "system skill context")
        regular_msg = self._make_regular_user_message(
            session_id, "u-001", "regular user question"
        )
        session_file = create_session_file(
            tmp_path, session_id, [meta_msg, regular_msg]
        )

        code, out = _run(
            [
                "claugs",
                "show",
                "--hide",
                "system-meta",
                "--hide",
                "timestamps",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "system skill context" not in out
        assert "regular user question" in out

    def test_regular_user_messages_not_hidden_by_system_meta(self, tmp_path, capsys):
        """--hide system-meta does NOT suppress regular (non-meta) user messages."""
        session_id = "sysmeta-test-002"
        regular_msg = self._make_regular_user_message(
            session_id, "u-002", "this should appear"
        )
        session_file = create_session_file(tmp_path, session_id, [regular_msg])

        code, out = _run(
            [
                "claugs",
                "show",
                "--hide",
                "system-meta",
                "--hide",
                "timestamps",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "this should appear" in out

    def test_meta_message_visible_by_default(self, tmp_path, capsys):
        """isMeta user messages are shown by default (no --hide system-meta)."""
        session_id = "sysmeta-test-003"
        meta_msg = self._make_meta_message(session_id, "m-002", "meta content visible")
        session_file = create_session_file(tmp_path, session_id, [meta_msg])

        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(session_file)], capsys
        )
        assert code == 0
        assert "meta content visible" in out

    def test_show_only_system_meta_hides_regular_user(self, tmp_path, capsys):
        """--show-only user,system-meta shows only system-meta user messages.

        Both the type ('user') and subtype ('system-meta') must appear in show_only:
        the type clears the FilterConfig.is_visible() gate, and the subtype name
        activates subtype-level whitelisting in should_show_message so that only
        isMeta=True messages pass through.
        """
        session_id = "sysmeta-test-004"
        meta_msg = self._make_meta_message(session_id, "m-003", "meta only content")
        regular_msg = self._make_regular_user_message(
            session_id, "u-003", "regular hidden content"
        )
        session_file = create_session_file(
            tmp_path, session_id, [meta_msg, regular_msg]
        )

        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "user,system-meta",
                "--hide",
                "timestamps",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "meta only content" in out
        assert "regular hidden content" not in out

    def test_show_only_user_input_hides_meta(self, tmp_path, capsys):
        """--show-only user,user-input shows only regular user input (not isMeta).

        Both the type ('user') and subtype ('user-input') must appear in show_only.
        The type name clears the FilterConfig.is_visible() gate; the subtype name
        activates subtype-level whitelisting so that isMeta=True messages (which have
        subtype 'system-meta') are excluded.
        """
        session_id = "sysmeta-test-005"
        meta_msg = self._make_meta_message(session_id, "m-004", "hidden meta")
        regular_msg = self._make_regular_user_message(
            session_id, "u-004", "visible regular"
        )
        session_file = create_session_file(
            tmp_path, session_id, [meta_msg, regular_msg]
        )

        code, out = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "user,user-input",
                "--hide",
                "timestamps",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "visible regular" in out
        assert "hidden meta" not in out

    def test_system_meta_label_in_header(self, tmp_path, capsys):
        """isMeta messages render with USER [meta] label by default."""
        session_id = "sysmeta-test-006"
        meta_msg = self._make_meta_message(session_id, "m-005", "label check")
        session_file = create_session_file(tmp_path, session_id, [meta_msg])

        code, out = _run(
            ["claugs", "show", "--hide", "timestamps", str(session_file)], capsys
        )
        assert code == 0
        assert "USER [meta]" in out
