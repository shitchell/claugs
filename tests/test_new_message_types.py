"""Tests for the session-state / metadata marker message types.

These types (mode, permission-mode, ai-title, bridge-session,
fork-context-ref, attachment) are written to the JSONL by the harness
alongside the conversation. Before they were modelled they triggered
"warning: unknown message type" on stderr. They are hidden by default but
must render a one-line summary when explicitly shown, and parse without
warnings.
"""

import sys
from pathlib import Path
from unittest.mock import patch

from claude_logs.cli import main
from claude_logs.models import (
    AttachmentMessage,
    parse_message,
)
from conftest import create_session_file


SESSION_ID = "new-types-001"


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    """Invoke main() with a given argv, return (exit_code, stdout, stderr)."""
    with patch.object(sys, "argv", argv):
        code = main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _msgs() -> list[dict]:
    """One message of each newly modelled state/metadata type."""
    return [
        {"type": "mode", "mode": "normal", "sessionId": SESSION_ID},
        {
            "type": "permission-mode",
            "permissionMode": "bypassPermissions",
            "sessionId": SESSION_ID,
        },
        {
            "type": "ai-title",
            "aiTitle": "Investigate the freezing issue",
            "sessionId": SESSION_ID,
        },
        {
            "type": "bridge-session",
            "sessionId": SESSION_ID,
            "bridgeSessionId": "cse_abc123",
            "lastSequenceNum": 0,
        },
        {
            "type": "fork-context-ref",
            "agentId": "ae8117e5b7cefdb47",
            "parentSessionId": "8215912b-parent",
            "parentLastUuid": "da9ee6d1-uuid",
            "contextLength": 196,
        },
        {
            "type": "attachment",
            "uuid": "att-001",
            "timestamp": "2026-06-10T00:00:08.448Z",
            "sessionId": SESSION_ID,
            "attachment": {
                "type": "hook_success",
                "hookName": "SessionStart:startup",
                "content": "Hook ran fine",
                "stdout": "Hook ran fine\n",
                "exitCode": 0,
            },
        },
    ]


# =============================================================================
# Parsing — no fallback to BaseMessage, no warnings
# =============================================================================


class TestParsing:
    def test_each_type_parses_to_its_own_class(self):
        """Every new type resolves to a concrete model, not bare BaseMessage."""
        for data in _msgs():
            msg = parse_message(data)
            assert type(msg).__name__ != "BaseMessage", data["type"]
            assert msg.type == data["type"]

    def test_no_unknown_type_warning(self, tmp_path, capsys):
        """Rendering the session emits no 'unknown message type' warnings."""
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        # --show all the (default-hidden) new types so render() actually runs
        code, _out, err = _run(
            [
                "claugs",
                "show",
                "--show",
                "mode,permission-mode,ai-title,bridge-session,"
                "fork-context-ref,attachment",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "unknown message type" not in err

    def test_attachment_subtype_kebab(self):
        """AttachmentMessage exposes the nested kind kebab-cased as .subtype."""
        msg = parse_message(_msgs()[-1])
        assert isinstance(msg, AttachmentMessage)
        assert msg.subtype == "hook-success"


# =============================================================================
# Default visibility — hidden unless explicitly shown
# =============================================================================


class TestDefaultHidden:
    def test_hidden_by_default(self, tmp_path, capsys):
        """None of the new state markers render without an explicit --show."""
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--hide", "timestamps", str(session_file)], capsys
        )
        assert code == 0
        for needle in (
            "Mode: normal",
            "Permission Mode:",
            "AI Title:",
            "Bridge Session:",
            "Fork Context",
            "Attachment (",
        ):
            assert needle not in out, needle


# =============================================================================
# Rendering when shown
# =============================================================================


class TestRenderWhenShown:
    def test_mode_renders(self, tmp_path, capsys):
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--show", "mode", str(session_file)], capsys
        )
        assert code == 0
        assert "Mode: normal" in out

    def test_permission_mode_renders(self, tmp_path, capsys):
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--show", "permission-mode", str(session_file)], capsys
        )
        assert code == 0
        assert "Permission Mode: bypassPermissions" in out

    def test_ai_title_renders(self, tmp_path, capsys):
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--show", "ai-title", str(session_file)], capsys
        )
        assert code == 0
        assert "AI Title: Investigate the freezing issue" in out

    def test_bridge_session_renders(self, tmp_path, capsys):
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--show", "bridge-session", str(session_file)], capsys
        )
        assert code == 0
        assert "Bridge Session: cse_abc123" in out

    def test_fork_context_ref_renders(self, tmp_path, capsys):
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--show", "fork-context-ref", str(session_file)], capsys
        )
        assert code == 0
        assert "Fork Context" in out
        assert "ae8117e5b7cefdb47" in out

    def test_attachment_renders_content(self, tmp_path, capsys):
        session_file = create_session_file(tmp_path, SESSION_ID, _msgs())
        code, out, _err = _run(
            ["claugs", "show", "--show", "attachment", str(session_file)], capsys
        )
        assert code == 0
        assert "Attachment (hook_success)" in out
        assert "Hook ran fine" in out


# =============================================================================
# Attachment subtype filtering via the shared subtype machinery
# =============================================================================


class TestAttachmentSubtypeFiltering:
    def _two_attachments(self) -> list[dict]:
        return [
            {
                "type": "attachment",
                "uuid": "a1",
                "sessionId": SESSION_ID,
                "attachment": {"type": "hook_success", "content": "HOOK OUTPUT"},
            },
            {
                "type": "attachment",
                "uuid": "a2",
                "sessionId": SESSION_ID,
                "attachment": {"type": "task_reminder", "content": "TODO REMINDER"},
            },
        ]

    def test_show_only_attachment_subtype(self, tmp_path, capsys):
        """--show-only attachment,hook-success shows only hook_success attachments."""
        session_file = create_session_file(
            tmp_path, SESSION_ID, self._two_attachments()
        )
        code, out, _err = _run(
            [
                "claugs",
                "show",
                "--show-only",
                "attachment,hook-success",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "HOOK OUTPUT" in out
        assert "TODO REMINDER" not in out

    def test_hide_attachment_subtype(self, tmp_path, capsys):
        """--show attachment --hide task-reminder drops only the reminder."""
        session_file = create_session_file(
            tmp_path, SESSION_ID, self._two_attachments()
        )
        code, out, _err = _run(
            [
                "claugs",
                "show",
                "--show",
                "attachment",
                "--hide",
                "task-reminder",
                str(session_file),
            ],
            capsys,
        )
        assert code == 0
        assert "HOOK OUTPUT" in out
        assert "TODO REMINDER" not in out


# =============================================================================
# --list-filters surfaces the new names
# =============================================================================


class TestListFilters:
    def test_new_type_names_listed(self, capsys):
        code, out, _err = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        for name in (
            "mode",
            "permission-mode",
            "ai-title",
            "bridge-session",
            "fork-context-ref",
            "attachment",
        ):
            assert name in out, name

    def test_attachment_subtypes_listed(self, capsys):
        code, out, _err = _run(["claugs", "show", "--list-filters"], capsys)
        assert code == 0
        for name in ("task-reminder", "hook-success", "edited-text-file"):
            assert name in out, name
