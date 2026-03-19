# Unified `--show`/`--hide` Filter System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered visibility flags with a unified `--show`/`--hide`/`--show-only` filter system using a single namespace of named filters.

**Architecture:** New `FilterConfig` dataclass replaces all visibility booleans on `RenderConfig`. A single `is_visible(name)` method resolves visibility through a priority chain: show-only → show → hide → defaults. All message filtering in `should_show_message()` and all render-level visibility checks use this method. CLI flags collapse into `--show-only`, `--show`, `--hide`, `--compact`, and `--list-filters`.

**Tech Stack:** Python 3.10+, Pydantic 2.0+, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-unified-filters-design.md`

---

### Task 1: FilterConfig Dataclass

**Files:**
- Modify: `src/claude_logs/models.py`
- Create: `tests/test_filter_config.py`

- [ ] **Step 1: Write failing tests for FilterConfig**

```python
"""Tests for FilterConfig visibility resolution."""

import pytest
from claude_logs.models import FilterConfig


class TestFilterConfigDefaults:
    def test_shown_by_default(self):
        fc = FilterConfig()
        assert fc.is_visible("assistant") is True
        assert fc.is_visible("user") is True
        assert fc.is_visible("thinking") is True

    def test_default_hidden(self):
        fc = FilterConfig()
        assert fc.is_visible("metadata") is False
        assert fc.is_visible("line-numbers") is False
        assert fc.is_visible("file-history-snapshot") is False


class TestFilterConfigHide:
    def test_hide_makes_invisible(self):
        fc = FilterConfig(hidden={"thinking"})
        assert fc.is_visible("thinking") is False

    def test_hide_does_not_affect_others(self):
        fc = FilterConfig(hidden={"thinking"})
        assert fc.is_visible("assistant") is True


class TestFilterConfigShow:
    def test_show_overrides_hide(self):
        fc = FilterConfig(shown={"thinking"}, hidden={"thinking"})
        assert fc.is_visible("thinking") is True

    def test_show_overrides_default_hidden(self):
        fc = FilterConfig(shown={"metadata"})
        assert fc.is_visible("metadata") is True


class TestFilterConfigShowOnly:
    def test_show_only_hides_unlisted(self):
        fc = FilterConfig(show_only={"assistant", "user"})
        assert fc.is_visible("assistant") is True
        assert fc.is_visible("user") is True
        assert fc.is_visible("system") is False
        assert fc.is_visible("thinking") is False

    def test_show_only_plus_show(self):
        fc = FilterConfig(show_only={"assistant"}, shown={"metadata"})
        assert fc.is_visible("assistant") is True
        assert fc.is_visible("metadata") is True
        assert fc.is_visible("user") is False

    def test_show_only_plus_hide(self):
        fc = FilterConfig(show_only={"assistant", "thinking"}, hidden={"thinking"})
        assert fc.is_visible("assistant") is True
        assert fc.is_visible("thinking") is False

    def test_show_overrides_show_only_hide(self):
        """--show always wins, even if item not in show-only."""
        fc = FilterConfig(show_only={"assistant"}, shown={"metadata"}, hidden={"metadata"})
        assert fc.is_visible("metadata") is True


class TestFilterConfigPriorityChain:
    def test_full_chain(self):
        """show-only=assistant,user → hide=timestamps → show=metadata."""
        fc = FilterConfig(
            show_only={"assistant", "user"},
            shown={"metadata"},
            hidden={"timestamps"},
        )
        assert fc.is_visible("assistant") is True
        assert fc.is_visible("user") is True
        assert fc.is_visible("system") is False  # not in show-only
        assert fc.is_visible("metadata") is True  # explicit show
        assert fc.is_visible("timestamps") is False  # explicit hide
        assert fc.is_visible("thinking") is False  # not in show-only

    def test_empty_show_only_means_no_whitelist(self):
        fc = FilterConfig(show_only=set())
        assert fc.is_visible("assistant") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/guy/git/github.com/shitchell/claude-stream && pytest tests/test_filter_config.py -v`
Expected: ImportError — `FilterConfig` does not exist.

- [ ] **Step 3: Implement FilterConfig**

In `src/claude_logs/models.py`, add just before `GroupByConfig` (around line 144). Import `ClassVar` from `typing`:

```python
from typing import Annotated, Any, ClassVar, Literal, Union

@dataclass
class FilterConfig:
    """Unified visibility configuration.

    Resolution priority:
    1. show (explicit --show) always wins
    2. hidden (explicit --hide) overrides defaults and show-only
    3. show_only (whitelist base) hides everything not listed
    4. DEFAULT_HIDDEN for items hidden by default
    """

    show_only: set[str] = field(default_factory=set)
    shown: set[str] = field(default_factory=set)
    hidden: set[str] = field(default_factory=set)

    DEFAULT_HIDDEN: ClassVar[set[str]] = {"metadata", "line-numbers", "file-history-snapshot"}

    def is_visible(self, name: str) -> bool:
        """Check if a filter name is visible."""
        if name in self.shown:
            return True
        if name in self.hidden:
            return False
        if self.show_only and name not in self.show_only:
            return False
        return name not in self.DEFAULT_HIDDEN
```

- [ ] **Step 4: Run tests**

Run: `cd /home/guy/git/github.com/shitchell/claude-stream && pytest tests/test_filter_config.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/claude_logs/models.py tests/test_filter_config.py
git commit -m "feat: add FilterConfig with unified visibility resolution"
```

---

### Task 2: Replace RenderConfig Booleans with FilterConfig

**Files:**
- Modify: `src/claude_logs/models.py`
- Modify: `src/claude_logs/stream.py`
- Modify: `src/claude_logs/models.py` (render methods)

This is the core migration. Replace all scattered visibility booleans on `RenderConfig` with `filters: FilterConfig`.

- [ ] **Step 1: Update RenderConfig**

Replace the current `RenderConfig` with:

```python
@dataclass
class RenderConfig:
    """Configuration for rendering messages."""

    filters: FilterConfig = field(default_factory=FilterConfig)
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"

    # Timestamp filtering (not part of the visibility system)
    before: _datetime | None = None
    after: _datetime | None = None

    # Text filtering
    grep_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    # Grouping
    group_by: GroupByConfig | None = None
```

- [ ] **Step 2: Update all render methods in models.py**

Replace every `config.show_thinking`, `config.show_tool_results`, `config.show_metadata`, `config.show_timestamps`, `config.show_line_numbers` with `config.filters.is_visible("name")`:

- `ThinkingContent.render()`: `config.show_thinking` → `config.filters.is_visible("thinking")`
- `ToolUseContent.render()`: Add `if not config.filters.is_visible("tools"): return []` as the **first line** of the method, BEFORE the header block. Also remove the old `config.show_tool_results` check on the inputs section. This hides the entire block (header + inputs) when tools are hidden.
- `ToolResultContent.render()`: Add `if not config.filters.is_visible("tools"): return []` as the **first line** of the method, BEFORE the header block. Also remove the old `config.show_tool_results` check on the content section. This hides the entire block (header + content) when tools are hidden.
- `BaseMessage.format_timestamp_suffix()`: `config.show_timestamps` → `config.filters.is_visible("timestamps")`
- `BaseMessage.render_metadata()`: `config.show_metadata` → `config.filters.is_visible("metadata")`
- `UserMessage.render_local_command()`: the `config.show_tool_results` check on local-command stdout → `config.filters.is_visible("tools")`

- [ ] **Step 3: Update process_stream in stream.py**

Replace `config.show_line_numbers` with `config.filters.is_visible("line-numbers")`.

- [ ] **Step 4: Rewrite should_show_message() in stream.py**

Replace the entire function body with FilterConfig-based logic:

```python
def should_show_message(
    msg: BaseMessage, data: dict[str, Any], config: RenderConfig
) -> bool:
    """Determine if a message should be displayed based on filters."""
    filters = config.filters

    # Check message type visibility
    if not filters.is_visible(msg.type):
        return False

    # Check subtype visibility (computed or JSON)
    if isinstance(msg, UserMessage):
        subtype = msg.get_subtype()
        if not filters.is_visible(subtype):
            return False
        # Synthetic "tools" filter hides tool-result/subagent-result messages
        if subtype in ("tool-result", "subagent-result") and not filters.is_visible("tools"):
            return False
    else:
        # Check JSON subtype if present (use parsed model, not raw data)
        raw_subtype = getattr(msg, "subtype", "")
        if raw_subtype:
            normalized = raw_subtype.replace("_", "-")
            if not filters.is_visible(normalized):
                return False

    # Check tool name visibility (for messages with tool_use content)
    content = data.get("message", {}).get("content", [])
    if isinstance(content, list):
        tool_names = {
            item.get("name")
            for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"
        }
        if tool_names:
            # If ANY tool in this message is explicitly hidden, hide the message
            for tool_name in tool_names:
                if not filters.is_visible(tool_name):
                    return False

    # Check timestamp filters
    if config.before or config.after:
        ts_str = data.get("timestamp", "")
        if ts_str:
            try:
                msg_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=timezone.utc)
                if config.after and msg_dt < config.after:
                    return False
                if config.before and msg_dt > config.before:
                    return False
            except (ValueError, OSError):
                pass

    # Check grep patterns (evaluated after visibility filters)
    if config.grep_patterns:
        msg_str = json.dumps(data)
        if not any(pattern in msg_str for pattern in config.grep_patterns):
            return False

    # Check exclude patterns
    if config.exclude_patterns:
        msg_str = json.dumps(data)
        if any(pattern in msg_str for pattern in config.exclude_patterns):
            return False

    return True
```

- [ ] **Step 5: Rename meta → system-meta in UserMessage.get_subtype()**

```python
def get_subtype(self) -> str:
    if self.is_subagent_result():
        return "subagent-result"
    elif self.is_tool_result():
        return "tool-result"
    elif self.is_meta():
        return "system-meta"
    elif self.is_local_command():
        return "local-command"
    else:
        return "user-input"
```

- [ ] **Step 6: Do NOT commit yet**

Tests will be broken until Task 3 (CLI) and Task 4 (test updates) are done. Continue to Task 3 immediately. All of Tasks 2-4 will be committed together as a single atomic change once tests pass.

**Important:** This entire implementation should be done on a feature branch (`feat/unified-filters`). Do not merge to main until all tasks through Task 5 are complete and all tests pass.

---

### Task 3: Update CLI Flags

**Files:**
- Modify: `src/claude_logs/cli.py`

- [ ] **Step 1: Replace old flags in parse_args()**

Replace the `display_parent` and `filter_parent` argument definitions. The new shared parents:

**display_parent** (retains only format-related flags):
```python
display_parent = argparse.ArgumentParser(add_help=False)
display_parent.add_argument(
    "--format", "-F", choices=["ansi", "markdown", "plain"], default=None,
    help="Output format (default: markdown if CLAUDECODE is set, ansi if TTY, plain if piped)",
)
display_parent.add_argument(
    "--timestamp-format", dest="timestamp_format", default=None,
    help="Timestamp format string (default: %%Y-%%m-%%d %%H:%%M:%%S)",
)
```

**filter_parent** (new unified filter flags):
```python
filter_parent = argparse.ArgumentParser(add_help=False)
filter_parent.add_argument(
    "--show-only", action="append", dest="show_only_filters", metavar="NAME[,NAME]",
    help="Show ONLY these, hide everything else (repeatable, comma-separated)",
)
filter_parent.add_argument(
    "--show", action="append", dest="show_filters", metavar="NAME[,NAME]",
    help="Ensure these are visible (repeatable, comma-separated)",
)
filter_parent.add_argument(
    "--hide", action="append", dest="hide_filters", metavar="NAME[,NAME]",
    help="Remove these from output (repeatable, comma-separated)",
)
filter_parent.add_argument(
    "--compact", action="store_true",
    help="Hide non-essential content (thinking, tools, metadata, timestamps, system messages)",
)
filter_parent.add_argument(
    "--after", "--since", dest="after", metavar="DATETIME",
    help="Only show messages after this time",
)
filter_parent.add_argument(
    "--before", "--until", dest="before", metavar="DATETIME",
    help="Only show messages before this time",
)
filter_parent.add_argument(
    "--grep", action="append", dest="grep_patterns",
    help="Include only messages matching pattern (repeatable)",
)
filter_parent.add_argument(
    "--exclude", action="append", dest="exclude_patterns",
    help="Exclude messages matching pattern (repeatable)",
)
filter_parent.add_argument(
    "-n", "--lines", type=int, default=0, metavar="N",
    help="Show only last N lines (works with files and watch)",
)
```

Add `--list-filters` to the show subparser only:
```python
show_parser.add_argument(
    "--list-filters", action="store_true",
    help="Show available filter names and exit",
)
```

- [ ] **Step 2: Rewrite _build_config()**

```python
def _build_config(args: argparse.Namespace) -> RenderConfig:
    """Build a RenderConfig from parsed arguments."""
    filters = _build_filters(args)
    config = RenderConfig(filters=filters)

    if args.timestamp_format is not None:
        config.timestamp_format = args.timestamp_format

    if args.before:
        config.before = parse_datetime(args.before)
    if args.after:
        config.after = parse_datetime(args.after)

    if args.grep_patterns:
        config.grep_patterns = args.grep_patterns
    if args.exclude_patterns:
        config.exclude_patterns = args.exclude_patterns

    return config


def _build_filters(args: argparse.Namespace) -> FilterConfig:
    """Build FilterConfig from --show-only/--show/--hide/--compact flags."""
    from .models import FilterConfig

    show_only: set[str] = set()
    shown: set[str] = set()
    hidden: set[str] = set()

    # 1. --show-only (whitelist base)
    if getattr(args, "show_only_filters", None):
        for spec in args.show_only_filters:
            show_only.update(name.strip() for name in spec.split(","))

    # --compact (adds to hidden)
    if args.compact:
        hidden.update({
            "thinking", "tools", "metadata", "timestamps",
            "system", "summary", "queue-operation", "result",
        })

    # 2. --hide
    if getattr(args, "hide_filters", None):
        for spec in args.hide_filters:
            hidden.update(name.strip() for name in spec.split(","))

    # 3. --show (overrides hide)
    if getattr(args, "show_filters", None):
        for spec in args.show_filters:
            shown.update(name.strip() for name in spec.split(","))

    # Validate filter names (warn on unknowns, don't error)
    _KNOWN_FILTERS = {
        # Synthetic
        "thinking", "tools", "metadata", "timestamps", "line-numbers",
        # Computed subtypes
        "user-input", "tool-result", "subagent-result", "system-meta", "local-command",
        # JSON subtypes
        "init", "compact-boundary", "success",
        # Message types
        "system", "assistant", "user", "summary", "queue-operation", "result",
        "file-history-snapshot",
    }
    for name in (show_only | shown | hidden):
        if name not in _KNOWN_FILTERS:
            # Could be a dynamic tool name — warn but don't error
            print(f"warning: unknown filter: {name}", file=sys.stderr)

    return FilterConfig(show_only=show_only, shown=shown, hidden=hidden)
```

- [ ] **Step 3: Add --list-filters handler in handle_show()**

At the top of `handle_show()`, before any source resolution:

```python
if args.list_filters:
    _print_filter_list()
    return 0
```

Add the helper function:

```python
def _print_filter_list() -> None:
    """Print all available filter names and exit."""
    print("""Synthetic filters (content/display):
  thinking              Thinking/reasoning blocks (default: shown)
  tools                 Tool invocations and results (default: shown)
  metadata              Message metadata (default: hidden)
  timestamps            Timestamp display in headers (default: shown)
  line-numbers          Line number prefixes (default: hidden)

Subtypes:
  user-input            Human-typed messages (default: shown)
  tool-result           Tool output messages (default: shown)
  subagent-result       Sub-agent output messages (default: shown)
  system-meta           System-injected meta messages (default: shown)
  local-command         Local slash command messages (default: shown)
  init                  System initialization (default: shown)
  compact-boundary      Compaction boundary (default: shown)
  success               Result status (default: shown)

Message types:
  system                System messages (default: shown)
  assistant             Claude's responses (default: shown)
  user                  User messages (default: shown)
  summary               Summary messages (default: shown)
  queue-operation       Queue operation messages (default: shown)
  result                Session completion (default: shown)
  file-history-snapshot File state snapshots (default: hidden)

Tool names are discovered from input data. Use --show or --hide
with any tool name (e.g., Bash, Read, Edit).""")
```

- [ ] **Step 4: Update the epilog examples**

Update the examples in the main parser epilog to use the new flags:

```python
epilog="""
Examples:
    %(prog)s show session.jsonl                         # Render a session
    %(prog)s show --latest -n 50                        # Last 50 of most recent
    %(prog)s show --since "today" ~/myproject            # Today's messages
    %(prog)s show --search "error" -l                   # List matching filepaths
    %(prog)s show --hide thinking,tools                 # No thinking or tool blocks
    %(prog)s show --show-only assistant,user             # Only assistant and user
    %(prog)s show --compact --show thinking              # Compact but keep thinking
    %(prog)s show --list-filters                        # Show available filter names
    %(prog)s watch ~/.claude/projects/                  # Watch all sessions
    %(prog)s watch . -n 10                              # Watch with context
        """,
```

- [ ] **Step 5: Commit**

```bash
git add src/claude_logs/cli.py
git commit -m "feat: replace old CLI flags with --show/--hide/--show-only system"
```

---

### Task 4: Update All Tests

**Files:**
- Modify: all `tests/test_*.py` files

This is a mechanical update: replace old flag syntax with new syntax in every test file.

- [ ] **Step 1: Update test_compact_bug.py**

Old `args.show_thinking is None` assertions → check that compact populates `FilterConfig.hidden` correctly. The tests need to be rewritten to test `_build_filters()` instead of raw argparse attributes, since the old boolean flags no longer exist.

- [ ] **Step 2: Update test_cli_timestamps.py**

Replace `--show-timestamps`/`--hide-timestamps` with `--show timestamps`/`--hide timestamps`. Replace `args.show_timestamps` assertions with FilterConfig checks.

- [ ] **Step 3: Update test_header_suffix.py**

Replace `RenderConfig(show_timestamps=False)` with `RenderConfig(filters=FilterConfig(hidden={"timestamps"}))`.

- [ ] **Step 4: Update test_timestamp_display.py**

Replace `RenderConfig(show_timestamps=True)` with `RenderConfig()` (default is shown). Replace `RenderConfig(show_timestamps=False)` with `RenderConfig(filters=FilterConfig(hidden={"timestamps"}))`.

- [ ] **Step 5: Update test_timestamp_filter.py**

No visibility flag changes needed — timestamp filtering uses `config.before`/`config.after` which are unchanged. Just verify `RenderConfig(after=...)` still works.

- [ ] **Step 6: Update test_search.py**

Replace `--hide-timestamps` with `--hide timestamps` in argv.

- [ ] **Step 7: Update test_before_after.py**

Replace `--hide-timestamps` with `--hide timestamps` in argv.

- [ ] **Step 8: Update test_cli_group_by.py**

Replace `--hide-timestamps` with `--hide timestamps` in argv. Replace `--compact` test assertions.

- [ ] **Step 9: Update test_claugs_integration.py**

This is the biggest update. Replace all `--hide-timestamps`, `--show-thinking`, `--hide-thinking`, `--show-tool-results`, `--hide-tool-results`, `--show-metadata`, `--hide-metadata`, `--show-type`, `--show-subtype`, `--show-tool`, `--line-numbers` with new syntax. Also update any `RenderConfig(show_timestamps=False, ...)` constructors.

- [ ] **Step 10: Update test_grouping_render.py**

Replace `RenderConfig(show_timestamps=False)` with `RenderConfig(filters=FilterConfig(hidden={"timestamps"}))`.

- [ ] **Step 11: Run all tests**

Run: `cd /home/guy/git/github.com/shitchell/claude-stream && pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 12: Commit**

```bash
git add tests/
git commit -m "test: update all tests for unified --show/--hide filter system"
```

---

### Task 5: New Filter System Tests

**Files:**
- Create: `tests/test_unified_filters.py`

Write comprehensive tests specific to the new filter system's CLI integration.

- [ ] **Step 1: Write tests**

```python
"""Tests for the unified --show/--hide/--show-only filter system."""

import sys
from unittest.mock import patch

from claude_logs.cli import main, parse_args
from claude_logs.models import FilterConfig, RenderConfig
from conftest import create_session_file


class TestShowHideFlags:
    def test_hide_thinking(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--hide", "thinking"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "Thinking" not in out

    def test_show_metadata(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--show", "metadata"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "uuid" in out.lower() or "session" in out.lower()

    def test_hide_timestamps(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--hide", "timestamps"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "·" not in out

    def test_hide_tools(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--hide", "tools"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "Tool:" not in out
        assert "Result" not in out

    def test_hide_message_type(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--hide", "system", "--hide", "timestamps"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "SYSTEM" not in out

    def test_comma_separated(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--hide", "thinking,tools,timestamps"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "Thinking" not in out
        assert "Tool:" not in out


class TestShowOnlyFlag:
    def test_show_only_assistant(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--show-only", "assistant", "--hide", "timestamps"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "ASSISTANT" in out
        assert "USER" not in out
        assert "SYSTEM" not in out

    def test_show_only_plus_show(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", [
            "claugs", "show", fpath,
            "--show-only", "assistant",
            "--show", "metadata",
            "--hide", "timestamps",
        ]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "ASSISTANT" in out
        assert "uuid" in out.lower() or "session" in out.lower()


class TestCompactFlag:
    def test_compact_hides_expected(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--compact"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "ASSISTANT" in out
        assert "Thinking" not in out
        assert "Tool:" not in out
        assert "SYSTEM" not in out
        assert "SESSION COMPLETE" not in out
        assert "·" not in out  # no timestamps

    def test_compact_plus_show_override(self, fixtures_dir, capsys):
        fpath = str(fixtures_dir / "v2.1.77" / "complete_session.jsonl")
        with patch.object(sys, "argv", ["claugs", "show", fpath, "--compact", "--show", "thinking"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "Thinking" in out  # overridden back to visible


class TestListFilters:
    def test_list_filters_output(self, capsys):
        with patch.object(sys, "argv", ["claugs", "show", "--list-filters"]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "thinking" in out
        assert "tools" in out
        assert "metadata" in out
        assert "assistant" in out
        assert "user-input" in out
        assert "system-meta" in out
        assert "file-history-snapshot" in out


class TestSubtypeNormalization:
    def test_compact_boundary_normalized(self, tmp_path, capsys):
        """compact_boundary (underscore) is filterable as compact-boundary (hyphen)."""
        create_session_file(tmp_path, "s1", [
            {"type": "system", "uuid": "1", "timestamp": "2026-03-17T14:00:00Z",
             "subtype": "compact_boundary", "content": "Compacted", "compactMetadata": {"preTokens": 1000}},
            {"type": "user", "uuid": "2", "timestamp": "2026-03-17T14:01:00Z",
             "message": {"content": "user content"}},
        ])
        with patch.object(sys, "argv", [
            "claugs", "show", str(tmp_path / "s1.jsonl"),
            "--hide", "compact-boundary", "--hide", "timestamps",
        ]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "user content" in out
        assert "Compacted" not in out


class TestSystemMetaRename:
    def test_meta_renamed_to_system_meta(self, tmp_path, capsys):
        create_session_file(tmp_path, "s1", [
            {"type": "user", "uuid": "1", "timestamp": "2026-03-17T14:00:00Z",
             "isMeta": True, "message": {"content": "meta content"}},
            {"type": "user", "uuid": "2", "timestamp": "2026-03-17T14:01:00Z",
             "message": {"content": "user content"}},
        ])
        with patch.object(sys, "argv", [
            "claugs", "show", str(tmp_path / "s1.jsonl"),
            "--hide", "system-meta", "--hide", "timestamps",
        ]):
            code = main()
        assert code == 0
        out = capsys.readouterr().out
        assert "user content" in out
        assert "meta content" not in out
```

- [ ] **Step 2: Run tests**

Run: `cd /home/guy/git/github.com/shitchell/claude-stream && pytest tests/test_unified_filters.py -v`
Expected: All PASS.

- [ ] **Step 3: Run full suite**

Run: `cd /home/guy/git/github.com/shitchell/claude-stream && pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_unified_filters.py
git commit -m "test: add comprehensive tests for unified filter system"
```

---

### Task 6: Update Grouping + Exports + Version Bump

**Files:**
- Modify: `src/claude_logs/grouping.py` (if any references to old fields)
- Modify: `src/claude_logs/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Check and update grouping.py**

Search for any references to `config.show_types`, `config.show_tools`, `config.show_subtypes`, `config.show_thinking`, etc. in grouping.py. Replace with `config.filters.is_visible()` calls if found.

- [ ] **Step 2: Export FilterConfig from __init__.py**

Add `FilterConfig` to imports and `__all__`.

- [ ] **Step 3: Bump version to 0.6.0**

In pyproject.toml: `version = "0.6.0"`.

- [ ] **Step 4: Run full suite**

Run: `cd /home/guy/git/github.com/shitchell/claude-stream && pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/claude_logs/__init__.py src/claude_logs/grouping.py pyproject.toml
git commit -m "chore: export FilterConfig, bump version to 0.6.0"
```

---

### Task 7: Integration Smoke Test

- [ ] **Step 1: Test --hide**

Run: `claugs show --latest -n 3 --hide thinking,tools`
Expected: No thinking blocks or tool blocks.

- [ ] **Step 2: Test --show-only**

Run: `claugs show --latest -n 5 --show-only assistant`
Expected: Only assistant messages.

- [ ] **Step 3: Test --compact --show**

Run: `claugs show --latest -n 3 --compact --show thinking`
Expected: Compact output but with thinking blocks visible.

- [ ] **Step 4: Test --list-filters**

Run: `claugs show --list-filters`
Expected: Full filter list printed.

- [ ] **Step 5: Test --show metadata**

Run: `claugs show --latest -n 2 --show metadata`
Expected: Metadata blocks visible (uuid, session).
