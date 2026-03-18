"""Tests for timestamp display in message rendering."""

from claude_stream.blocks import HeaderBlock, Style
from claude_stream.models import (
    AssistantMessage,
    QueueOperationMessage,
    RenderConfig,
    ResultMessage,
    SummaryMessage,
    SystemMessage,
    UserMessage,
    parse_message,
)


def _find_header(blocks) -> HeaderBlock | None:
    """Find the first HeaderBlock in a list of render blocks."""
    for b in blocks:
        if isinstance(b, HeaderBlock):
            return b
    return None


class TestTimestampInHeaders:
    def test_assistant_message_has_timestamp_suffix(self, sample_assistant_message):
        config = RenderConfig(show_timestamps=True)
        msg = parse_message(sample_assistant_message)
        blocks = msg.render(config)
        header = _find_header(blocks)
        assert header is not None
        assert "· 2026-03-17 14:23:05" in header.suffix

    def test_user_message_has_timestamp_suffix(self, sample_user_message):
        config = RenderConfig(show_timestamps=True)
        msg = parse_message(sample_user_message)
        blocks = msg.render(config)
        header = _find_header(blocks)
        assert header is not None
        assert "· 2026-03-17 14:23:12" in header.suffix

    def test_system_message_has_timestamp_suffix(self, sample_system_init_message):
        config = RenderConfig(show_timestamps=True)
        msg = parse_message(sample_system_init_message)
        blocks = msg.render(config)
        header = _find_header(blocks)
        assert header is not None
        assert "· 2026-03-17 14:22:58" in header.suffix

    def test_result_message_has_timestamp_suffix(self, sample_result_message):
        config = RenderConfig(show_timestamps=True)
        msg = parse_message(sample_result_message)
        blocks = msg.render(config)
        headers = [b for b in blocks if isinstance(b, HeaderBlock)]
        assert any("· 2026-03-17 14:30:00" in h.suffix for h in headers)


class TestTimestampHidden:
    def test_no_suffix_when_hidden(self, sample_assistant_message):
        config = RenderConfig(show_timestamps=False)
        msg = parse_message(sample_assistant_message)
        blocks = msg.render(config)
        header = _find_header(blocks)
        assert header is not None
        assert header.suffix == ""


class TestTimestampFormat:
    def test_custom_format(self, sample_assistant_message):
        config = RenderConfig(show_timestamps=True, timestamp_format="%H:%M")
        msg = parse_message(sample_assistant_message)
        blocks = msg.render(config)
        header = _find_header(blocks)
        assert header is not None
        assert "· 14:23" in header.suffix


class TestTimestampMissing:
    def test_no_timestamp_no_suffix(self, sample_message_no_timestamp):
        config = RenderConfig(show_timestamps=True)
        msg = parse_message(sample_message_no_timestamp)
        blocks = msg.render(config)
        header = _find_header(blocks)
        assert header is not None
        assert header.suffix == ""


class TestToolResultNoTimestamp:
    def test_tool_result_no_header(self, sample_tool_result_message):
        config = RenderConfig(show_timestamps=True, show_tool_results=True)
        msg = parse_message(sample_tool_result_message)
        blocks = msg.render(config)
        headers = [b for b in blocks if isinstance(b, HeaderBlock)]
        for h in headers:
            assert "USER" not in h.text
