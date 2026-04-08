"""Token usage statistics for Claude Code JSONL session logs.

This module provides a small, dependency-free API for aggregating
per-message token usage across a JSONL session log, with optional
filtering by line range, UUID range, and message type. It also
exposes :func:`compute_context_window_usage`, which mirrors the
context-window number Claude Code's UI displays.

It reuses :func:`claude_logs.models.parse_message` for parsing so the
shape of the returned objects stays consistent with the rest of the
package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import AssistantMessage, BaseMessage, ResultMessage, parse_message


# Core per-call token fields that contribute to the flat totals.
_KNOWN_TOKEN_FIELDS: frozenset[str] = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }
)

# Fields seen inside a `usage` block that are NOT summable token counts.
# They are informational, structural, or derivatives of other fields.
#   - cache_creation: nested dict whose ephemeral_*_input_tokens children
#     sum to cache_creation_input_tokens. Summing it would double-count.
#   - server_tool_use: {web_search_requests, web_fetch_requests} — counts,
#     not tokens.
#   - service_tier / inference_geo / speed: string metadata.
#   - iterations: list metadata.
_IGNORED_USAGE_FIELDS: frozenset[str] = frozenset(
    {
        "cache_creation",
        "server_tool_use",
        "service_tier",
        "inference_geo",
        "iterations",
        "speed",
    }
)


@dataclass
class TokenStatsFilter:
    """Filter criteria for :func:`compute_token_stats`.

    All criteria are conjunctive (AND). Unset / ``None`` fields are
    no-ops. ``types`` defaults to ``{"assistant"}`` because assistant
    messages carry the authoritative per-API-call usage; ``result``
    messages contain a grand total that would double-count if combined.

    Attributes:
        line_start: 1-indexed inclusive lower bound on JSONL line number.
        line_end: 1-indexed inclusive upper bound on JSONL line number.
        since_uuid: Inclusive start of a UUID window; the window opens
            at the first line whose ``uuid`` matches.
        until_uuid: Inclusive end of a UUID window; the window closes
            after the line whose ``uuid`` matches.
        types: Set of :attr:`BaseMessage.type` values to include. An
            empty frozenset means "include every type that carries a
            usage block".
    """

    line_start: int | None = None
    line_end: int | None = None
    since_uuid: str | None = None
    until_uuid: str | None = None
    types: frozenset[str] = field(default_factory=lambda: frozenset({"assistant"}))


@dataclass
class TokenStats:
    """Aggregated token usage for a JSONL session log (or a subset).

    The four core fields mirror the keys that appear inside Claude Code's
    ``usage`` blocks. ``unknown_token_fields`` captures any additional
    integer-valued usage keys that may appear in newer Claude Code
    versions without breaking older callers.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    # How many JSONL entries passed the filter (regardless of whether
    # they carried a `usage` block). Use this to sanity-check filters.
    messages_considered: int = 0

    # Subset of `messages_considered` that had a non-empty `usage` block.
    # Streamed-snapshot dedup happens later — this is the raw line count.
    messages_with_usage: int = 0

    # Count of distinct Anthropic API calls after de-duping by
    # ``message.id``. A single API call can be split across several
    # JSONL lines (one per content block: thinking / text / tool_use),
    # each carrying a snapshot of ``usage``; grouping by id and taking
    # field-wise maxima gives real per-call totals. Lines with no
    # ``message.id`` are treated as their own unique "call".
    unique_api_calls: int = 0

    # Additional integer-valued usage keys seen but not among the core
    # four. Summed across deduped API calls the same way as core fields.
    unknown_token_fields: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Sum of the four core token fields."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def _usage_for(msg: BaseMessage) -> dict[str, Any]:
    """Return the ``usage`` dict for a message, or ``{}`` if absent."""
    if isinstance(msg, AssistantMessage):
        return dict(msg.get_usage() or {})
    if isinstance(msg, ResultMessage):
        return dict(msg.usage or {})
    return {}


def _api_call_key(msg: BaseMessage, line_num: int) -> str:
    """Return a stable key used to de-duplicate streamed usage snapshots.

    Assistant messages carry ``message.id`` (the Anthropic API response
    id); lines sharing one id are snapshots of the same call and must
    be collapsed. Anything without an id falls back to the line number
    so each entry remains its own group.
    """
    if isinstance(msg, AssistantMessage):
        mid = msg.message.get("id", "") if isinstance(msg.message, dict) else ""
        if mid:
            return f"mid:{mid}"
    return f"line:{line_num}"


def compute_token_stats(
    path: str | Path,
    filter: TokenStatsFilter | None = None,
) -> TokenStats:
    """Aggregate token usage for a Claude Code JSONL session log.

    Args:
        path: Path to the JSONL file to read.
        filter: Optional :class:`TokenStatsFilter` restricting which
            messages contribute. Defaults to "assistant messages only,
            whole file".

    Returns:
        A :class:`TokenStats` with per-field sums and bookkeeping.

    Notes:
        * **Per-call semantics.** Values inside each ``usage`` block are
          per-call (not cumulative). Verified on real session logs by
          inspecting non-monotonic ``output_tokens`` across consecutive
          assistant entries.
        * **Stream-snapshot dedup.** A single Anthropic API call may be
          logged as several JSONL lines (one per content block), each
          with the same ``message.id`` but a progressively larger
          ``output_tokens`` snapshot. This function groups by
          ``message.id`` and takes field-wise maxima, avoiding the
          overcount a naive sum would produce.
        * **Result vs assistant.** ``result`` messages hold a session
          grand total that overlaps per-assistant usage. The default
          filter excludes them; pass ``types=frozenset({"result"})`` to
          read them specifically.
        * **Unknown fields.** Any int-valued ``usage`` key that is not
          one of the four core fields and not in the ignore list goes
          into :attr:`TokenStats.unknown_token_fields` rather than the
          core totals.
    """
    filt = filter or TokenStatsFilter()
    path = Path(path)

    line_start = filt.line_start if filt.line_start is not None else 1
    line_end = filt.line_end  # None means "no upper bound"
    since_uuid = filt.since_uuid
    until_uuid = filt.until_uuid
    types = filt.types

    # UUID window state. If no lower bound, the window is open from line 1.
    in_window = since_uuid is None

    # message.id / line-fallback -> {field_name: max_int_value}
    groups: dict[str, dict[str, int]] = {}
    messages_considered = 0
    messages_with_usage = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            if line_end is not None and line_num > line_end:
                break
            if line_num < line_start:
                continue

            raw = raw.strip()
            if not raw:
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg = parse_message(data)
            uuid = getattr(msg, "uuid", "") or ""

            # UUID windowing: `since_uuid` opens the window inclusively,
            # `until_uuid` closes it inclusively (applied after this
            # entry is processed).
            if not in_window:
                if since_uuid is not None and uuid == since_uuid:
                    in_window = True
                else:
                    continue

            # Type filter.
            if types and msg.type not in types:
                if until_uuid is not None and uuid == until_uuid:
                    in_window = False
                continue

            messages_considered += 1

            usage = _usage_for(msg)
            if usage:
                messages_with_usage += 1
                key = _api_call_key(msg, line_num)
                bucket = groups.setdefault(key, {})
                for field_name, value in usage.items():
                    if field_name in _IGNORED_USAGE_FIELDS:
                        continue
                    if not isinstance(value, int) or isinstance(value, bool):
                        continue
                    prev = bucket.get(field_name, 0)
                    if value > prev:
                        bucket[field_name] = value

            if until_uuid is not None and uuid == until_uuid:
                in_window = False

    stats = TokenStats(
        messages_considered=messages_considered,
        messages_with_usage=messages_with_usage,
        unique_api_calls=len(groups),
    )
    for bucket in groups.values():
        for field_name, value in bucket.items():
            if field_name == "input_tokens":
                stats.input_tokens += value
            elif field_name == "output_tokens":
                stats.output_tokens += value
            elif field_name == "cache_creation_input_tokens":
                stats.cache_creation_input_tokens += value
            elif field_name == "cache_read_input_tokens":
                stats.cache_read_input_tokens += value
            else:
                stats.unknown_token_fields[field_name] = (
                    stats.unknown_token_fields.get(field_name, 0) + value
                )

    return stats


@dataclass
class ContextWindowUsage:
    """Snapshot of how much context a session is currently holding.

    All four token fields are copied verbatim from the most recent
    non-sidechain assistant message's ``usage`` block. Compare
    :attr:`total` against the model's context window size (e.g.
    200_000 or 1_000_000) to derive a percentage matching what
    Claude Code's UI displays in its status line.

    Attributes:
        input_tokens: New input tokens for that turn (after the last
            cache breakpoint).
        cache_creation_input_tokens: Tokens written to the cache on
            that turn.
        cache_read_input_tokens: Tokens served from the cache on
            that turn.
        output_tokens: Generated tokens for that turn. Provided for
            reference; **not** included in :attr:`total` because the
            model's context window measures input footprint only.
        source_uuid: ``uuid`` of the JSONL line the snapshot came from.
        source_message_id: Anthropic API ``message.id`` of that turn.
        source_line: 1-indexed JSONL line number of the snapshot.
    """

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    source_uuid: str = ""
    source_message_id: str = ""
    source_line: int = 0

    @property
    def total(self) -> int:
        """``input + cache_creation + cache_read``.

        This is the formula Claude Code's UI uses to compute the
        percentage shown in its context-window indicator (see
        anthropics/claude-agent-sdk-typescript#66 for the official
        SDK formula). ``output_tokens`` is intentionally excluded.
        """
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def compute_context_window_usage(
    path: str | Path,
) -> ContextWindowUsage | None:
    """Return the live context window footprint for a session.

    Walks the JSONL once and returns the ``usage`` snapshot from the
    most recent non-sidechain assistant message. The resulting
    :attr:`ContextWindowUsage.total` mirrors the number Claude Code
    displays in its UI status line — i.e. how full the model's
    context window currently is, ready to be divided by
    200_000 / 1_000_000 to get a percentage.

    Args:
        path: Path to the JSONL session log.

    Returns:
        A :class:`ContextWindowUsage` snapshot, or ``None`` if the
        file contains no non-sidechain assistant messages.

    Why "non-sidechain": sidechain assistant messages are sub-agent
    (Task) calls. Their ``usage`` reflects the sub-agent's private
    context, not the main thread, so they would give a misleading
    picture of "what the main model is currently holding".

    .. note::
       **Live-streaming caveat.** When a session is *currently*
       running, the very last assistant line may have been flushed to
       the JSONL while the API response was still streaming. The
       ``input_tokens``, ``cache_creation_input_tokens`` and
       ``cache_read_input_tokens`` fields are stable (they're known
       up-front), so :attr:`total` is correct. But the writer can
       sometimes leave a stale ``output_tokens`` snapshot (e.g. ``1``
       or ``0``) on disk while the model is still emitting text;
       Claude Code's live UI sees the final value over the wire and
       counts the in-flight assistant turn against the *next* call's
       cache footprint, so its number can be ~few-hundred-to-few-
       thousand tokens higher than this function reports until the
       turn fully finalises in the JSONL. For a session that has
       fully settled (no streaming in progress) the two numbers
       agree.
    """
    path = Path(path)

    last_usage: dict[str, Any] | None = None
    last_uuid = ""
    last_message_id = ""
    last_line = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg = parse_message(data)
            if not isinstance(msg, AssistantMessage):
                continue
            if msg.isSidechain:
                continue

            usage = _usage_for(msg)
            if not usage:
                continue

            last_usage = usage
            last_uuid = msg.uuid or ""
            last_message_id = (
                msg.message.get("id", "") if isinstance(msg.message, dict) else ""
            )
            last_line = line_num

    if last_usage is None:
        return None

    return ContextWindowUsage(
        input_tokens=int(last_usage.get("input_tokens", 0) or 0),
        cache_creation_input_tokens=int(
            last_usage.get("cache_creation_input_tokens", 0) or 0
        ),
        cache_read_input_tokens=int(
            last_usage.get("cache_read_input_tokens", 0) or 0
        ),
        output_tokens=int(last_usage.get("output_tokens", 0) or 0),
        source_uuid=last_uuid,
        source_message_id=last_message_id,
        source_line=last_line,
    )
