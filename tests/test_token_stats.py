"""Tests for the token_stats API."""

import json
from pathlib import Path

import pytest

from claude_logs import (
    ContextWindowUsage,
    TokenStats,
    TokenStatsFilter,
    compute_context_window_usage,
    compute_token_stats,
)

FIXTURE_V177 = Path(__file__).parent / "fixtures" / "v2.1.77" / "complete_session.jsonl"
FIXTURE_V175 = Path(__file__).parent / "fixtures" / "v2.1.75" / "complete_session.jsonl"


# -----------------------------------------------------------------------------
# Known fixture sums (v2.1.77). These mirror what's actually in the JSONL
# file so the tests fail loudly if a fixture is rewritten.
# Assistant lines (3, 4, 6, 8):
#   input_tokens:          510 + 605 + 710 + 820 = 2645
#   output_tokens:          95 +  48 +  75 +  55 =  273
#   cache_read_input_tokens:210 +   0 + 410 + 510 = 1130
# Result line (9): input=2800, output=450, cache_read=1400
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 1. Whole-file totals (default: assistants only)
# -----------------------------------------------------------------------------


class TestWholeFileTotals:
    def test_v177_default_sums_assistants_only(self):
        stats = compute_token_stats(FIXTURE_V177)
        assert stats.input_tokens == 2645
        assert stats.output_tokens == 273
        assert stats.cache_read_input_tokens == 1130
        assert stats.cache_creation_input_tokens == 0
        assert stats.messages_considered == 4
        assert stats.messages_with_usage == 4
        assert stats.unique_api_calls == 4

    def test_v177_string_path_also_works(self):
        stats = compute_token_stats(str(FIXTURE_V177))
        assert stats.input_tokens == 2645

    def test_v175_default(self):
        stats = compute_token_stats(FIXTURE_V175)
        # v2.1.75 fixture: assistants have in=500+600+700+800, out=100+50+80+60
        assert stats.input_tokens == 2600
        assert stats.output_tokens == 290
        assert stats.cache_read_input_tokens == 1100
        assert stats.messages_considered == 4

    def test_total_tokens_property(self):
        stats = compute_token_stats(FIXTURE_V177)
        assert stats.total_tokens == (
            stats.input_tokens
            + stats.output_tokens
            + stats.cache_creation_input_tokens
            + stats.cache_read_input_tokens
        )

    def test_no_unknown_fields_in_fixtures(self):
        stats = compute_token_stats(FIXTURE_V177)
        assert stats.unknown_token_fields == {}


# -----------------------------------------------------------------------------
# 2. Line range filter
# -----------------------------------------------------------------------------


class TestLineRangeFilter:
    def test_line_range_subset(self):
        # Lines 3..4 → first two assistants (510+605=1115, 95+48=143)
        stats = compute_token_stats(
            FIXTURE_V177, TokenStatsFilter(line_start=3, line_end=4)
        )
        assert stats.input_tokens == 1115
        assert stats.output_tokens == 143
        assert stats.messages_considered == 2

    def test_line_range_single_line(self):
        # Line 6 only → one assistant (in=710, out=75, cr=410)
        stats = compute_token_stats(
            FIXTURE_V177, TokenStatsFilter(line_start=6, line_end=6)
        )
        assert stats.input_tokens == 710
        assert stats.output_tokens == 75
        assert stats.cache_read_input_tokens == 410
        assert stats.messages_considered == 1

    def test_line_range_skips_non_assistant_lines(self):
        # Lines 1..3 include system + user + one assistant.
        # Only the assistant contributes.
        stats = compute_token_stats(
            FIXTURE_V177, TokenStatsFilter(line_start=1, line_end=3)
        )
        assert stats.input_tokens == 510
        assert stats.messages_considered == 1

    def test_line_start_only(self):
        # Lines 8..end → one assistant (line 8) + one result (line 9,
        # excluded by default type filter).
        stats = compute_token_stats(FIXTURE_V177, TokenStatsFilter(line_start=8))
        assert stats.input_tokens == 820
        assert stats.messages_considered == 1

    def test_line_end_only(self):
        stats = compute_token_stats(FIXTURE_V177, TokenStatsFilter(line_end=3))
        assert stats.input_tokens == 510
        assert stats.messages_considered == 1


# -----------------------------------------------------------------------------
# 3. UUID since/until filter
# -----------------------------------------------------------------------------


class TestUuidFilter:
    # From the fixture:
    #   line 3 uuid: d2c3b4a5-96e7-8901-dcba-987654321012
    #   line 4 uuid: c3b4a596-e7d8-9012-cbae-876543210123
    #   line 6 uuid: a596e7d8-c9b0-1234-ae9b-654321012345
    #   line 8 uuid: e7d8c9b0-a1f2-3456-eabc-432101234567

    def test_since_only(self):
        stats = compute_token_stats(
            FIXTURE_V177,
            TokenStatsFilter(since_uuid="a596e7d8-c9b0-1234-ae9b-654321012345"),
        )
        # Lines 6 and 8 → in=710+820=1530, out=75+55=130
        assert stats.input_tokens == 1530
        assert stats.output_tokens == 130
        assert stats.messages_considered == 2

    def test_until_only(self):
        stats = compute_token_stats(
            FIXTURE_V177,
            TokenStatsFilter(until_uuid="c3b4a596-e7d8-9012-cbae-876543210123"),
        )
        # Lines 1..4 → assistants on lines 3 and 4
        assert stats.input_tokens == 510 + 605
        assert stats.output_tokens == 95 + 48
        assert stats.messages_considered == 2

    def test_since_and_until(self):
        stats = compute_token_stats(
            FIXTURE_V177,
            TokenStatsFilter(
                since_uuid="c3b4a596-e7d8-9012-cbae-876543210123",
                until_uuid="a596e7d8-c9b0-1234-ae9b-654321012345",
            ),
        )
        # Lines 4 and 6 → in=605+710=1315, out=48+75=123
        assert stats.input_tokens == 1315
        assert stats.output_tokens == 123
        assert stats.messages_considered == 2

    def test_since_uuid_never_matches(self):
        stats = compute_token_stats(
            FIXTURE_V177, TokenStatsFilter(since_uuid="zzz-not-in-file")
        )
        assert stats.input_tokens == 0
        assert stats.messages_considered == 0


# -----------------------------------------------------------------------------
# 4. Message-type filter
# -----------------------------------------------------------------------------


class TestTypeFilter:
    def test_assistant_only_explicit(self):
        stats = compute_token_stats(
            FIXTURE_V177, TokenStatsFilter(types=frozenset({"assistant"}))
        )
        assert stats.input_tokens == 2645
        assert stats.messages_considered == 4

    def test_result_only(self):
        stats = compute_token_stats(
            FIXTURE_V177, TokenStatsFilter(types=frozenset({"result"}))
        )
        # From fixture: result usage = in=2800, out=450, cache_read=1400
        assert stats.input_tokens == 2800
        assert stats.output_tokens == 450
        assert stats.cache_read_input_tokens == 1400
        assert stats.messages_considered == 1

    def test_empty_types_means_all_types(self):
        stats = compute_token_stats(FIXTURE_V177, TokenStatsFilter(types=frozenset()))
        # Sums assistants AND result (expected double-count; documented).
        assert stats.input_tokens == 2645 + 2800
        assert stats.output_tokens == 273 + 450
        assert stats.cache_read_input_tokens == 1130 + 1400
        # All 9 lines pass the (empty) type filter, but only 5 carry usage
        # (4 assistants + 1 result). The remaining 4 are system/user lines.
        assert stats.messages_considered == 9
        assert stats.messages_with_usage == 5
        assert stats.unique_api_calls == 5


# -----------------------------------------------------------------------------
# 5. Graceful handling of missing/unknown fields
# -----------------------------------------------------------------------------


class TestMissingAndUnknownFields:
    def _write(self, tmp_path, messages):
        f = tmp_path / "session.jsonl"
        f.write_text("".join(json.dumps(m) + "\n" for m in messages))
        return f

    def test_assistant_with_no_usage_block(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {
                    "type": "assistant",
                    "uuid": "a-001",
                    "sessionId": "s",
                    "isSidechain": False,
                    "message": {
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hi"}],
                        # no usage
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "a-002",
                    "sessionId": "s",
                    "isSidechain": False,
                    "message": {
                        "id": "msg_2",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                },
            ],
        )
        stats = compute_token_stats(path)
        assert stats.input_tokens == 10
        assert stats.output_tokens == 5
        assert stats.messages_considered == 2
        assert stats.messages_with_usage == 1
        assert stats.unique_api_calls == 1

    def test_unknown_int_field_goes_to_unknown_bucket(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "sessionId": "s",
                    "isSidechain": False,
                    "message": {
                        "id": "msg_x",
                        "role": "assistant",
                        "content": [],
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "new_future_tokens": 7,
                        },
                    },
                }
            ],
        )
        stats = compute_token_stats(path)
        assert stats.input_tokens == 100
        assert stats.output_tokens == 20
        assert stats.unknown_token_fields == {"new_future_tokens": 7}

    def test_ignored_structural_fields_are_skipped(self, tmp_path):
        """Nested / non-token `usage` keys must not hit totals or unknowns."""
        path = self._write(
            tmp_path,
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "sessionId": "s",
                    "isSidechain": False,
                    "message": {
                        "id": "msg_y",
                        "role": "assistant",
                        "content": [],
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 4,
                            "cache_creation_input_tokens": 20,
                            "cache_creation": {
                                "ephemeral_5m_input_tokens": 0,
                                "ephemeral_1h_input_tokens": 20,
                            },
                            "server_tool_use": {
                                "web_search_requests": 0,
                                "web_fetch_requests": 0,
                            },
                            "service_tier": "standard",
                            "inference_geo": "not_available",
                        },
                    },
                }
            ],
        )
        stats = compute_token_stats(path)
        assert stats.input_tokens == 3
        assert stats.output_tokens == 4
        assert stats.cache_creation_input_tokens == 20
        # cache_creation nested dict must not sneak in as unknown
        assert stats.unknown_token_fields == {}

    def test_streamed_snapshot_dedup_by_message_id(self, tmp_path):
        """Four JSONL lines sharing one message.id represent one API call.

        The final snapshot holds the true output_tokens; earlier snapshots
        are partial. Dedup must collapse them to one call and take max.
        """
        base_meta = {"sessionId": "s", "isSidechain": False}
        common_usage = {
            "input_tokens": 1,
            "cache_creation_input_tokens": 459,
            "cache_read_input_tokens": 570778,
        }
        lines = []
        for i, out in enumerate([1, 1, 1, 4402], start=1):
            lines.append(
                {
                    "type": "assistant",
                    "uuid": f"a-{i}",
                    **base_meta,
                    "message": {
                        "id": "msg_shared",
                        "role": "assistant",
                        "content": [],
                        "usage": {**common_usage, "output_tokens": out},
                    },
                }
            )
        path = self._write(tmp_path, lines)

        stats = compute_token_stats(path)
        assert stats.messages_considered == 4
        assert stats.messages_with_usage == 4
        assert stats.unique_api_calls == 1
        assert stats.input_tokens == 1
        assert stats.output_tokens == 4402  # MAX not sum
        assert stats.cache_creation_input_tokens == 459
        assert stats.cache_read_input_tokens == 570778

    def test_malformed_json_line_is_skipped(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "sessionId": "s",
                    "isSidechain": False,
                    "message": {
                        "id": "m1",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 11, "output_tokens": 3},
                    },
                }
            )
            + "\n"
            + "this is not valid json\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a-2",
                    "sessionId": "s",
                    "isSidechain": False,
                    "message": {
                        "id": "m2",
                        "role": "assistant",
                        "content": [],
                        "usage": {"input_tokens": 22, "output_tokens": 5},
                    },
                }
            )
            + "\n"
        )
        stats = compute_token_stats(f)
        assert stats.input_tokens == 33
        assert stats.output_tokens == 8
        assert stats.unique_api_calls == 2

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        stats = compute_token_stats(f)
        assert stats == TokenStats()


# -----------------------------------------------------------------------------
# 6. compute_context_window_usage — "what's in the model's head right now"
# -----------------------------------------------------------------------------


class TestContextWindowUsage:
    def _write(self, tmp_path, messages):
        f = tmp_path / "session.jsonl"
        f.write_text("".join(json.dumps(m) + "\n" for m in messages))
        return f

    def _asst(self, line_id, msg_id, usage, *, sidechain=False):
        return {
            "type": "assistant",
            "uuid": line_id,
            "sessionId": "s",
            "isSidechain": sidechain,
            "message": {
                "id": msg_id,
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
                "usage": usage,
            },
        }

    def test_v177_fixture(self):
        # Last assistant on line 8: in=820, out=55, cache_read=510
        cw = compute_context_window_usage(FIXTURE_V177)
        assert cw is not None
        assert cw.input_tokens == 820
        assert cw.cache_read_input_tokens == 510
        assert cw.cache_creation_input_tokens == 0
        assert cw.output_tokens == 55
        # total = input + cache_creation + cache_read (output excluded)
        assert cw.total == 820 + 0 + 510
        assert cw.source_line == 8
        assert cw.source_uuid == "e7d8c9b0-a1f2-3456-eabc-432101234567"

    def test_string_path_works(self):
        cw = compute_context_window_usage(str(FIXTURE_V177))
        assert cw is not None and cw.input_tokens == 820

    def test_takes_last_assistant_not_just_last_line(self, tmp_path):
        # Result message after the last assistant must NOT shadow it.
        path = self._write(
            tmp_path,
            [
                self._asst(
                    "a-1",
                    "m1",
                    {
                        "input_tokens": 100,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 10,
                    },
                ),
                self._asst(
                    "a-2",
                    "m2",
                    {
                        "input_tokens": 200,
                        "cache_read_input_tokens": 9000,
                        "cache_creation_input_tokens": 300,
                        "output_tokens": 5,
                    },
                ),
                {
                    "type": "result",
                    "uuid": "r-1",
                    "sessionId": "s",
                    "subtype": "success",
                    "duration_ms": 1,
                    "num_turns": 1,
                    "total_cost_usd": 0.0,
                    "usage": {
                        "input_tokens": 999999,
                        "cache_read_input_tokens": 999999,
                        "cache_creation_input_tokens": 999999,
                        "output_tokens": 999999,
                    },
                },
            ],
        )
        cw = compute_context_window_usage(path)
        assert cw is not None
        assert cw.input_tokens == 200
        assert cw.cache_read_input_tokens == 9000
        assert cw.cache_creation_input_tokens == 300
        assert cw.total == 200 + 300 + 9000
        assert cw.source_uuid == "a-2"
        assert cw.source_message_id == "m2"

    def test_skips_sidechain_assistants(self, tmp_path):
        # The very last assistant entry is a sub-agent (sidechain). The
        # function should fall back to the previous main-thread assistant.
        path = self._write(
            tmp_path,
            [
                self._asst(
                    "main-1",
                    "m_main",
                    {
                        "input_tokens": 11,
                        "cache_read_input_tokens": 22,
                        "cache_creation_input_tokens": 33,
                        "output_tokens": 4,
                    },
                ),
                self._asst(
                    "sub-1",
                    "m_sub",
                    {
                        "input_tokens": 7777,
                        "cache_read_input_tokens": 8888,
                        "cache_creation_input_tokens": 9999,
                        "output_tokens": 1,
                    },
                    sidechain=True,
                ),
            ],
        )
        cw = compute_context_window_usage(path)
        assert cw is not None
        assert cw.source_uuid == "main-1"
        assert cw.input_tokens == 11
        assert cw.cache_read_input_tokens == 22
        assert cw.cache_creation_input_tokens == 33
        assert cw.total == 11 + 22 + 33

    def test_returns_none_when_no_assistant(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "sessionId": "s",
                    "userType": "external",
                    "message": {"role": "user", "content": "hi"},
                }
            ],
        )
        assert compute_context_window_usage(path) is None

    def test_returns_none_when_only_sidechain_assistants(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                self._asst(
                    "sub-1",
                    "m1",
                    {"input_tokens": 1, "output_tokens": 1},
                    sidechain=True,
                )
            ],
        )
        assert compute_context_window_usage(path) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert compute_context_window_usage(f) is None

    def test_handles_missing_cache_fields_gracefully(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                self._asst(
                    "a-1",
                    "m1",
                    {"input_tokens": 42, "output_tokens": 7},
                )
            ],
        )
        cw = compute_context_window_usage(path)
        assert cw is not None
        assert cw.input_tokens == 42
        assert cw.cache_creation_input_tokens == 0
        assert cw.cache_read_input_tokens == 0
        assert cw.output_tokens == 7
        assert cw.total == 42

    def test_total_excludes_output_tokens(self, tmp_path):
        path = self._write(
            tmp_path,
            [
                self._asst(
                    "a-1",
                    "m1",
                    {
                        "input_tokens": 1,
                        "cache_creation_input_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "output_tokens": 999,
                    },
                )
            ],
        )
        cw = compute_context_window_usage(path)
        assert cw is not None and cw.total == 6  # 1 + 2 + 3, not + 999

    def test_streaming_snapshots_pick_last_line(self, tmp_path):
        # Simulates the streaming-snapshot pattern: same message.id, four
        # lines, output_tokens grows. The function should return the final
        # snapshot. (input/cc/cr are stable so the result is the same
        # regardless of which snapshot we pick — but source_line should
        # be the *last* one for predictability.)
        common = {
            "input_tokens": 1,
            "cache_creation_input_tokens": 459,
            "cache_read_input_tokens": 570778,
        }
        path = self._write(
            tmp_path,
            [
                self._asst("a-1", "m_shared", {**common, "output_tokens": 1}),
                self._asst("a-2", "m_shared", {**common, "output_tokens": 1}),
                self._asst("a-3", "m_shared", {**common, "output_tokens": 1}),
                self._asst("a-4", "m_shared", {**common, "output_tokens": 4402}),
            ],
        )
        cw = compute_context_window_usage(path)
        assert cw is not None
        assert cw.source_line == 4
        assert cw.source_uuid == "a-4"
        assert cw.input_tokens == 1
        assert cw.cache_creation_input_tokens == 459
        assert cw.cache_read_input_tokens == 570778
        assert cw.output_tokens == 4402
        assert cw.total == 1 + 459 + 570778
