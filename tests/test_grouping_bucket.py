"""Tests for bucket key computation."""

from datetime import datetime, timezone

from claude_stream.grouping import compute_bucket_key


class TestComputeBucketKey:
    def test_hourly(self):
        dt = datetime(2026, 3, 17, 14, 23, 5, tzinfo=timezone.utc)
        local_dt = dt.astimezone()
        expected = local_dt.strftime("%Y%m%d%H")
        assert compute_bucket_key(dt, "%Y%m%d%H") == expected

    def test_daily(self):
        dt = datetime(2026, 3, 17, 14, 23, 5, tzinfo=timezone.utc)
        local_dt = dt.astimezone()
        expected = local_dt.strftime("%Y%m%d")
        assert compute_bucket_key(dt, "%Y%m%d") == expected

    def test_hour_of_day(self):
        dt = datetime(2026, 3, 17, 14, 23, 5, tzinfo=timezone.utc)
        local_dt = dt.astimezone()
        expected = local_dt.strftime("%H")
        assert compute_bucket_key(dt, "%H") == expected
