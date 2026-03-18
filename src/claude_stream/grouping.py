"""Grouping logic for multi-file rendering.

This module handles --group-by parsing, file scouting (pass 1),
and interleaved rendering (pass 2).
"""

from __future__ import annotations

from datetime import datetime

from .models import GroupByConfig


def parse_group_by_spec(spec: str) -> GroupByConfig:
    """Parse a --group-by spec string into GroupByConfig.

    Args:
        spec: Comma-separated grouping keys (e.g., "project,time:%Y%m%d%H")

    Returns:
        Parsed GroupByConfig.

    Raises:
        ValueError: If the spec is invalid.
    """
    config = GroupByConfig()
    seen_project = False
    seen_time = False
    project_index = -1
    time_index = -1

    keys = [k.strip() for k in spec.split(",")]

    for i, key in enumerate(keys):
        if key == "project":
            if not seen_project:
                seen_project = True
                project_index = i
            config.by_project = True

        elif key.startswith("time:"):
            if seen_time:
                raise ValueError("only one time: group-by key is allowed")
            pattern = key[5:]  # strip "time:" prefix
            if not pattern:
                raise ValueError(
                    f"invalid group-by key: {key!r}. "
                    "Expected 'project' or 'time:<strftime>'"
                )
            # Validate strftime pattern: must contain at least one
            # directive that actually formats (output != input).
            # CPython on Linux silently passes unknown directives like %Q.
            test_result = datetime.now().strftime(pattern)
            if test_result == pattern:
                raise ValueError(
                    f"invalid strftime pattern in group-by: {pattern!r}"
                )
            seen_time = True
            time_index = i
            config.time_format = pattern

        else:
            raise ValueError(
                f"invalid group-by key: {key!r}. "
                "Expected 'project' or 'time:<strftime>'"
            )

    # Determine ordering
    if seen_project and seen_time:
        config.project_first = project_index < time_index
        if not config.project_first:
            raise ValueError(
                "time-then-project ordering is not yet supported. "
                "Use 'project,time:<fmt>' instead."
            )

    return config
